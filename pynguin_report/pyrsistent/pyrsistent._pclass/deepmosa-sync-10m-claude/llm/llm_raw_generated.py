####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_pclass_meta_new_sets_pclass_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    class MockField(_PField):
        def __init__(self):
            pass
    
    dct = {'field1': MockField(), 'field2': MockField()}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert '_pclass_fields' in result.__dict__
    assert 'field1' in result._pclass_fields
    assert 'field2' in result._pclass_fields


def test_pclass_meta_new_sets_pclass_invariants():
    from pyrsistent._pclass import PClassMeta
    
    def test_invariant(self):
        return True, None
    
    dct = {'__invariant__': test_invariant}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert '_pclass_invariants' in result.__dict__
    assert isinstance(result._pclass_invariants, tuple)
    assert len(result._pclass_invariants) > 0


def test_pclass_meta_new_sets_slots():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    class MockField(_PField):
        def __init__(self):
            pass
    
    dct = {'field1': MockField()}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert '__slots__' in result.__dict__
    assert '_pclass_frozen' in result.__slots__
    assert 'field1' in result.__slots__


def test_pclass_meta_new_adds_weakref_slot_for_top_level():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    dct = {}
    bases = (CheckedType,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert '__weakref__' in result.__slots__


def test_pclass_meta_new_no_weakref_slot_for_subclass():
    from pyrsistent._pclass import PClassMeta, PClass
    
    dct = {}
    bases = (PClass,)
    name = 'TestSubClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert '__weakref__' not in result.__slots__


def test_pclass_meta_new_removes_field_from_dct():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    class MockField(_PField):
        def __init__(self):
            pass
    
    dct = {'field1': MockField()}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert 'field1' not in result.__dict__
    assert 'field1' in result._pclass_fields


def test_pclass_meta_new_returns_type_instance():
    from pyrsistent._pclass import PClassMeta
    
    dct = {}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert isinstance(result, type)
    assert result.__name__ == name


# LLM-generated content at query #2
#--------------------------

```python
def test_set_new_key():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('key1', 'value1')
    assert evolver._pclass_evolver_data['key1'] == 'value1'
    assert 'key1' in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver


def test_set_existing_key_different_value():
    original = object()
    initial_dict = {'key1': 'old_value'}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('key1', 'new_value')
    assert evolver._pclass_evolver_data['key1'] == 'new_value'
    assert 'key1' in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver


def test_set_same_value_no_change():
    original = object()
    value_obj = 'same_value'
    initial_dict = {'key1': value_obj}
    evolver = _PClassEvolver(original, initial_dict)
    evolver._pclass_evolver_data_is_dirty = False
    result = evolver.set('key1', value_obj)
    assert evolver._pclass_evolver_data['key1'] == value_obj
    assert 'key1' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is False
    assert result is evolver


def test_set_multiple_keys():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('key1', 'value1')
    evolver.set('key2', 'value2')
    assert evolver._pclass_evolver_data['key1'] == 'value1'
    assert evolver._pclass_evolver_data['key2'] == 'value2'
    assert 'key1' in evolver._factory_fields
    assert 'key2' in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True


def test_set_returns_self_for_chaining():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    result1 = evolver.set('key1', 'value1')
    result2 = result1.set('key2', 'value2')
    assert result2 is evolver
    assert evolver._pclass_evolver_data['key1'] == 'value1'
    assert evolver._pclass_evolver_data['key2'] == 'value2'


def test_set_with_none_value():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('key1', None)
    assert evolver._pclass_evolver_data['key1'] is None
    assert 'key1' in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver


def test_set_overwrites_previous_value():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('key1', 'value2')
    evolver.set('key1', 'value3')
    assert evolver._pclass_evolver_data['key1'] == 'value3'
    assert 'key1' in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #3
#--------------------------

```python
def test_remove_existing_item():
    original = object()
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.remove('key1')
    
    assert result is evolver
    assert 'key1' not in evolver._pclass_evolver_data
    assert 'key1' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data == {'key2': 'value2'}


def test_remove_nonexistent_item():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    try:
        evolver.remove('nonexistent')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == 'nonexistent'


def test_remove_item_that_was_set():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('key2', 'value2')
    
    result = evolver.remove('key2')
    
    assert result is evolver
    assert 'key2' not in evolver._pclass_evolver_data
    assert 'key2' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True


def test_remove_multiple_items():
    original = object()
    initial_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.remove('key1')
    evolver.remove('key3')
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert 'key3' not in evolver._pclass_evolver_data
    assert 'key2' in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True


def test_remove_via_delitem():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    del evolver['key1']
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #4
#--------------------------

```python
def test_pclass_eq_same_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2


def test_pclass_eq_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    assert not (obj1 == obj2)


def test_pclass_eq_different_types():
    from pyrsistent import PClass, field
    
    class TestClass1(PClass):
        x = field()
    
    class TestClass2(PClass):
        x = field()
    
    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    assert obj1 != obj2


def test_pclass_eq_with_non_pclass():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = {"x": 1}
    assert obj1 != obj2


def test_pclass_eq_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1, y=5)
    assert obj1 == obj2


def test_pclass_eq_self():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    assert obj1 == obj1


def test_pclass_eq_empty_classes():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj1 = EmptyClass()
    obj2 = EmptyClass()
    assert obj1 == obj2


def test_pclass_eq_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj1 = TestClass(a=1, b=2, c=3, d=4)
    obj2 = TestClass(a=1, b=2, c=3, d=4)
    assert obj1 == obj2


def test_pclass_eq_one_field_different():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj1 = TestClass(a=1, b=2, c=3)
    obj2 = TestClass(a=1, b=2, c=999)
    assert not (obj1 == obj2)


# LLM-generated content at query #5
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


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        instance = TestClass(x="invalid")
        assert False, "Should have raised PTypeError"
    except Exception:
        pass


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields=set(), ignore_extra=True)
    assert instance.x == 1


def test_pclass_new_with_field_invariant_violation():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive_invariant(value):
        return (value > 0, 'must be positive')
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        instance = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'must be positive' in e.error_codes


def test_pclass_new_multiple_fields_with_mixed_initialization():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field(initial=5)
        c = field(factory=str)
    
    instance = TestClass(a=1, c=123)
    assert instance.a == 1
    assert instance.b == 5
    assert instance.c == "123"


# LLM-generated content at query #6
#--------------------------

```python
def test_is_pclass_returns_false_for_empty_bases():
    from pyrsistent._pclass import PClassMeta
    
    # Create a class using PClassMeta with empty bases (no parent classes)
    # This should result in _is_pclass(bases) evaluating to False
    class TestClass(metaclass=PClassMeta):
        pass
    
    # Verify that __weakref__ was NOT added to __slots__
    # (it should only be added when _is_pclass(bases) is True)
    assert '__weakref__' not in TestClass.__slots__
    assert '_pclass_frozen' in TestClass.__slots__


# LLM-generated content at query #7
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    # The predicate at line 2 (isinstance(other, self.__class__)) evaluates to True
    # when comparing two instances of the same PClass
    result = obj1 == obj2
    assert result is True


# LLM-generated content at query #8
#--------------------------

```python
def test_persistent_returns_original_when_not_dirty():
    original = type('MockPClass', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.persistent()
    
    assert result is original


def test_persistent_creates_new_instance_when_dirty():
    MockPClass = type('MockPClass', (), {'__init__': lambda self, _factory_fields=None, **kwargs: setattr(self, 'data', kwargs) or setattr(self, '_factory_fields', _factory_fields)})
    original = MockPClass()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.set('a', 10)
    result = evolver.persistent()
    
    assert result is not original
    assert result.__class__ is original.__class__


def test_persistent_passes_factory_fields_and_data():
    captured_args = {}
    
    def mock_init(self, _factory_fields=None, **kwargs):
        captured_args['_factory_fields'] = _factory_fields
        captured_args['kwargs'] = kwargs
    
    MockPClass = type('MockPClass', (), {'__init__': mock_init})
    original = MockPClass()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.set('a', 10)
    evolver.set('c', 3)
    result = evolver.persistent()
    
    assert captured_args['_factory_fields'] == {'a', 'c'}
    assert captured_args['kwargs'] == {'a': 10, 'b': 2, 'c': 3}


def test_persistent_with_removed_field():
    MockPClass = type('MockPClass', (), {'__init__': lambda self, _factory_fields=None, **kwargs: None})
    original = MockPClass()
    initial_dict = {'a': 1, 'b': 2, 'c': 3}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.remove('b')
    result = evolver.persistent()
    
    assert result is not original
    assert 'b' not in evolver._pclass_evolver_data


def test_persistent_multiple_calls_after_set():
    MockPClass = type('MockPClass', (), {'__init__': lambda self, _factory_fields=None, **kwargs: None})
    original = MockPClass()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.set('a', 5)
    result1 = evolver.persistent()
    result2 = evolver.persistent()
    
    assert result1 is result2


# LLM-generated content at query #9
#--------------------------

```python
def test_set_with_kwargs():
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


def test_set_with_positional_args():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
        y = field()
    
    a = AClass(x=1, y=2)
    a2 = a.set('x', 10)
    
    assert a.x == 1
    assert a2.x == 10
    assert a2.y == 2


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
        y = field()
        z = field()
    
    a = AClass(x=1, y=2, z=3)
    a2 = a.set(x=10, y=20)
    
    assert a2.x == 10
    assert a2.y == 20
    assert a2.z == 3


def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
    
    a = AClass(x=1)
    a2 = a.set(x=2)
    
    assert a is not a2
    assert isinstance(a2, AClass)


def test_set_preserves_original():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
        y = field()
    
    a = AClass(x=1, y=2)
    a2 = a.set(x=100)
    
    assert a.x == 1
    assert a.y == 2


def test_set_with_optional_fields():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
        y = field(initial=0)
    
    a = AClass(x=1)
    a2 = a.set(x=5)
    
    assert a2.x == 5
    assert a2.y == 0


def test_set_mixed_args_and_kwargs():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
        y = field()
    
    a = AClass(x=1, y=2)
    a2 = a.set('x', 10, y=20)
    
    assert a2.x == 10
    assert a2.y == 20


# LLM-generated content at query #10
#--------------------------

```python
def test_pclass_meta_weakref_slot_added_when_bases_is_pclass():
    from pyrsistent._pclass import PClassMeta, _is_pclass
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    
    # Create a mock base class that _is_pclass would return True for
    class MockPClassBase(metaclass=PClassMeta):
        pass
    
    # Verify the base is recognized as a pclass
    bases = (MockPClassBase,)
    assert _is_pclass(bases)
    
    # Create a new class using PClassMeta with a pclass base
    dct = {}
    name = 'TestPClass'
    
    # Call __new__ to test the predicate
    new_class = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    # Verify that __weakref__ was added to __slots__
    assert '__weakref__' in new_class.__slots__
    assert '_pclass_frozen' in new_class.__slots__


# LLM-generated content at query #11
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    instance = TestClass(x=5)
    result = instance.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert 'x' in result[1][1]
    assert result[1][1]['x'] == 5
    assert 'y' in result[1][1]
    assert result[1][1]['y'] == 10


def test_pclass_reduce_with_only_defined_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    result = instance.__reduce__()
    
    assert result[1][1] == {'a': 1, 'b': 2, 'c': 3}


def test_pclass_reduce_empty_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=None)
        y = field(initial=None)
    
    instance = TestClass()
    result = instance.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass


# LLM-generated content at query #12
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_invariant_errors_present():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    class TestClass(PClass):
        x = field()
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) > 0
        assert 'TestClass.y' in e.missing_fields


def test_pclass_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) > 0
        assert 'TestClass.y' in e.missing_fields


def test_pclass_predicate_at_line_25_true_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    class MyClass(PClass):
        required_field = field(mandatory=True)
        optional_field = field()
    
    exception_raised = False
    try:
        MyClass(optional_field="value")
    except InvariantException:
        exception_raised = True
    
    assert exception_raised is True


# LLM-generated content at query #13
#--------------------------

```python
def test_is_pclass_returns_false_for_empty_bases():
    from pyrsistent._pclass import PClassMeta
    
    # Create a class with empty bases using PClassMeta
    # When bases is empty, _is_pclass(bases) should return False
    dct = {'_pclass_fields': {}, '_pclass_invariants': ()}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', (), dct)
    
    # Verify that __weakref__ was NOT added to __slots__
    # (it should only be added when _is_pclass(bases) is True)
    assert '__weakref__' not in result.__slots__
    assert result.__slots__ == ('_pclass_frozen',)


# LLM-generated content at query #14
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_with_args():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set('x', 10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=10, y=20)
    
    assert result.x == 10
    assert result.y == 20
    assert result.z == 3
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3


def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=2)
    
    assert obj is not result
    assert isinstance(result, TestClass)


def test_set_preserves_all_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a=1, b=2, c=3)
    result = obj.set(b=20)
    
    assert result.a == 1
    assert result.b == 20
    assert result.c == 3


def test_set_with_single_arg_and_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
    
    obj = TestClass(name='original')
    result = obj.set('name', 'updated')
    
    assert result.name == 'updated'
    assert obj.name == 'original'


def test_set_with_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj = TestClass(x=1)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y is None


# LLM-generated content at query #15
#--------------------------

```python
def test_pclass_hash_same_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert hash(obj1) != hash(obj2)


def test_pclass_hash_with_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_hashable():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    hash_value = hash(obj)
    
    assert isinstance(hash_value, int)


def test_pclass_hash_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=2, y=3)
    
    hash_set = {obj1, obj2, obj3}
    
    assert len(hash_set) == 2


def test_pclass_hash_in_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    test_dict = {obj1: 'value1'}
    test_dict[obj2] = 'value2'
    
    assert len(test_dict) == 1
    assert test_dict[obj1] == 'value2'


def test_pclass_hash_single_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=5)
    obj2 = TestClass(x=5)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_with_string_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
        value = field()
    
    obj1 = TestClass(name='test', value='data')
    obj2 = TestClass(name='test', value='data')
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_with_none_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=None)
        y = field(initial=None)
    
    obj1 = TestClass()
    obj2 = TestClass()
    
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #16
#--------------------------

```python
def test_serialize_with_no_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        pass
    
    obj = SimpleClass()
    result = obj.serialize()
    assert result == {}


def test_serialize_with_single_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=42)
    result = obj.serialize()
    assert result == {'x': 42}


def test_serialize_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = SimpleClass(x=1, y='hello', z=3.14)
    result = obj.serialize()
    assert result == {'x': 1, 'y': 'hello', 'z': 3.14}


def test_serialize_with_missing_optional_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj = SimpleClass(x=42)
    result = obj.serialize()
    assert result == {'x': 42, 'y': None}


def test_serialize_with_initial_value():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = SimpleClass(y=20)
    result = obj.serialize()
    assert result == {'x': 10, 'y': 20}


def test_serialize_with_nested_pclass():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        a = field()
    
    class OuterClass(PClass):
        inner = field()
        b = field()
    
    inner_obj = InnerClass(a=5)
    outer_obj = OuterClass(inner=inner_obj, b=10)
    result = outer_obj.serialize()
    assert result['b'] == 10
    assert result['inner'] == inner_obj


def test_serialize_returns_dict():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=100)
    result = obj.serialize()
    assert isinstance(result, dict)


def test_serialize_does_not_modify_original():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=42)
    result = obj.serialize()
    result['x'] = 100
    assert obj.x == 42


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=42)
    result = obj.serialize(format='json')
    assert result == {'x': 42}


def test_serialize_with_boolean_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        flag1 = field()
        flag2 = field()
    
    obj = SimpleClass(flag1=True, flag2=False)
    result = obj.serialize()
    assert result == {'flag1': True, 'flag2': False}


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
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_new_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="string")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_invariant_failure():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive(val):
        return val > 0, "must be positive"
    
    class TestClass(PClass):
        x = field(invariant=positive)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'must be positive' in e.error_codes


def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def sum_positive(obj):
        return obj.x + obj.y > 0, "sum must be positive"
    
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = (sum_positive,)
    
    try:
        TestClass(x=-5, y=2)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'sum must be positive' in e.error_codes


def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_new_with_default_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=100)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 100
    assert instance.y == 5


# LLM-generated content at query #18
#--------------------------

```python
def test_serialize_with_no_fields():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    result = instance.serialize()
    assert result == {}


def test_serialize_with_simple_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    instance = SimpleClass(x=1, y="hello")
    result = instance.serialize()
    assert result == {'x': 1, 'y': "hello"}


def test_serialize_with_missing_optional_fields():
    from pyrsistent import PClass, field
    
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance = OptionalClass(x=42)
    result = instance.serialize()
    assert 'x' in result
    assert result['x'] == 42


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    class FormattedClass(PClass):
        name = field()
        value = field()
    
    instance = FormattedClass(name="test", value=100)
    result = instance.serialize(format="json")
    assert result['name'] == "test"
    assert result['value'] == 100


def test_serialize_returns_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
    
    instance = TestClass(a=1, b=2)
    result = instance.serialize()
    assert isinstance(result, dict)
    assert len(result) == 2


def test_serialize_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        field1 = field()
        field2 = field()
        field3 = field()
        field4 = field()
    
    instance = MultiFieldClass(field1="a", field2="b", field3="c", field4="d")
    result = instance.serialize()
    assert result == {'field1': 'a', 'field2': 'b', 'field3': 'c', 'field4': 'd'}


# LLM-generated content at query #19
#--------------------------

```python
def test_pclass_meta_new_with_pclass_bases():
    from pyrsistent._pclass import PClassMeta, _is_pclass
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    
    # Create a mock base class that would be considered a PClass
    class MockPClassBase(metaclass=PClassMeta):
        pass
    
    # Verify that _is_pclass returns True for bases containing a PClass
    bases = (MockPClassBase,)
    result = _is_pclass(bases)
    
    assert result is True


# LLM-generated content at query #20
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_invariant_errors_exist():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    # Mock the _check_and_set_attr to add an invariant error
    import pyrsistent._pclass as pclass_module
    original_check_and_set_attr = pclass_module._check_and_set_attr
    
    def mock_check_and_set_attr(cls, field_obj, name, value, result, invariant_errors):
        invariant_errors.append('test_error')
    
    pclass_module._check_and_set_attr = mock_check_and_set_attr
    
    try:
        exception_raised = False
        try:
            TestClass(x=1)
        except InvariantException:
            exception_raised = True
        
        assert exception_raised is True
    finally:
        pclass_module._check_and_set_attr = original_check_and_set_attr


def test_pclass_raises_invariant_exception_when_missing_fields_exist():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    exception_raised = False
    try:
        TestClass()
    except InvariantException:
        exception_raised = True
    
    assert exception_raised is True


def test_pclass_raises_invariant_exception_with_both_invariant_errors_and_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    exception_raised = False
    try:
        TestClass()
    except InvariantException as e:
        exception_raised = True
        assert len(e.missing_fields) == 2
    
    assert exception_raised is True


# LLM-generated content at query #21
#--------------------------

```python
def test_pclass_repr_single_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=1)
    assert repr(obj) == "SimpleClass(x=1)"


def test_pclass_repr_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = MultiFieldClass(x=1, y=2, z=3)
    result = repr(obj)
    assert "MultiFieldClass(" in result
    assert "x=1" in result
    assert "y=2" in result
    assert "z=3" in result


def test_pclass_repr_string_field():
    from pyrsistent import PClass, field
    
    class StringClass(PClass):
        name = field()
    
    obj = StringClass(name="test")
    assert repr(obj) == "StringClass(name='test')"


def test_pclass_repr_nested_pclass():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner_obj = InnerClass(value=42)
    outer_obj = OuterClass(inner=inner_obj)
    assert repr(outer_obj) == "OuterClass(inner=InnerClass(value=42))"


def test_pclass_repr_with_none():
    from pyrsistent import PClass, field
    
    class NullableClass(PClass):
        x = field()
    
    obj = NullableClass(x=None)
    assert repr(obj) == "NullableClass(x=None)"


def test_pclass_repr_with_list():
    from pyrsistent import PClass, field
    
    class ListClass(PClass):
        items = field()
    
    obj = ListClass(items=[1, 2, 3])
    assert repr(obj) == "ListClass(items=[1, 2, 3])"


def test_pclass_repr_empty_pclass():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    assert repr(obj) == "EmptyClass()"


def test_pclass_repr_optional_field_not_set():
    from pyrsistent import PClass, field
    
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj = OptionalClass(x=1)
    result = repr(obj)
    assert "OptionalClass(" in result
    assert "x=1" in result


# LLM-generated content at query #22
#--------------------------

```python
def test_pclass_meta_weakref_not_added_when_not_pclass():
    from pyrsistent._pclass import PClassMeta
    
    # Create a metaclass instance with bases that are not PClass instances
    # This should result in _is_pclass(bases) returning False
    name = 'TestClass'
    bases = (object,)
    dct = {'_pclass_fields': {}, '_pclass_invariants': ()}
    
    result = PClassMeta(name, bases, dct)
    
    # When _is_pclass(bases) is False, __weakref__ should NOT be added to __slots__
    assert '__weakref__' not in result.__slots__
    assert result.__slots__ == ('_pclass_frozen',)


# LLM-generated content at query #23
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.serialize()
    
    assert 'x' in result
    assert 'y' in result


# LLM-generated content at query #24
#--------------------------

```python
def test_repr_format():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y="hello")
    repr_str = repr(instance)
    
    assert "TestClass" in repr_str
    assert "x=1" in repr_str
    assert "y='hello'" in repr_str
    assert repr_str.startswith("TestClass(")
    assert repr_str.endswith(")")


# LLM-generated content at query #25
#--------------------------

```python
def test_repr_format():
    class TestPClass(PClass):
        x = field()
        y = field()
    
    obj = TestPClass(x=1, y="hello")
    repr_str = repr(obj)
    
    assert "TestPClass(" in repr_str
    assert "x=1" in repr_str
    assert "y='hello'" in repr_str
    assert repr_str.endswith(")")


# LLM-generated content at query #26
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.invariant_errors or 'TestClass.x' in e.missing_fields


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_becomes_frozen():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


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
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2


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


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(ignore_extra=True, x=1, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #28
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    # The predicate at line 7 is: `for name in self._pclass_fields`
    # This evaluates to True when _pclass_fields is iterable and non-empty
    assert hasattr(instance, '_pclass_fields')
    assert len(instance._pclass_fields) > 0
    
    # Verify iteration works
    field_names = []
    for name in instance._pclass_fields:
        field_names.append(name)
    
    assert 'x' in field_names
    assert 'y' in field_names
    assert len(field_names) == 2


# LLM-generated content at query #29
#--------------------------

```python
def test_pclass_new_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3
    assert hasattr(instance, '_pclass_frozen')
    assert instance._pclass_frozen is True


# LLM-generated content at query #30
#--------------------------

```python
def test_is_pclass_with_single_checked_type_base():
    from your_module import _is_pclass, CheckedType
    result = _is_pclass((CheckedType,))
    assert result is True


def test_is_pclass_with_multiple_bases():
    from your_module import _is_pclass, CheckedType
    result = _is_pclass((CheckedType, object))
    assert result is False


def test_is_pclass_with_no_bases():
    from your_module import _is_pclass
    result = _is_pclass(())
    assert result is False


def test_is_pclass_with_different_single_base():
    from your_module import _is_pclass
    result = _is_pclass((object,))
    assert result is False


def test_is_pclass_with_empty_tuple():
    from your_module import _is_pclass
    result = _is_pclass(())
    assert result is False


# LLM-generated content at query #31
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_missing_mandatory_fields():
    from pyrsistent import PClass, field
    from pyrsistent._precord import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.y' in e.missing_fields
        assert len(e.missing_fields) == 1


def test_pclass_raises_invariant_exception_when_field_invariant_fails():
    from pyrsistent import PClass, field
    from pyrsistent._precord import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, _factory_fields=set())
        assert False, "Expected InvariantException to be raised"
    except InvariantException:
        pass


def test_pclass_predicate_line_25_true_with_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._precord import InvariantException
    
    class TestClass(PClass):
        mandatory_field = field(mandatory=True)
    
    exception_raised = False
    try:
        TestClass()
    except InvariantException as e:
        exception_raised = True
        assert len(e.missing_fields) > 0
    
    assert exception_raised, "InvariantException should be raised when mandatory field is missing"


def test_pclass_predicate_line_25_true_with_invariant_errors():
    from pyrsistent import PClass, field
    from pyrsistent._precord import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    exception_raised = False
    try:
        obj = TestClass(x=1)
        invariant_errors = ['test_error']
        if invariant_errors:
            raise InvariantException(tuple(invariant_errors), (), 'Field invariant failed')
    except InvariantException as e:
        exception_raised = True
        assert len(e.invariant_errors) > 0
    
    assert exception_raised, "InvariantException should be raised when invariant errors exist"


# LLM-generated content at query #32
#--------------------------

```python
def test_repr_format():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y="hello")
    repr_str = repr(obj)
    
    assert repr_str.startswith("TestClass(")
    assert repr_str.endswith(")")
    assert "x=1" in repr_str
    assert "y='hello'" in repr_str
    assert repr_str == "TestClass(x=1, y='hello')" or repr_str == "TestClass(y='hello', x=1)"


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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        TestClass(y=5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'TestClass' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #34
#--------------------------

```python
def test_pclass_eq_same_class_same_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert obj1 == obj2


def test_pclass_eq_same_class_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert not (obj1 == obj2)


def test_pclass_eq_same_class_one_field_missing():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1, y=None)
    
    assert obj1 == obj2


def test_pclass_eq_different_classes():
    from pyrsistent import PClass, field
    
    class TestClass1(PClass):
        x = field()
    
    class TestClass2(PClass):
        x = field()
    
    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    
    assert not (obj1 == obj2)


def test_pclass_eq_with_non_pclass_object():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = {"x": 1}
    
    result = obj1 == obj2
    assert result is NotImplemented or result is False


def test_pclass_eq_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj1 = TestClass(a=1, b="test", c=[1, 2, 3])
    obj2 = TestClass(a=1, b="test", c=[1, 2, 3])
    
    assert obj1 == obj2


def test_pclass_eq_reflexive():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    
    assert obj == obj


def test_pclass_eq_symmetric():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert (obj1 == obj2) == (obj2 == obj1)


# LLM-generated content at query #35
#--------------------------

```python
def test_set_method_predicate_line_25():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    # Call set with positional args to trigger the condition at line 20
    result = instance.set('x', 10)
    
    # At line 25, the loop iterates over self._pclass_fields
    # The predicate "for key in self._pclass_fields" should evaluate to True
    # because _pclass_fields is a non-empty dict containing field definitions
    assert hasattr(instance, '_pclass_fields')
    assert len(instance._pclass_fields) > 0
    assert result.x == 10
    assert result.y == 2
    assert result.z == 3


# LLM-generated content at query #36
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    assert obj._pclass_frozen == True


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=42)
        y = field()
    
    obj = TestClass(y=2)
    assert obj.x == 42
    assert obj.y == 2


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        obj = TestClass(x="not an int")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    obj = TestClass(x="42")
    assert obj.x == 42


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field, invariant
    
    def positive(val):
        return val > 0, "Must be positive"
    
    class TestClass(PClass):
        x = field(invariant=positive)
    
    try:
        obj = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_pclass_new_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a=1, b=2, c=3)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3


def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, ignore_extra=True, y=2)
    assert obj.x == 1
    assert not hasattr(obj, 'y')


def test_pclass_new_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert obj._pclass_frozen == True


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    
    def sum_invariant(obj):
        return obj.x + obj.y > 0, "Sum must be positive"
    
    class TestClass(PClass):
        __invariant__ = sum_invariant
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_new_with_failing_global_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def sum_invariant(obj):
        return obj.x + obj.y > 0, "Sum must be positive"
    
    class TestClass(PClass):
        __invariant__ = sum_invariant
        x = field()
        y = field()
    
    try:
        obj = TestClass(x=-5, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #37
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
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_partial_initialization():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    instance = TestClass(x=10)
    assert instance.x == 10
    assert instance.y == 5


# LLM-generated content at query #38
#--------------------------

```python
def test_hash_returns_consistent_hash_for_same_pclass():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert hash1 == hash2
    assert isinstance(hash1, int)


def test_hash_differs_for_different_pclass_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert hash1 != hash2


def test_hash_pclass_with_single_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        value = field()
    
    instance = SimpleClass(value=42)
    hash_value = hash(instance)
    
    assert isinstance(hash_value, int)


def test_hash_pclass_with_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1, y=None)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert hash1 == hash2


def test_hash_pclass_hashable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    instance3 = TestClass(x=2)
    
    hash_set = {instance1, instance2, instance3}
    
    assert len(hash_set) >= 2
    assert hash(instance1) in [hash(item) for item in hash_set]


def test_hash_pclass_hashable_in_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    
    test_dict = {instance1: 'value1', instance2: 'value2'}
    
    assert test_dict[instance1] == 'value1'
    assert test_dict[instance2] == 'value2'


# LLM-generated content at query #39
#--------------------------

```python
def test_reduce_returns_restore_pickle_and_class_data():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


# LLM-generated content at query #40
#--------------------------

```python
def test_pclass_fields_iteration():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    fields_dict = TestClass._pclass_fields
    assert len(fields_dict) > 0
    assert 'x' in fields_dict
    assert 'y' in fields_dict
    
    count = 0
    for name, field_obj in fields_dict.items():
        count += 1
        assert name in ['x', 'y']
        assert hasattr(field_obj, 'factory')
    
    assert count == 2
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #41
#--------------------------

```python
def test_pclass_new_basic_field_assignment():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
    
    instance = TestClass()
    assert instance.x == 10


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_new_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'not among the specified fields' in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert instance._pclass_frozen is True


def test_pclass_new_cannot_modify_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_with_type_checking():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="not_an_int")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive_invariant(value):
        if value > 0:
            return True, None
        return False, "must_be_positive"
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert "must_be_positive" in e.error_codes


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def global_check(obj):
        if obj.x > obj.y:
            return True, None
        return False, "x_must_be_greater_than_y"
    
    class TestClass(PClass):
        __invariants__ = (global_check,)
        x = field()
        y = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert "x_must_be_greater_than_y" in e.error_codes


def test_pclass_new_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_new_partial_field_initialization():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=20)
    
    instance = TestClass(x=10)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_new_with_factory_fields_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
        y = field()
    
    instance = TestClass(x="42", y=100, _factory_fields={'x'})
    assert instance.x == 42
    assert instance.y == 100


def test_pclass_new_ignore_extra_false():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, extra=2, ignore_extra=False)
        assert False, "Should raise AttributeError"
    except AttributeError:
        pass


def test_pclass_new_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


def test_pclass_new_empty_pclass():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


# LLM-generated content at query #42
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
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


def test_pclass_constructor_with_default_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='42')
    assert instance.x == 42


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


# LLM-generated content at query #43
#--------------------------

```python
def test_set_method_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #44
#--------------------------

```python
def test_reduce_returns_tuple_with_restore_pickle_and_class_data():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1]['x'] == 1
    assert result[1][1]['y'] == 2


# LLM-generated content at query #45
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        TestClass(y=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields
        assert len(e.missing_fields) == 1


def test_pclass_raises_invariant_exception_when_field_invariant_fails():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    # Create a field with an invariant that always fails
    test_field = field()
    test_field.invariant = lambda x: (False, "invariant_error")
    TestClass._pclass_fields['x'] = test_field
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.error_codes) > 0


def test_pclass_raises_invariant_exception_with_both_errors_and_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields
        assert 'TestClass.y' in e.missing_fields
        assert len(e.missing_fields) == 2


# LLM-generated content at query #46
#--------------------------

```python
def test_pclass_meta_new_with_pclass_bases():
    from pyrsistent._pclass import PClassMeta, _is_pclass
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    
    # Create a mock base class that _is_pclass would return True for
    class MockPClassBase(metaclass=PClassMeta):
        pass
    
    # Verify that _is_pclass returns True for bases containing PClass
    bases = (MockPClassBase,)
    result = _is_pclass(bases)
    
    assert result is True


# LLM-generated content at query #47
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
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_multiple_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5)
    assert instance.x == 5


# LLM-generated content at query #48
#--------------------------

```python
def test_pclass_meta_new_with_single_checkedtype_base():
    from pyrsistent._pclass import PClassMeta, CheckedType
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self):
            self.initial = None
            self.factory = None
            self.invariant = None
    
    dct = {
        'field1': TestField(),
        'field2': TestField(),
        '__invariant__': lambda self: (True, None)
    }
    bases = (CheckedType,)
    name = 'TestPClass'
    
    result = PClassMeta(name, bases, dct)
    
    assert result.__name__ == 'TestPClass'
    assert hasattr(result, '_pclass_fields')
    assert 'field1' in result._pclass_fields
    assert 'field2' in result._pclass_fields
    assert hasattr(result, '_pclass_invariants')
    assert hasattr(result, '__slots__')
    assert '_pclass_frozen' in result.__slots__
    assert 'field1' in result.__slots__
    assert 'field2' in result.__slots__
    assert '__weakref__' in result.__slots__


def test_pclass_meta_new_with_multiple_bases():
    from pyrsistent._pclass import PClassMeta, CheckedType
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self):
            self.initial = None
            self.factory = None
            self.invariant = None
    
    class ParentPClass(CheckedType, metaclass=PClassMeta):
        parent_field = TestField()
    
    dct = {
        'child_field': TestField(),
    }
    bases = (ParentPClass,)
    name = 'ChildPClass'
    
    result = PClassMeta(name, bases, dct)
    
    assert result.__name__ == 'ChildPClass'
    assert hasattr(result, '_pclass_fields')
    assert 'child_field' in result._pclass_fields
    assert '__weakref__' not in result.__slots__


def test_pclass_meta_new_fields_removed_from_dct():
    from pyrsistent._pclass import PClassMeta, CheckedType
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self):
            self.initial = None
            self.factory = None
            self.invariant = None
    
    dct = {
        'field1': TestField(),
    }
    bases = (CheckedType,)
    name = 'TestPClass'
    
    result = PClassMeta(name, bases, dct)
    
    assert 'field1' not in dct
    assert hasattr(result, '_pclass_fields')
    assert 'field1' in result._pclass_fields


def test_pclass_meta_new_invariants_stored():
    from pyrsistent._pclass import PClassMeta, CheckedType
    
    def test_invariant(self):
        return (True, None)
    
    dct = {
        '__invariant__': test_invariant,
    }
    bases = (CheckedType,)
    name = 'TestPClass'
    
    result = PClassMeta(name, bases, dct)
    
    assert hasattr(result, '_pclass_invariants')
    assert len(result._pclass_invariants) > 0


def test_pclass_meta_new_slots_includes_fields_and_frozen():
    from pyrsistent._pclass import PClassMeta, CheckedType
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self):
            self.initial = None
            self.factory = None
            self.invariant = None
    
    dct = {
        'field_a': TestField(),
        'field_b': TestField(),
    }
    bases = (CheckedType,)
    name = 'TestPClass'
    
    result = PClassMeta(name, bases, dct)
    
    assert '_pclass_frozen' in result.__slots__
    assert 'field_a' in result.__slots__
    assert 'field_b' in result.__slots__


# LLM-generated content at query #49
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    # The predicate at line 2 (isinstance(other, self.__class__)) should evaluate to True
    # when comparing two instances of the same PClass
    result = obj1 == obj2
    assert result is True


# LLM-generated content at query #50
#--------------------------

```python
def test_hash_basic():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert hash(obj1) == hash(obj2)


def test_hash_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert hash(obj1) != hash(obj2)


def test_hash_single_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=5)
    obj2 = TestClass(x=5)
    
    assert hash(obj1) == hash(obj2)


def test_hash_with_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert hash(obj1) == hash(obj2)


def test_hash_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    obj3 = TestClass(x=2)
    
    hash_set = {obj1, obj2, obj3}
    assert len(hash_set) == 2


def test_hash_in_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    test_dict = {obj1: 'value1'}
    test_dict[obj2] = 'value2'
    
    assert len(test_dict) == 1
    assert test_dict[obj1] == 'value2'


def test_hash_with_multiple_fields_different_order():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj1 = TestClass(a=1, b=2, c=3)
    obj2 = TestClass(c=3, b=2, a=1)
    
    assert hash(obj1) == hash(obj2)


def test_hash_consistency():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=42)
    hash1 = hash(obj)
    hash2 = hash(obj)
    
    assert hash1 == hash2


def test_hash_with_string_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
        value = field()
    
    obj1 = TestClass(name='test', value='data')
    obj2 = TestClass(name='test', value='data')
    
    assert hash(obj1) == hash(obj2)


def test_hash_with_none_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=None)
    obj2 = TestClass(x=None)
    
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #51
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj = TestClass(x=1)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert 'x' in reduce_result[1][1]
    assert 'y' in reduce_result[1][1]


def test_pclass_reduce_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field(initial=20)
    
    obj = TestClass()
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 10, 'y': 20}


def test_pclass_reduce_single_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=42)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[1][1] == {'x': 42}


# LLM-generated content at query #52
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5)
    assert instance.x == 5


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_no_fields():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #53
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields=set(['x']), x=5)
    assert instance.x == 5


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


# LLM-generated content at query #54
#--------------------------

```python
def test_hash_returns_consistent_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert isinstance(hash1, int)
    assert hash1 == hash2
    assert hash(instance1) == hash(instance1)


def test_hash_different_for_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    
    assert hash(instance1) != hash(instance2)


def test_hash_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=10, b=20, c=30)
    hash_value = hash(instance)
    
    assert isinstance(hash_value, int)
    assert hash(instance) == hash_value


def test_hash_with_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
        y = field(initial=10)
    
    instance1 = TestClass()
    instance2 = TestClass()
    
    assert hash(instance1) == hash(instance2)


# LLM-generated content at query #55
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._checked_types import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42", _factory_fields={'x'})
    assert instance.x == 42


def test_pclass_constructor_without_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x=42, _factory_fields=set())
    assert instance.x == 42


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #56
#--------------------------

```python
def test_pclass_invariant_errors_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return False, "Test invariant failed"
    
    class TestPClass(PClass):
        x = field()
        __invariants__ = (failing_invariant,)
    
    try:
        TestPClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert "Test invariant failed" in e.invariant_errors


def test_pclass_missing_mandatory_field_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestPClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestPClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert "TestPClass.x" in e.missing_fields


def test_pclass_both_invariant_and_missing_field_errors():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return False, "Invariant error"
    
    class TestPClass(PClass):
        x = field(mandatory=True)
        y = field()
        __invariants__ = (failing_invariant,)
    
    try:
        TestPClass(y=2)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert "Invariant error" in e.invariant_errors
        assert "TestPClass.x" in e.missing_fields


# LLM-generated content at query #57
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
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_new_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="5")
    assert instance.x == 5


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_invariant_failure():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive(value):
        return (value > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert "must be positive" in e.error_codes


def test_pclass_new_with_ignore_extra_false():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, extra_field=2, ignore_extra=False)
        assert False, "Should raise AttributeError"
    except AttributeError:
        pass


def test_pclass_new_with_factory_fields_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
        y = field()
    
    instance = TestClass(x="5", y=10, _factory_fields={'x'})
    assert instance.x == 5
    assert instance.y == 10


def test_pclass_new_multiple_field_invariants_fail():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive(value):
        return (value > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive)
        y = field(invariant=positive)
    
    try:
        TestClass(x=-1, y=-2)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 2


def test_pclass_new_empty_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
        y = field(initial=10)
    
    instance = TestClass()
    assert instance.x == 5
    assert instance.y == 10


def test_pclass_new_with_type_checking():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="not an int")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_multiple_valid_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(type=(int, str))
    
    instance1 = TestClass(x=5)
    instance2 = TestClass(x="hello")
    assert instance1.x == 5
    assert instance2.x == "hello"


# LLM-generated content at query #58
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._precord import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_fields_with_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field(initial=1)
        b = field(initial=2)
        c = field()
    
    instance = TestClass(c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


# LLM-generated content at query #59
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 5)
    
    instance = TestClass()
    assert instance.x == 5


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "y" in str(e)
        assert "not among the specified fields" in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)


def test_pclass_constructor_with_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)
        assert "TestClass.y" in str(e)


# LLM-generated content at query #60
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class MockField:
        def __init__(self, field_type, invariant_result):
            self.type = field_type
            self._invariant_result = invariant_result
        
        def invariant(self, value):
            return self._invariant_result
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    field = MockField([int], (True, None))
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._exceptions import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "MockClass"
    
    class TestClass:
        pass
    
    result = TestClass()
    field = MockField([int])
    invariant_errors = []
    
    try:
        _check_and_set_attr(MockClass, field, "test_field", "invalid", result, invariant_errors)
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_failed_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = [int]
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    field = MockField()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == ["invariant_error_code"]


def test_check_and_set_attr_no_type_check():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    field = MockField()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "any_value", result, invariant_errors)
    
    assert result.test_field == "any_value"
    assert invariant_errors == []


def test_check_and_set_attr_multiple_valid_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = [int, str]
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    field = MockField()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "string_value", result, invariant_errors)
    
    assert result.test_field == "string_value"
    assert invariant_errors == []


# LLM-generated content at query #61
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    class MockClass:
        pass
    
    field = MockField()
    name = "test_field"
    value = "test_value"
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error_code"
    assert not hasattr(result, name)


# LLM-generated content at query #62
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_instances_independent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    
    assert instance1.x == 1
    assert instance2.x == 2


# LLM-generated content at query #63
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PClass, field
    
    class TestField:
        def __init__(self):
            self.type = (int,)
            self.invariant = lambda x: (False, "invariant_error_code")
    
    class TestClass(PClass):
        pass
    
    test_field = TestField()
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, test_field, "test_name", 42, result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error_code"
    assert not hasattr(result, "test_name")


# LLM-generated content at query #64
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self, invariant_result):
            self.type = None
            self.invariant_result = invariant_result
        
        def invariant(self, value):
            return self.invariant_result
    
    class MockClass:
        pass
    
    class MockResult:
        pass
    
    field = MockField((False, "error_code_1"))
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert invariant_errors == ["error_code_1"]
    assert not hasattr(result, "test_field")


# LLM-generated content at query #65
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    class MockClass:
        pass
    
    field = MockField()
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error_code"
    assert not hasattr(result, "test_field")


# LLM-generated content at query #66
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._exceptions import PTypeError
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        __name__ = "MockClass"
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    try:
        _check_and_set_attr(MockClass, field, "test_field", "invalid", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError:
        pass
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == []


def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        __name__ = "MockClass"
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (False, "value must be positive"))
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == ["value must be positive"]


def test_check_and_set_attr_multiple_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        __name__ = "MockClass"
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int, str], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", "hello", result, invariant_errors)
    
    assert result.test_field == "hello"
    assert invariant_errors == []


def test_check_and_set_attr_no_type_constraint():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        __name__ = "MockClass"
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField(None, lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", "any_value", result, invariant_errors)
    
    assert result.test_field == "any_value"
    assert invariant_errors == []


# LLM-generated content at query #67
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
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        instance = TestClass(y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass()
    assert not hasattr(instance, 'x')


def test_pclass_constructor_with_multiple_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
        y = field(initial=10)
        z = field()
    
    instance = TestClass(z=15)
    assert instance.x == 5
    assert instance.y == 10
    assert instance.z == 15


# LLM-generated content at query #68
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PyrsistentAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    assert instance is not None


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #69
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
        z = field(initial=3)
    
    instance = TestClass()
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3


def test_pclass_constructor_override_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
    
    instance = TestClass(x=20)
    assert instance.x == 20


def test_pclass_constructor_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


def test_pclass_constructor_preserves_type():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=42)
    assert isinstance(instance, TestClass)


# LLM-generated content at query #70
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    class MockClass:
        pass
    
    field = MockField()
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error_code"
    assert not hasattr(result, "test_field")


# LLM-generated content at query #71
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields=set(), x=1)
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(ignore_extra=True, x=1, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5)
    assert instance.x == 5


def test_pclass_constructor_multiple_instances_independent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=0)
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    assert instance1.x == 1
    assert instance2.x == 2


# LLM-generated content at query #72
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    obj = TestClass()
    assert obj.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)


def test_pclass_constructor_extra_fields_rejected():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(_factory_fields={'x'}, x=1)
    assert obj.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(ignore_extra=True, x=1, y=2)
    assert obj.x == 1
    assert not hasattr(obj, 'y')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert obj is not None


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)
        assert "TestClass.y" in str(e)


# LLM-generated content at query #73
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = (int,)
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    field = MockField()
    result = Result()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._exceptions import PTypeError
    
    class MockField:
        def __init__(self):
            self.type = (int,)
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    field = MockField()
    result = Result()
    invariant_errors = []
    
    try:
        _check_and_set_attr(MockClass, field, "test_field", "not_an_int", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = (int,)
        
        def invariant(self, value):
            return (False, "Value must be positive")
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    field = MockField()
    result = Result()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert invariant_errors == ["Value must be positive"]
    assert not hasattr(result, "test_field")


def test_check_and_set_attr_no_type_check():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    field = MockField()
    result = Result()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "any_value", result, invariant_errors)
    
    assert result.test_field == "any_value"
    assert invariant_errors == []


def test_check_and_set_attr_multiple_allowed_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = (int, str)
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    field = MockField()
    result = Result()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "string_value", result, invariant_errors)
    
    assert result.test_field == "string_value"
    assert invariant_errors == []


# LLM-generated content at query #74
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e.missing_fields)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        data = field()
    
    instance = TestClass(data={'a': 1})
    assert instance.data == {'a': 1}


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


def test_pclass_constructor_partial_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=0)
    
    instance = TestClass(x=5)
    assert instance.x == 5
    assert instance.y == 0


# LLM-generated content at query #75
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, extra_field=999)
    assert instance.x == 1


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x='5')
    assert instance.x == '5'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
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


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


def test_pclass_constructor_empty_pclass():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


# LLM-generated content at query #2
#--------------------------

```python
def test_pclass_meta_new_creates_slots():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    # Create a simple field
    test_field = _PField(initial=None, factory=None, invariant=None, initial_factory=None)
    
    # Create a class using PClassMeta
    dct = {'test_attr': test_field}
    bases = (object,)
    
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    
    assert hasattr(result, '__slots__')
    assert '_pclass_frozen' in result.__slots__


def test_pclass_meta_new_moves_pfield_to_pclass_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    # Create a simple field
    test_field = _PField(initial=None, factory=None, invariant=None, initial_factory=None)
    
    # Create a class using PClassMeta
    dct = {'my_field': test_field}
    bases = (object,)
    
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    
    assert 'my_field' in result._pclass_fields
    assert result._pclass_fields['my_field'] is test_field
    assert 'my_field' not in dct or dct.get('my_field') is None


def test_pclass_meta_new_with_weakref_slot():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    # Create a class that directly inherits from CheckedType (the base pclass case)
    dct = {}
    bases = (CheckedType,)
    
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    
    assert '__weakref__' in result.__slots__


def test_pclass_meta_new_without_weakref_slot():
    from pyrsistent._pclass import PClassMeta
    
    # Create a class that doesn't directly inherit from CheckedType
    dct = {}
    bases = (object,)
    
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    
    assert '__weakref__' not in result.__slots__


def test_pclass_meta_new_stores_invariants():
    from pyrsistent._pclass import PClassMeta
    
    def test_invariant(self):
        return True, "test"
    
    # Create a class with an invariant
    dct = {'__invariant__': test_invariant}
    bases = (object,)
    
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    
    assert hasattr(result, '_pclass_invariants')
    assert isinstance(result._pclass_invariants, tuple)
    assert len(result._pclass_invariants) > 0


def test_pclass_meta_new_multiple_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    # Create multiple fields
    field1 = _PField(initial=None, factory=None, invariant=None, initial_factory=None)
    field2 = _PField(initial=None, factory=None, invariant=None, initial_factory=None)
    
    # Create a class using PClassMeta
    dct = {'field1': field1, 'field2': field2}
    bases = (object,)
    
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    
    assert 'field1' in result._pclass_fields
    assert 'field2' in result._pclass_fields
    assert 'field1' in result.__slots__
    assert 'field2' in result.__slots__


# LLM-generated content at query #3
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_with_args():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set('x', 10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=10, y=20)
    
    assert result.x == 10
    assert result.y == 20
    assert result.z == 3
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3


def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=2)
    
    assert obj is not result
    assert isinstance(result, TestClass)


def test_set_with_missing_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    obj = TestClass(x=1)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 5


def test_set_preserves_all_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a=1, b=2, c=3)
    result = obj.set(b=20)
    
    assert result.a == 1
    assert result.b == 20
    assert result.c == 3


# LLM-generated content at query #4
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
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)


def test_pclass_new_with_type_checking():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        instance = TestClass(x="not an int")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="5")
    assert instance.x == 5


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive(value):
        return (value > 0, "Value must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive)
    
    try:
        instance = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_pclass_new_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert instance._pclass_frozen is True


def test_pclass_new_cannot_set_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_multiple_fields_and_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=100)
        z = field(initial=200)
    
    instance = TestClass(x=1)
    assert instance.x == 1
    assert instance.y == 100
    assert instance.z == 200


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    
    def sum_invariant(obj):
        return (obj.x + obj.y > 0, "Sum must be positive")
    
    class TestClass(PClass):
        __invariants__ = (sum_invariant,)
        x = field()
        y = field()
    
    try:
        instance = TestClass(x=-5, y=1)
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
        y = field()
    
    instance = TestClass(_factory_fields={'x'}, x="10", y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_new_without_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
        y = field()
    
    instance = TestClass(_factory_fields=set(), x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_new_with_ignore_extra_true():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(ignore_extra=True, x=1)
    assert instance.x == 1


# LLM-generated content at query #5
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj = TestClass(x=5)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[1][1] == {'x': 5, 'y': 10}


def test_pclass_reduce_with_no_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=100)
    reduce_result = obj.__reduce__()
    
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert isinstance(reduce_result[1][1], dict)


def test_pclass_reduce_preserves_all_attributes():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a='hello', b=42, c=[1, 2, 3])
    reduce_result = obj.__reduce__()
    
    assert reduce_result[1][1]['a'] == 'hello'
    assert reduce_result[1][1]['b'] == 42
    assert reduce_result[1][1]['c'] == [1, 2, 3]


# LLM-generated content at query #6
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_with_positional_args():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set('x', 5)
    
    assert result.x == 5
    assert result.y == 2
    assert obj.x == 1


def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=2)
    
    assert obj is not result
    assert isinstance(result, TestClass)


def test_set_preserves_unmodified_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert result.z == 3


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=10, y=20)
    
    assert result.x == 10
    assert result.y == 20
    assert result.z == 3


def test_set_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x='2')
    
    assert result.x == '2'


def test_set_original_unchanged():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    obj_x_before = obj.x
    obj_y_before = obj.y
    
    result = obj.set(x=100, y=200)
    
    assert obj.x == obj_x_before
    assert obj.y == obj_y_before
    assert result.x == 100
    assert result.y == 200


# LLM-generated content at query #7
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        TestClass(y=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields
        assert len(e.missing_fields) == 1


# LLM-generated content at query #8
#--------------------------

```python
def test_pclass_repr():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y='test')
    result = repr(obj)
    assert result == "SimpleClass(x=1, y='test')"


def test_pclass_repr_single_field():
    from pyrsistent import PClass, field
    
    class SingleField(PClass):
        name = field()
    
    obj = SingleField(name='hello')
    result = repr(obj)
    assert result == "SingleField(name='hello')"


def test_pclass_repr_empty():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        x = field(initial=None)
    
    obj = EmptyClass()
    result = repr(obj)
    assert "EmptyClass" in result


def test_pclass_repr_nested():
    from pyrsistent import PClass, field
    
    class Inner(PClass):
        value = field()
    
    class Outer(PClass):
        inner = field()
    
    inner_obj = Inner(value=42)
    outer_obj = Outer(inner=inner_obj)
    result = repr(outer_obj)
    assert "Outer" in result
    assert "Inner" in result
    assert "value=42" in result


def test_pclass_repr_with_special_characters():
    from pyrsistent import PClass, field
    
    class SpecialClass(PClass):
        text = field()
    
    obj = SpecialClass(text="hello'world\"test")
    result = repr(obj)
    assert "SpecialClass" in result
    assert "text=" in result


def test_pclass_repr_with_numeric_types():
    from pyrsistent import PClass, field
    
    class NumericClass(PClass):
        integer = field()
        floating = field()
    
    obj = NumericClass(integer=42, floating=3.14)
    result = repr(obj)
    assert "integer=42" in result
    assert "floating=3.14" in result


def test_pclass_repr_with_boolean():
    from pyrsistent import PClass, field
    
    class BoolClass(PClass):
        flag = field()
    
    obj = BoolClass(flag=True)
    result = repr(obj)
    assert "flag=True" in result


def test_pclass_repr_with_list():
    from pyrsistent import PClass, field
    
    class ListClass(PClass):
        items = field()
    
    obj = ListClass(items=[1, 2, 3])
    result = repr(obj)
    assert "ListClass" in result
    assert "items=" in result
    assert "[1, 2, 3]" in result


# LLM-generated content at query #9
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
        x = field(initial=lambda: [1, 2, 3])
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == [1, 2, 3]
    assert instance.y == 5


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    try:
        instance = TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "mandatory" in str(e).lower() or "invariant" in str(e).lower()


def test_pclass_constructor_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "frozen" in str(e).lower()


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass.create({'x': 1, 'z': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    assert isinstance(instance, EmptyClass)


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #10
#--------------------------

```python
def test_pclass_meta_new_weakref_slot_added_when_is_pclass_true():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    # Mock _is_pclass to return True
    import pyrsistent._pclass as pclass_module
    original_is_pclass = pclass_module._is_pclass
    pclass_module._is_pclass = lambda bases: True
    
    try:
        # Create a test dictionary with fields
        dct = {'_pclass_fields': {}}
        bases = (object,)
        name = 'TestPClass'
        
        # Call PClassMeta.__new__
        result = PClassMeta.__new__(PClassMeta, name, bases, dct)
        
        # Verify that __weakref__ is in __slots__
        assert '__weakref__' in result.__slots__
        assert '_pclass_frozen' in result.__slots__
    finally:
        # Restore original function
        pclass_module._is_pclass = original_is_pclass


def test_pclass_meta_new_weakref_slot_not_added_when_is_pclass_false():
    from pyrsistent._pclass import PClassMeta
    
    # Mock _is_pclass to return False
    import pyrsistent._pclass as pclass_module
    original_is_pclass = pclass_module._is_pclass
    pclass_module._is_pclass = lambda bases: False
    
    try:
        # Create a test dictionary with fields
        dct = {'_pclass_fields': {}}
        bases = (object,)
        name = 'TestPClass'
        
        # Call PClassMeta.__new__
        result = PClassMeta.__new__(PClassMeta, name, bases, dct)
        
        # Verify that __weakref__ is NOT in __slots__
        assert '__weakref__' not in result.__slots__
        assert '_pclass_frozen' in result.__slots__
    finally:
        # Restore original function
        pclass_module._is_pclass = original_is_pclass


# LLM-generated content at query #11
#--------------------------

```python
def test_pclass_invariant_errors_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    # Mock a field with an invariant that fails
    original_field = TestClass._pclass_fields['x']
    
    class FailingField:
        def __init__(self):
            self.type = None
            self.initial = None
            self.mandatory = False
            self.factory = lambda v: v
            self.serializer = None
        
        def invariant(self, owner, value):
            return False, "invariant_error"
    
    TestClass._pclass_fields['x'] = FailingField()
    
    try:
        instance = TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0 or len(e.missing_fields) > 0
    finally:
        TestClass._pclass_fields['x'] = original_field


def test_pclass_missing_mandatory_field_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) > 0
        assert 'TestClass.x' in e.missing_fields


# LLM-generated content at query #12
#--------------------------

```python
def test_pclass_meta_new_creates_slots_for_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(initial=None, factory=None, mandatory=False, invariant=None)
    field2 = _PField(initial=None, factory=None, mandatory=False, invariant=None)
    
    dct = {'field1': field1, 'field2': field2}
    bases = (CheckedType,)
    
    result = PClassMeta('TestClass', bases, dct)
    
    assert '_pclass_frozen' in result.__slots__
    assert 'field1' in result.__slots__
    assert 'field2' in result.__slots__


def test_pclass_meta_new_adds_weakref_for_direct_checkedtype_subclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    dct = {}
    bases = (CheckedType,)
    
    result = PClassMeta('TestClass', bases, dct)
    
    assert '__weakref__' in result.__slots__


def test_pclass_meta_new_no_weakref_for_indirect_subclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(initial=None, factory=None, mandatory=False, invariant=None)
    dct1 = {'field1': field1}
    bases1 = (CheckedType,)
    parent_class = PClassMeta('ParentClass', bases1, dct1)
    
    dct2 = {}
    bases2 = (parent_class,)
    
    result = PClassMeta('ChildClass', bases2, dct2)
    
    assert '__weakref__' not in result.__slots__


def test_pclass_meta_new_stores_invariants():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    def my_invariant(self):
        return True, None
    
    dct = {'__invariant__': my_invariant}
    bases = (CheckedType,)
    
    result = PClassMeta('TestClass', bases, dct)
    
    assert '_pclass_invariants' in result.__dict__
    assert len(result._pclass_invariants) > 0


def test_pclass_meta_new_moves_fields_to_pclass_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(initial=None, factory=None, mandatory=False, invariant=None)
    
    dct = {'field1': field1}
    bases = (CheckedType,)
    
    result = PClassMeta('TestClass', bases, dct)
    
    assert 'field1' in result._pclass_fields
    assert 'field1' not in result.__dict__ or result.__dict__.get('field1') is not field1


def test_pclass_meta_new_inherits_parent_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(initial=None, factory=None, mandatory=False, invariant=None)
    dct1 = {'field1': field1}
    bases1 = (CheckedType,)
    parent_class = PClassMeta('ParentClass', bases1, dct1)
    
    field2 = _PField(initial=None, factory=None, mandatory=False, invariant=None)
    dct2 = {'field2': field2}
    bases2 = (parent_class,)
    
    result = PClassMeta('ChildClass', bases2, dct2)
    
    assert 'field1' in result._pclass_fields
    assert 'field2' in result._pclass_fields


# LLM-generated content at query #13
#--------------------------

```python
def test_repr_format():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y="hello")
    repr_str = repr(obj)
    
    assert repr_str.startswith("TestClass(")
    assert repr_str.endswith(")")
    assert "x=1" in repr_str
    assert "y='hello'" in repr_str
    assert ", " in repr_str or repr_str == "TestClass(x=1, y='hello')" or repr_str == "TestClass(y='hello', x=1)"


# LLM-generated content at query #14
#--------------------------

```python
def test_is_pclass_with_single_checked_type_base():
    from your_module import _is_pclass, CheckedType
    result = _is_pclass((CheckedType,))
    assert result is True


def test_is_pclass_with_multiple_bases():
    from your_module import _is_pclass, CheckedType
    result = _is_pclass((CheckedType, object))
    assert result is False


def test_is_pclass_with_no_bases():
    from your_module import _is_pclass
    result = _is_pclass(())
    assert result is False


def test_is_pclass_with_different_single_base():
    from your_module import _is_pclass
    result = _is_pclass((object,))
    assert result is False


def test_is_pclass_with_empty_tuple():
    from your_module import _is_pclass
    result = _is_pclass(())
    assert result is False


# LLM-generated content at query #15
#--------------------------

```python
def test_pclass_meta_weakref_not_added_when_not_pclass_bases():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    # Create a metaclass instance with bases that are NOT PClass instances
    # This ensures _is_pclass(bases) returns False
    name = "TestClass"
    bases = (object,)
    dct = {'_pclass_fields': {}}
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    # Verify that __weakref__ was NOT added to __slots__
    assert '__weakref__' not in result.__slots__
    assert result.__slots__ == ('_pclass_frozen',)


# LLM-generated content at query #16
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field(type=str)
    
    test_field = TestClass.__dict__['name']
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, test_field, 'name', 'test_value', result, invariant_errors)
    
    assert result.name == 'test_value'
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        name = field(type=str)
    
    test_field = TestClass.__dict__['name']
    result = TestClass()
    invariant_errors = []
    
    try:
        _check_and_set_attr(TestClass, test_field, 'name', 123, result, invariant_errors)
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_failed_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    
    def positive_invariant(value):
        if value > 0:
            return True, None
        return False, "must_be_positive"
    
    class TestClass(PClass):
        count = field(type=int, invariant=positive_invariant)
    
    test_field = TestClass.__dict__['count']
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, test_field, 'count', -5, result, invariant_errors)
    
    assert invariant_errors == ["must_be_positive"]
    assert not hasattr(result, 'count') or result.count is None


def test_check_and_set_attr_passed_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    
    def positive_invariant(value):
        if value > 0:
            return True, None
        return False, "must_be_positive"
    
    class TestClass(PClass):
        count = field(type=int, invariant=positive_invariant)
    
    test_field = TestClass.__dict__['count']
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, test_field, 'count', 5, result, invariant_errors)
    
    assert result.count == 5
    assert invariant_errors == []


def test_check_and_set_attr_no_type_check():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        value = field()
    
    test_field = TestClass.__dict__['value']
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, test_field, 'value', 'any_value', result, invariant_errors)
    
    assert result.value == 'any_value'
    assert invariant_errors == []


# LLM-generated content at query #17
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_invariant_errors_present():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    TestClass._pclass_invariants = [lambda obj: (False, 'test_error')]
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'test_error' in e.error_codes
        assert e.message == 'Global invariant failed'


def test_pclass_raises_invariant_exception_when_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields
        assert e.message == 'Field invariant failed'


def test_pclass_raises_invariant_exception_when_both_errors_and_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    TestClass._pclass_invariants = [lambda obj: (False, 'invariant_error')]
    
    try:
        TestClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'invariant_error' in e.error_codes
        assert 'TestClass.x' in e.missing_fields
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #18
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


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_with_field_invariant_violation():
    from pyrsistent import PClass, field
    from pyrsistent import InvariantException
    
    def positive(value):
        return (value > 0, "Value must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive)
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_pclass_new_with_factory_fields_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
        y = field()
    
    instance = TestClass(x="10", y=20, _factory_fields={'x'})
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_new_with_ignore_extra_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True)
    assert instance.x == 1


def test_pclass_new_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field(initial=30)
        d = field()
    
    instance = TestClass(a=1, b=2, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 30
    assert instance.d == 4


def test_pclass_new_with_type_checking():
    from pyrsistent import PClass, field
    from pyrsistent import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="not an int")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_inherits_frozen_attribute():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert getattr(instance, '_pclass_frozen', False) is True


def test_pclass_new_with_no_fields():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_new_with_multiple_invariant_errors():
    from pyrsistent import PClass, field
    from pyrsistent import InvariantException
    
    def positive(value):
        return (value > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive)
        y = field(invariant=positive)
    
    try:
        TestClass(x=-1, y=-2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 2


# LLM-generated content at query #19
#--------------------------

```python
def test_pclass_hash_same_values_same_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_different_values_different_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert hash(obj1) != hash(obj2)


def test_pclass_hash_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_usable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    obj3 = TestClass(x=2)
    
    hash_set = {obj1, obj2, obj3}
    assert len(hash_set) == 2


def test_pclass_hash_usable_as_dict_key():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    d = {obj1: 'value1'}
    d[obj2] = 'value2'
    
    assert len(d) == 1
    assert d[obj1] == 'value2'


def test_pclass_hash_with_string_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
        description = field()
    
    obj1 = TestClass(name='test', description='desc')
    obj2 = TestClass(name='test', description='desc')
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_with_nested_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=[1, 2, 3], y={'a': 1})
    obj2 = TestClass(x=[1, 2, 3], y={'a': 1})
    
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #20
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    instance = TestClass(x=5)
    result = instance.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 5, 'y': 10}


def test_pclass_reduce_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    instance = TestClass()
    result = instance.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_only_assigned_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    result = instance.__reduce__()
    
    assert result[1][1] == {'x': 1, 'y': 2, 'z': 3}


# LLM-generated content at query #21
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
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_new_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PAttrError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'z' are not among the specified fields" in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field, PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="not an int")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_field_invariant_violation():
    from pyrsistent import PClass, field, InvariantException
    
    def positive_invariant(value):
        return value > 0, "must be positive"
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "must be positive" in e.error_codes


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, InvariantException
    
    def sum_invariant(obj):
        return obj.x + obj.y == 10, "sum must be 10"
    
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = (sum_invariant,)
    
    try:
        TestClass(x=3, y=4)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "sum must be 10" in e.error_codes


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    instance = TestClass(x={'a': 1})
    assert instance.x == pmap({'a': 1})


def test_pclass_new_with_multiple_type_options():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(type=(int, str))
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x="hello")
    assert instance1.x == 1
    assert instance2.x == "hello"


def test_pclass_new_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_new_with_ignore_extra_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields=set(), ignore_extra=True)
    assert instance.x == 1


# LLM-generated content at query #22
#--------------------------

```python
def test_remove_existing_item():
    original = object()
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.remove('key1')
    
    assert result is evolver
    assert 'key1' not in evolver._pclass_evolver_data
    assert 'key1' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data == {'key2': 'value2'}


def test_remove_nonexistent_item():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    try:
        evolver.remove('nonexistent_key')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == 'nonexistent_key'


def test_remove_after_set():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.set('key2', 'value2')
    result = evolver.remove('key2')
    
    assert result is evolver
    assert 'key2' not in evolver._pclass_evolver_data
    assert 'key2' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True


def test_remove_discards_from_factory_fields():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.set('key1', 'new_value')
    assert 'key1' in evolver._factory_fields
    
    evolver.remove('key1')
    
    assert 'key1' not in evolver._factory_fields
    assert 'key1' not in evolver._pclass_evolver_data


def test_remove_multiple_items():
    original = object()
    initial_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.remove('key1')
    evolver.remove('key2')
    
    assert evolver._pclass_evolver_data == {'key3': 'value3'}
    assert evolver._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #23
#--------------------------

```python
def test_pclass_repr_single_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=1)
    result = repr(obj)
    assert result == "SimpleClass(x=1)"


def test_pclass_repr_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = MultiFieldClass(x=1, y=2, z=3)
    result = repr(obj)
    assert "MultiFieldClass(" in result
    assert "x=1" in result
    assert "y=2" in result
    assert "z=3" in result


def test_pclass_repr_with_string_value():
    from pyrsistent import PClass, field
    
    class StringClass(PClass):
        name = field()
    
    obj = StringClass(name="test")
    result = repr(obj)
    assert result == "StringClass(name='test')"


def test_pclass_repr_with_none_value():
    from pyrsistent import PClass, field
    
    class NoneClass(PClass):
        value = field()
    
    obj = NoneClass(value=None)
    result = repr(obj)
    assert result == "NoneClass(value=None)"


def test_pclass_repr_with_list_value():
    from pyrsistent import PClass, field
    
    class ListClass(PClass):
        items = field()
    
    obj = ListClass(items=[1, 2, 3])
    result = repr(obj)
    assert result == "ListClass(items=[1, 2, 3])"


def test_pclass_repr_empty_pclass():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    result = repr(obj)
    assert result == "EmptyClass()"


def test_pclass_repr_with_initial_value_not_set():
    from pyrsistent import PClass, field
    
    class OptionalFieldClass(PClass):
        required = field()
        optional = field(initial=None)
    
    obj = OptionalFieldClass(required=5)
    result = repr(obj)
    assert "OptionalFieldClass(" in result
    assert "required=5" in result
    assert "optional=None" in result


# LLM-generated content at query #24
#--------------------------

```python
def test_serialize_with_no_fields():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    result = instance.serialize()
    assert result == {}


def test_serialize_with_simple_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    instance = SimpleClass(x=1, y="hello")
    result = instance.serialize()
    assert result == {'x': 1, 'y': "hello"}


def test_serialize_with_missing_optional_fields():
    from pyrsistent import PClass, field
    
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance = OptionalClass(x=42)
    result = instance.serialize()
    assert 'x' in result
    assert result['x'] == 42
    assert 'y' not in result or result['y'] is None


def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    
    def custom_serializer(value):
        return value * 2
    
    class CustomSerializerClass(PClass):
        x = field(serializer=custom_serializer)
    
    instance = CustomSerializerClass(x=5)
    result = instance.serialize()
    assert result == {'x': 10}


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    class FormatClass(PClass):
        x = field()
    
    instance = FormatClass(x=100)
    result = instance.serialize(format="json")
    assert result == {'x': 100}


def test_serialize_nested_pclass():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner_instance = InnerClass(value=42)
    outer_instance = OuterClass(inner=inner_instance)
    result = outer_instance.serialize()
    assert 'inner' in result


def test_serialize_with_multiple_fields_and_types():
    from pyrsistent import PClass, field
    
    class MultiTypeClass(PClass):
        int_field = field()
        str_field = field()
        list_field = field()
        dict_field = field()
    
    instance = MultiTypeClass(
        int_field=123,
        str_field="test",
        list_field=[1, 2, 3],
        dict_field={'key': 'value'}
    )
    result = instance.serialize()
    assert result['int_field'] == 123
    assert result['str_field'] == "test"
    assert result['list_field'] == [1, 2, 3]
    assert result['dict_field'] == {'key': 'value'}


# LLM-generated content at query #25
#--------------------------

```python
def test_pclass_hash_same_values_same_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_different_values_different_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert hash(obj1) != hash(obj2)


def test_pclass_hash_hashable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    obj3 = TestClass(x=2)
    
    hash_set = {obj1, obj2, obj3}
    assert len(hash_set) == 2


def test_pclass_hash_hashable_as_dict_key():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=2)
    
    hash_dict = {obj1: 'value1', obj2: 'value2'}
    assert hash_dict[obj1] == 'value1'
    assert hash_dict[obj2] == 'value2'


def test_pclass_hash_with_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_consistent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    hash1 = hash(obj)
    hash2 = hash(obj)
    
    assert hash1 == hash2


def test_pclass_hash_with_nested_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=(1, 2, 3))
    obj2 = TestClass(x=1, y=(1, 2, 3))
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_with_string_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
    
    obj1 = TestClass(name='test')
    obj2 = TestClass(name='test')
    
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #26
#--------------------------

```python
def test_pclass_repr_with_single_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=1)
    result = repr(obj)
    assert result == "SimpleClass(x=1)"


def test_pclass_repr_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        x = field()
        y = field()
    
    obj = MultiFieldClass(x=1, y=2)
    result = repr(obj)
    assert "MultiFieldClass(" in result
    assert "x=1" in result
    assert "y=2" in result


def test_pclass_repr_with_string_field():
    from pyrsistent import PClass, field
    
    class StringClass(PClass):
        name = field()
    
    obj = StringClass(name="test")
    result = repr(obj)
    assert result == "StringClass(name='test')"


def test_pclass_repr_with_nested_object():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner = InnerClass(value=42)
    outer = OuterClass(inner=inner)
    result = repr(outer)
    assert "OuterClass(inner=InnerClass(value=42))" == result


def test_pclass_repr_with_missing_optional_field():
    from pyrsistent import PClass, field
    
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj = OptionalClass(x=1)
    result = repr(obj)
    assert "x=1" in result


def test_pclass_repr_with_list_field():
    from pyrsistent import PClass, field
    
    class ListClass(PClass):
        items = field()
    
    obj = ListClass(items=[1, 2, 3])
    result = repr(obj)
    assert "ListClass(items=[1, 2, 3])" == result


def test_pclass_repr_with_dict_field():
    from pyrsistent import PClass, field
    
    class DictClass(PClass):
        data = field()
    
    obj = DictClass(data={"key": "value"})
    result = repr(obj)
    assert "DictClass(data=" in result
    assert "'key': 'value'" in result


def test_pclass_repr_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    result = repr(obj)
    assert result == "EmptyClass()"


def test_pclass_repr_with_boolean_field():
    from pyrsistent import PClass, field
    
    class BoolClass(PClass):
        flag = field()
    
    obj = BoolClass(flag=True)
    result = repr(obj)
    assert result == "BoolClass(flag=True)"


def test_pclass_repr_with_none_field():
    from pyrsistent import PClass, field
    
    class NoneClass(PClass):
        value = field()
    
    obj = NoneClass(value=None)
    result = repr(obj)
    assert result == "NoneClass(value=None)"


# LLM-generated content at query #27
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_with_args():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set('x', 10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=10, y=20)
    
    assert result.x == 10
    assert result.y == 20
    assert result.z == 3


def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=2)
    
    assert obj is not result
    assert isinstance(result, TestClass)


def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a=1, b=2, c=3)
    result = obj.set(b=20)
    
    assert result.a == 1
    assert result.b == 20
    assert result.c == 3


def test_set_with_string_key_and_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
    
    obj = TestClass(name='old')
    result = obj.set('name', 'new')
    
    assert result.name == 'new'
    assert obj.name == 'old'


def test_set_with_complex_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        items = field()
    
    obj = TestClass(items=[1, 2, 3])
    result = obj.set(items=[4, 5, 6])
    
    assert result.items == [4, 5, 6]
    assert obj.items == [1, 2, 3]


# LLM-generated content at query #28
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class MockField:
        def __init__(self, field_type=None, invariant_result=(True, None)):
            self.type = field_type
            self._invariant_result = invariant_result
        
        def invariant(self, value):
            return self._invariant_result
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    result = Result()
    field = MockField(field_type=[int], invariant_result=(True, None))
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._exceptions import PTypeError
    
    class MockField:
        def __init__(self, field_type=None):
            self.type = field_type
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    result = Result()
    field = MockField(field_type=[int])
    invariant_errors = []
    
    try:
        _check_and_set_attr(MockClass, field, "test_field", "not_an_int", result, invariant_errors)
        assert False, "Expected PTypeError to be raised"
    except PTypeError:
        pass


def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type=None):
            self.type = field_type
        
        def invariant(self, value):
            return (False, "value_too_small")
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    result = Result()
    field = MockField(field_type=[int])
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == ["value_too_small"]


def test_check_and_set_attr_no_type_check():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    result = Result()
    field = MockField()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "any_value", result, invariant_errors)
    
    assert result.test_field == "any_value"
    assert invariant_errors == []


def test_check_and_set_attr_multiple_valid_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = [int, str]
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class Result:
        pass
    
    result = Result()
    field = MockField()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "string_value", result, invariant_errors)
    
    assert result.test_field == "string_value"
    assert invariant_errors == []


# LLM-generated content at query #29
#--------------------------

```python
def test_pclass_invariant_errors_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field()
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.y' in e.missing_fields
        assert e.message == 'Field invariant failed'


def test_pclass_invariant_errors_with_invariant_function():
    from pyrsistent import PClass, field, InvariantException
    
    def check_positive(obj):
        if obj.x < 0:
            return False, "x must be positive"
        return True, None
    
    class TestClass(PClass):
        __invariants__ = (check_positive,)
        x = field()
    
    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0
        assert e.message == 'Field invariant failed'


def test_pclass_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        name = field(mandatory=True)
        age = field()
    
    try:
        TestClass(age=25)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.name' in e.missing_fields
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #30
#--------------------------

```python
def test_serialize_with_no_fields():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    result = obj.serialize()
    assert result == {}


def test_serialize_with_simple_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y="hello")
    result = obj.serialize()
    assert result == {'x': 1, 'y': "hello"}


def test_serialize_with_missing_optional_fields():
    from pyrsistent import PClass, field
    
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj = OptionalClass(x=5)
    result = obj.serialize()
    assert result == {'x': 5, 'y': None}


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    class FormattedClass(PClass):
        value = field()
    
    obj = FormattedClass(value=42)
    result = obj.serialize(format='json')
    assert result == {'value': 42}


def test_serialize_with_nested_pclass():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        inner_value = field()
    
    class OuterClass(PClass):
        outer_value = field()
        nested = field()
    
    inner = InnerClass(inner_value=10)
    outer = OuterClass(outer_value=20, nested=inner)
    result = outer.serialize()
    assert 'outer_value' in result
    assert 'nested' in result
    assert result['outer_value'] == 20


def test_serialize_excludes_missing_values():
    from pyrsistent import PClass, field
    
    class SparseClass(PClass):
        a = field(initial=100)
        b = field(initial=None)
        c = field()
    
    obj = SparseClass(c=30)
    result = obj.serialize()
    assert 'a' in result
    assert 'b' in result
    assert 'c' in result
    assert result['a'] == 100
    assert result['b'] is None
    assert result['c'] == 30


def test_serialize_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        field1 = field()
        field2 = field()
        field3 = field()
        field4 = field()
    
    obj = MultiFieldClass(field1=1, field2=2, field3=3, field4=4)
    result = obj.serialize()
    assert result == {'field1': 1, 'field2': 2, 'field3': 3, 'field4': 4}


def test_serialize_preserves_field_values():
    from pyrsistent import PClass, field
    
    class DataClass(PClass):
        name = field()
        count = field()
        active = field()
    
    obj = DataClass(name="test", count=42, active=True)
    result = obj.serialize()
    assert result['name'] == "test"
    assert result['count'] == 42
    assert result['active'] is True


# LLM-generated content at query #31
#--------------------------

```python
def test_pclass_meta_new_basic():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    # Create a basic PClass
    class TestPClass(CheckedType, metaclass=PClassMeta):
        pass
    
    assert hasattr(TestPClass, '_pclass_fields')
    assert isinstance(TestPClass._pclass_fields, dict)
    assert hasattr(TestPClass, '_pclass_invariants')
    assert hasattr(TestPClass, '__slots__')
    assert '_pclass_frozen' in TestPClass.__slots__
    assert '__weakref__' in TestPClass.__slots__


def test_pclass_meta_new_with_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    # Create a mock field
    class MockField(_PField):
        def __init__(self):
            pass
    
    field1 = MockField()
    field2 = MockField()
    
    class TestPClass(CheckedType, metaclass=PClassMeta):
        x = field1
        y = field2
    
    assert 'x' in TestPClass._pclass_fields
    assert 'y' in TestPClass._pclass_fields
    assert TestPClass._pclass_fields['x'] is field1
    assert TestPClass._pclass_fields['y'] is field2
    assert not hasattr(TestPClass, 'x')
    assert not hasattr(TestPClass, 'y')


def test_pclass_meta_new_slots_includes_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    class MockField(_PField):
        def __init__(self):
            pass
    
    class TestPClass(CheckedType, metaclass=PClassMeta):
        field_a = MockField()
        field_b = MockField()
    
    assert 'field_a' in TestPClass.__slots__
    assert 'field_b' in TestPClass.__slots__
    assert '_pclass_frozen' in TestPClass.__slots__


def test_pclass_meta_new_inherited_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    class MockField(_PField):
        def __init__(self):
            pass
    
    class BaseClass(CheckedType, metaclass=PClassMeta):
        base_field = MockField()
    
    class DerivedClass(BaseClass, metaclass=PClassMeta):
        derived_field = MockField()
    
    assert 'base_field' in DerivedClass._pclass_fields
    assert 'derived_field' in DerivedClass._pclass_fields


def test_pclass_meta_new_invariants():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    def test_invariant(self):
        return True, None
    
    class TestPClass(CheckedType, metaclass=PClassMeta):
        __invariant__ = test_invariant
    
    assert hasattr(TestPClass, '_pclass_invariants')
    assert isinstance(TestPClass._pclass_invariants, tuple)
    assert len(TestPClass._pclass_invariants) > 0


def test_pclass_meta_new_no_weakref_for_non_direct_subclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    class MockField(_PField):
        def __init__(self):
            pass
    
    class BaseClass(CheckedType, metaclass=PClassMeta):
        pass
    
    class DerivedClass(BaseClass, metaclass=PClassMeta):
        field1 = MockField()
    
    # __weakref__ should only be in BaseClass (direct subclass of CheckedType)
    assert '__weakref__' in BaseClass.__slots__
    assert '__weakref__' not in DerivedClass.__slots__


def test_pclass_meta_new_empty_class():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    class EmptyPClass(CheckedType, metaclass=PClassMeta):
        pass
    
    assert EmptyPClass._pclass_fields == {}
    assert EmptyPClass._pclass_invariants == ()
    assert '_pclass_frozen' in EmptyPClass.__slots__


# LLM-generated content at query #32
#--------------------------

```python
def test_pclass_meta_new_adds_weakref_slot_when_is_pclass_true():
    from pyrsistent._pclass import PClassMeta
    from unittest.mock import patch, MagicMock
    
    # Mock _is_pclass to return True
    with patch('pyrsistent._pclass._is_pclass', return_value=True):
        # Mock set_fields and store_invariants to avoid side effects
        with patch('pyrsistent._pclass.set_fields'):
            with patch('pyrsistent._pclass.store_invariants'):
                # Create test inputs
                name = 'TestClass'
                bases = (object,)
                dct = {'_pclass_fields': {}}
                
                # Call __new__
                result = PClassMeta.__new__(PClassMeta, name, bases, dct)
                
                # Assert that __weakref__ is in __slots__
                assert '__weakref__' in result.__slots__
                assert '_pclass_frozen' in result.__slots__


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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_constructor_multiple_fields_with_mixed_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field(initial=5)
        c = field(mandatory=True)
    
    instance = TestClass(a=1, c=3)
    assert instance.a == 1
    assert instance.b == 5
    assert instance.c == 3


# LLM-generated content at query #34
#--------------------------

```python
def test_pclass_hash_returns_consistent_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert isinstance(hash1, int)
    assert hash1 == hash2
    assert hash(instance1) == hash(instance1)


def test_pclass_hash_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    
    assert hash(instance1) != hash(instance2)


def test_pclass_hash_with_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1, y=None)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert isinstance(hash1, int)
    assert isinstance(hash2, int)


def test_pclass_hash_usable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=2, y=3)
    
    hash_set = {instance1, instance2, instance3}
    
    assert len(hash_set) >= 2
    assert hash(instance1) in {hash(instance1), hash(instance2), hash(instance3)}


def test_pclass_hash_usable_in_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    
    test_dict = {instance1: 'value1', instance2: 'value2'}
    
    assert test_dict[instance1] == 'value1'
    assert test_dict[instance2] == 'value2'
    assert len(test_dict) == 2


# LLM-generated content at query #35
#--------------------------

```python
def test_pclass_invariant_errors_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) > 0
        assert e.missing_fields == ('TestClass.x',)


def test_pclass_missing_fields_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) > 0
        assert 'TestClass.y' in e.missing_fields


def test_pclass_invariant_errors_or_missing_fields_true():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    exception_raised = False
    try:
        TestClass()
    except InvariantException:
        exception_raised = True
    
    assert exception_raised is True


# LLM-generated content at query #36
#--------------------------

```python
def test_remove_item_exists_in_data():
    class MockPClass:
        pass
    
    original = MockPClass()
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.remove('key1')
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver


# LLM-generated content at query #37
#--------------------------

```python
def test_pclass_meta_weakref_not_added_when_not_pclass_bases():
    from pyrsistent._pclass import PClassMeta
    
    # Create a metaclass instance with bases that are not PClass instances
    # This should result in _is_pclass(bases) evaluating to False
    name = 'TestClass'
    bases = (object,)
    dct = {'_pclass_fields': {}, '_pclass_invariants': ()}
    
    result = PClassMeta(name, bases, dct)
    
    # Verify that __weakref__ was NOT added to __slots__
    assert '__weakref__' not in result.__slots__
    assert result.__slots__ == ('_pclass_frozen',)


# LLM-generated content at query #38
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj = TestClass(x=1)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 10}


def test_pclass_reduce_empty_object():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
        y = field(initial=10)
    
    obj = TestClass()
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 5, 'y': 10}


def test_pclass_reduce_single_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x='test')
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][1] == {'x': 'test'}


def test_pclass_reduce_with_complex_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=[1, 2, 3], y={'key': 'value'}, z=(1, 2))
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[1][1] == {'x': [1, 2, 3], 'y': {'key': 'value'}, 'z': (1, 2)}


# LLM-generated content at query #39
#--------------------------

```python
def test_pclass_new_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3
    assert instance._pclass_frozen is True


# LLM-generated content at query #40
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj = TestClass(x=5)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 5, 'y': 10}


def test_pclass_reduce_with_no_fields():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {}


def test_pclass_reduce_with_multiple_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a='string', b=123, c=[1, 2, 3])
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[1][1] == {'a': 'string', 'b': 123, 'c': [1, 2, 3]}


# LLM-generated content at query #41
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


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='5')
    assert instance.x == 5


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass._pclass_fields
    # Using create method with ignore_extra
    result = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert result.x == 1


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    assert instance is not None


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #42
#--------------------------

```python
def test_pclass_eq_same_class_same_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert obj1 == obj2


def test_pclass_eq_same_class_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert not (obj1 == obj2)


def test_pclass_eq_different_class():
    from pyrsistent import PClass, field
    
    class TestClass1(PClass):
        x = field()
    
    class TestClass2(PClass):
        x = field()
    
    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    
    assert not (obj1 == obj2)


def test_pclass_eq_with_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1, y=5)
    
    assert obj1 == obj2


def test_pclass_eq_with_non_pclass_object():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = {"x": 1}
    
    assert (obj1 == obj2) is NotImplemented


def test_pclass_eq_reflexive():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    
    assert obj1 == obj1


def test_pclass_eq_with_none_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=None)
    obj2 = TestClass(x=1, y=None)
    
    assert obj1 == obj2


def test_pclass_eq_complex_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=[1, 2, 3], y={"a": 1})
    obj2 = TestClass(x=[1, 2, 3], y={"a": 1})
    
    assert obj1 == obj2


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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 5)
    
    instance = TestClass()
    assert instance.x == 5


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_with_default_and_provided_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field(initial=20)
    
    instance = TestClass(x=100)
    assert instance.x == 100
    assert instance.y == 20


# LLM-generated content at query #44
#--------------------------

```python
def test_pclass_new_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3
    assert hasattr(instance, '_pclass_frozen')
    assert instance._pclass_frozen is True


# LLM-generated content at query #45
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    # The predicate at line 7 is: "for name in self._pclass_fields"
    # This evaluates to True when _pclass_fields is iterable and non-empty
    assert hasattr(instance, '_pclass_fields')
    assert len(instance._pclass_fields) > 0
    assert 'x' in instance._pclass_fields
    assert 'y' in instance._pclass_fields
    
    # Verify iteration works
    field_names = list(instance._pclass_fields)
    assert len(field_names) == 2
    assert set(field_names) == {'x', 'y'}


# LLM-generated content at query #46
#--------------------------

```python
def test_hash_basic():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)
    
    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


def test_hash_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1, y=None)
    
    assert hash(obj1) == hash(obj2)


def test_hash_consistent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    hash1 = hash(obj)
    hash2 = hash(obj)
    
    assert hash1 == hash2


def test_hash_different_field_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=2, y=1)
    
    assert hash(obj1) != hash(obj2)


def test_hash_with_string_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
        value = field()
    
    obj1 = TestClass(name="test", value="data")
    obj2 = TestClass(name="test", value="data")
    
    assert hash(obj1) == hash(obj2)


def test_hash_hashable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    obj3 = TestClass(x=2)
    
    hash_set = {obj1, obj2, obj3}
    assert len(hash_set) == 2


def test_hash_hashable_in_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    test_dict = {obj1: "value1"}
    test_dict[obj2] = "value2"
    
    assert len(test_dict) == 1
    assert test_dict[obj1] == "value2"


# LLM-generated content at query #47
#--------------------------

```python
def test_pclass_eq_same_class_equal_fields():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2


def test_pclass_eq_same_class_different_fields():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    assert not (obj1 == obj2)


def test_pclass_eq_same_class_one_field_missing():
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1)
    assert not (obj1 == obj2)


def test_pclass_eq_different_classes():
    class TestClass1(PClass):
        x = field()
    
    class TestClass2(PClass):
        x = field()
    
    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    assert obj1 != obj2


def test_pclass_eq_with_none_values():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=None, y=2)
    obj2 = TestClass(x=None, y=2)
    assert obj1 == obj2


def test_pclass_eq_with_complex_values():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=[1, 2, 3], y={'a': 1})
    obj2 = TestClass(x=[1, 2, 3], y={'a': 1})
    assert obj1 == obj2


def test_pclass_eq_returns_not_implemented_for_different_type():
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    result = obj1.__eq__("not a pclass")
    assert result is NotImplemented


def test_pclass_eq_single_field():
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    assert obj1 == obj2


def test_pclass_eq_multiple_fields_all_match():
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj1 = TestClass(a=1, b=2, c=3, d=4)
    obj2 = TestClass(a=1, b=2, c=3, d=4)
    assert obj1 == obj2


def test_pclass_eq_multiple_fields_one_differs():
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj1 = TestClass(a=1, b=2, c=3, d=4)
    obj2 = TestClass(a=1, b=2, c=3, d=5)
    assert not (obj1 == obj2)


# LLM-generated content at query #48
#--------------------------

```python
def test_pclass_meta_new_predicate_line_1():
    from pyrsistent._pclass import PClassMeta
    
    # Test that __new__ is a method that can be called
    assert callable(PClassMeta.__new__)
    
    # Test that __new__ is defined on PClassMeta
    assert hasattr(PClassMeta, '__new__')
    
    # Verify __new__ is actually a method (bound or unbound)
    assert '__new__' in PClassMeta.__dict__ or '__new__' in type(PClassMeta).__dict__


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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field, AttributeError as PAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="5")
    assert instance.x == 5


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b="test", c=[1, 2, 3])
    assert instance.a == 1
    assert instance.b == "test"
    assert instance.c == [1, 2, 3]


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


# LLM-generated content at query #50
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields
        assert len(e.missing_fields) == 1


# LLM-generated content at query #51
#--------------------------

```python
def test_pclass_new_raises_invariant_exception_when_invariant_errors_exist():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return False, "test_error"
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (failing_invariant,)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert "test_error" in e.error_codes


def test_pclass_new_raises_invariant_exception_when_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields


def test_pclass_new_raises_invariant_exception_when_both_invariant_errors_and_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return False, "invariant_error"
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
        __invariants__ = (failing_invariant,)
    
    try:
        TestClass(y=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.error_codes) > 0
        assert len(e.missing_fields) > 0


# LLM-generated content at query #52
#--------------------------

```python
def test_set_method_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    updated = instance.set(x=10)
    
    assert updated.x == 10
    assert updated.y == 2
    assert updated.z == 3
    assert instance.x == 1


# LLM-generated content at query #53
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    # Call __eq__ which will evaluate the isinstance predicate at line 2
    result = obj1.__eq__(obj2)
    
    # The predicate at line 3 will be entered (the for loop), which means
    # the isinstance check at line 2 evaluated to True
    assert result == True


# LLM-generated content at query #54
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    # The predicate at line 2 (isinstance(other, self.__class__)) evaluates to True
    # when comparing two instances of the same PClass
    result = obj1 == obj2
    assert result is True


# LLM-generated content at query #55
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    # The predicate at line 7 is: `for name in self._pclass_fields:`
    # This evaluates to True when self._pclass_fields is iterable and non-empty
    assert hasattr(instance, '_pclass_fields')
    assert len(instance._pclass_fields) > 0
    assert all(name in instance._pclass_fields for name in ['x', 'y', 'z'])


# LLM-generated content at query #56
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self, invariant_result):
            self.type = None
            self.invariant_result = invariant_result
        
        def invariant(self, value):
            return self.invariant_result
    
    class MockClass:
        pass
    
    result = MockClass()
    invariant_errors = []
    field = MockField((False, "error_code_1"))
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert invariant_errors == ["error_code_1"]
    assert not hasattr(result, "test_field")


# LLM-generated content at query #57
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


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, extra_field=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'extra_field' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, extra=2)
    assert instance.x == 1
    assert not hasattr(instance, 'extra')


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_with_none_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=None, y=5)
    assert instance.x is None
    assert instance.y == 5


# LLM-generated content at query #58
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=42)
        y = field()
    
    instance = TestClass(y=10)
    assert instance.x == 42
    assert instance.y == 10


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, extra_field=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'extra_field' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, extra=2)
    assert instance.x == 1
    assert not hasattr(instance, 'extra')


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


# LLM-generated content at query #59
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._precord import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='5')
    assert instance.x == 5
    assert isinstance(instance.x, int)


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field
    from pyrsistent._precord import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
        z = field()
    
    try:
        instance = TestClass(z=3)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #60
#--------------------------

```python
def test_remove_item_exists():
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    
    class MockOriginal:
        pass
    
    original = MockOriginal()
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.remove('key1')
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver


# LLM-generated content at query #61
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class MockField:
        def __init__(self, field_type=None, invariant_result=(True, None)):
            self.type = field_type
            self._invariant_result = invariant_result
        
        def invariant(self, value):
            return self._invariant_result
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField(field_type=(int,), invariant_result=(True, None))
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._exceptions import PTypeError
    
    class MockField:
        def __init__(self, field_type=None):
            self.type = field_type
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField(field_type=(int,))
    result = MockResult()
    invariant_errors = []
    
    try:
        _check_and_set_attr(MockClass, field, "test_field", "not_an_int", result, invariant_errors)
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type=None):
            self.type = field_type
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField(field_type=(int,))
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == ["invariant_error_code"]


def test_check_and_set_attr_no_type_check():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField()
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "any_value", result, invariant_errors)
    
    assert result.test_field == "any_value"
    assert invariant_errors == []


def test_check_and_set_attr_multiple_valid_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = (int, str, float)
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField()
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 3.14, result, invariant_errors)
    
    assert result.test_field == 3.14
    assert invariant_errors == []


# LLM-generated content at query #62
#--------------------------

```python
def test_remove_existing_item():
    original = object()
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.remove('key1')
    
    assert result is evolver
    assert 'key1' not in evolver._pclass_evolver_data
    assert 'key1' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data == {'key2': 'value2'}


def test_remove_nonexistent_item():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    try:
        evolver.remove('nonexistent')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == 'nonexistent'


def test_remove_item_after_set():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.set('key2', 'value2')
    result = evolver.remove('key2')
    
    assert result is evolver
    assert 'key2' not in evolver._pclass_evolver_data
    assert 'key2' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True


def test_remove_multiple_items():
    original = object()
    initial_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.remove('key1')
    evolver.remove('key3')
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert 'key3' not in evolver._pclass_evolver_data
    assert 'key2' in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True


def test_remove_via_delitem():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    del evolver['key1']
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #63
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field(type=str)
    
    test_field = TestClass.__pclass_fields__['name']
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, test_field, 'name', 'valid_name', result, invariant_errors)
    
    assert result.name == 'valid_name'
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        name = field(type=str)
    
    test_field = TestClass.__pclass_fields__['name']
    result = TestClass()
    invariant_errors = []
    
    try:
        _check_and_set_attr(TestClass, test_field, 'name', 123, result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_failed_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    
    def positive_invariant(value):
        if value > 0:
            return True, None
        return False, "must_be_positive"
    
    class TestClass(PClass):
        count = field(type=int, invariant=positive_invariant)
    
    test_field = TestClass.__pclass_fields__['count']
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, test_field, 'count', -5, result, invariant_errors)
    
    assert invariant_errors == ["must_be_positive"]
    assert not hasattr(result, 'count') or result.count is None


def test_check_and_set_attr_multiple_types():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        value = field(type=(str, int))
    
    test_field = TestClass.__pclass_fields__['value']
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, test_field, 'value', 42, result, invariant_errors)
    
    assert result.value == 42
    assert invariant_errors == []


def test_check_and_set_attr_no_type_checking():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        value = field()
    
    test_field = TestClass.__pclass_fields__['value']
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, test_field, 'value', [1, 2, 3], result, invariant_errors)
    
    assert result.value == [1, 2, 3]
    assert invariant_errors == []


# LLM-generated content at query #64
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._exceptions import PTypeError
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        __name__ = "MockClass"
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    try:
        _check_and_set_attr(MockClass, field, "test_field", "invalid", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (False, "value_too_small"))
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert invariant_errors == ["value_too_small"]
    assert not hasattr(result, "test_field")


def test_check_and_set_attr_no_type_constraint():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField(None, lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", "any_value", result, invariant_errors)
    
    assert result.test_field == "any_value"
    assert invariant_errors == []


def test_check_and_set_attr_multiple_valid_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int, str], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", "hello", result, invariant_errors)
    
    assert result.test_field == "hello"
    assert invariant_errors == []


# LLM-generated content at query #65
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


# LLM-generated content at query #66
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PyrsistentAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_instances_independent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    assert instance1.x == 1
    assert instance2.x == 2


# LLM-generated content at query #67
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    try:
        class TestClass(PClass):
            x = field(mandatory=True)
        
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "missing_fields" in str(type(e)).lower() or "invariant" in str(type(e)).lower()


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_constructor_ignore_extra_true():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "missing_fields" in str(type(e)).lower() or "invariant" in str(type(e)).lower()


def test_pclass_constructor_with_all_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    instance = TestClass()
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #68
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self, field_type, invariant_result):
            self.type = field_type
            self._invariant_result = invariant_result
        
        def invariant(self, value):
            return self._invariant_result
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField([int], (True, None))
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_result):
            self.type = field_type
            self._invariant_result = invariant_result
        
        def invariant(self, value):
            return self._invariant_result
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField([int], (False, "error_code_1"))
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == ["error_code_1"]


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField([int])
    result = MockResult()
    invariant_errors = []
    
    try:
        _check_and_set_attr(MockClass, field, "test_field", "not_an_int", result, invariant_errors)
        assert False, "Expected PTypeError to be raised"
    except PTypeError:
        pass


def test_check_and_set_attr_multiple_valid_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField([int, str])
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "string_value", result, invariant_errors)
    
    assert result.test_field == "string_value"
    assert invariant_errors == []


def test_check_and_set_attr_no_type_check():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        __name__ = "TestClass"
    
    class MockResult:
        pass
    
    field = MockField()
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", object(), result, invariant_errors)
    
    assert hasattr(result, "test_field")
    assert invariant_errors == []


# LLM-generated content at query #69
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
        x = field(initial=lambda: 5)
    
    instance = TestClass()
    assert instance.x == 5


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e.missing_fields)


def test_pclass_constructor_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(ignore_extra=True, x=1, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


# LLM-generated content at query #70
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
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_no_arguments():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
    
    instance = TestClass()
    assert instance.x == 5


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field(initial=4)
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #71
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        TestClass(y=20)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=3, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x={'a': 1}, _factory_fields={'x'})
    assert instance.x == pmap({'a': 1})


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
        z = field()
    
    try:
        TestClass(z=30)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #72
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=20)
    assert obj.x == 10
    assert obj.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    obj = TestClass()
    assert obj.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, _factory_fields={'x'})
    assert obj.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, ignore_extra=True, z=2)
    assert obj.x == 1
    assert not hasattr(obj, 'z')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert obj is not None


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj = TestClass(a=1, b=2, c=3, d=4)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3
    assert obj.d == 4


def test_pclass_constructor_with_none_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=None, y=2)
    assert obj.x is None
    assert obj.y == 2


# LLM-generated content at query #73
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
        y = field()
    
    instance = TestClass(x="5", y=10)
    assert instance.x == 5
    assert instance.y == 10


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


# LLM-generated content at query #74
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return False, "invariant_error_code"
    
    class MockClass:
        pass
    
    field = MockField()
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert invariant_errors == ["invariant_error_code"]
    assert not hasattr(result, "test_field")


# LLM-generated content at query #75
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


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PyrsistentAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        data = field(factory=pmap)
    
    instance = TestClass(data={'a': 1})
    assert instance.data['a'] == 1


def test_pclass_constructor_multiple_instances_independent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    assert instance1.x == 1
    assert instance2.x == 2


def test_pclass_constructor_with_no_arguments():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
        y = field(initial=10)
    
    instance = TestClass()
    assert instance.x == 5
    assert instance.y == 10


# LLM-generated content at query #76
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
        x = field(initial=lambda: [])
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == []
    assert instance.y == 2


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        instance = TestClass(y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=3, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert instance._pclass_frozen is True


def test_pclass_constructor_frozen_prevents_modification():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


def test_pclass_constructor_field_with_factory():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    instance = TestClass(x={'key': 'value'})
    assert instance.x == pmap({'key': 'value'})


# LLM-generated content at query #77
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


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


def test_pclass_constructor_empty_pclass():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    assert isinstance(instance, EmptyClass)


# LLM-generated content at query #78
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
    
    obj = TestClass()
    assert obj.x == 10


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    obj = TestClass()
    assert obj.x == 42


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "x must be positive"))
    
    try:
        obj = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) > 0


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    obj = TestClass(x={'a': 1})
    assert obj.x == pmap({'a': 1})


def test_pclass_new_with_ignore_extra_false():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, extra=2, ignore_extra=False)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_pclass_new_with_type_checking():
    from pyrsistent import PClass, field, PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        obj = TestClass(x="not an int")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_multiple_fields_and_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
        z = field()
    
    obj = TestClass(z=3)
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3


def test_pclass_new_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert isinstance(obj, TestClass)


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, InvariantException
    
    def global_inv(obj):
        return (obj.x > 0, "x must be positive")
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (global_inv,)
    
    try:
        obj = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Global invariant failed" in str(e)


def test_pclass_new_with_factory_fields_parameter():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
        y = field()
    
    obj = TestClass(_factory_fields={'x'}, x={'a': 1}, y=2)
    assert obj.x == pmap({'a': 1})
    assert obj.y == 2


def test_pclass_new_override_initial_with_kwarg():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
    
    obj = TestClass(x=20)
    assert obj.x == 20


# LLM-generated content at query #79
#--------------------------

```python
def test_pclass_meta_new_creates_slots_with_pclass_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    # Create a simple field for testing
    test_field = _PField(type=int, initial=0, factory=None, invariant=None, initial_factory=None)
    
    # Create a test dictionary with fields
    dct = {
        'test_field': test_field,
        '__module__': '__main__'
    }
    
    # Create a new class using PClassMeta
    TestClass = PClassMeta('TestClass', (), dct)
    
    # Verify that __slots__ was created
    assert hasattr(TestClass, '__slots__')
    
    # Verify that __slots__ contains '_pclass_frozen'
    assert '_pclass_frozen' in TestClass.__slots__
    
    # Verify that __slots__ contains the field name
    assert 'test_field' in TestClass.__slots__
    
    # Verify the structure of __slots__
    assert TestClass.__slots__[0] == '_pclass_frozen'


# LLM-generated content at query #80
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


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_new_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_with_type_check():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="not an int")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive_invariant(value):
        return (value > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert "must be positive" in e.error_codes


def test_pclass_new_preserves_field_order():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_new_with_factory_fields_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
        y = field()
    
    instance = TestClass(_factory_fields={'x'}, x="42", y=99)
    assert instance.x == 42
    assert instance.y == 99


def test_pclass_new_with_ignore_extra_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(ignore_extra=True, x=1)
    assert instance.x == 1


def test_pclass_new_multiple_instances_independent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    assert instance1.x == 1
    assert instance2.x == 2


def test_pclass_new_with_optional_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert hasattr(instance, 'x')
    assert hasattr(instance, 'y')


def test_pclass_new_field_not_set_remains_missing():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def global_check(obj):
        return (obj.x > 0, "x must be positive")
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (global_check,)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert "x must be positive" in e.error_codes


def test_pclass_new_with_all_fields_optional():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass()
    assert instance._pclass_frozen


def test_pclass_new_passes_ignore_extra_to_factory():
    from pyrsistent import PClass, field
    from pyrsistent._pmap import pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    instance = TestClass(x={'a': 1}, ignore_extra=True)
    assert instance.x == pmap({'a': 1})


# LLM-generated content at query #81
#--------------------------

```python
def test_repr_format():
    class TestPClass(PClass):
        x = field()
        y = field()
    
    obj = TestPClass(x=1, y="hello")
    repr_str = repr(obj)
    
    assert "TestPClass(" in repr_str
    assert "x=1" in repr_str
    assert "y='hello'" in repr_str
    assert repr_str.endswith(")")
    assert repr_str == "TestPClass(x=1, y='hello')" or repr_str == "TestPClass(y='hello', x=1)"


# LLM-generated content at query #82
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    # The predicate at line 7 is: `for name in self._pclass_fields:`
    # This evaluates to True when _pclass_fields is iterable and contains field names
    assert hasattr(instance, '_pclass_fields')
    assert 'x' in instance._pclass_fields
    assert 'y' in instance._pclass_fields
    assert 'z' in instance._pclass_fields
    
    # Verify iteration works
    field_names = list(instance._pclass_fields)
    assert len(field_names) == 3
    assert set(field_names) == {'x', 'y', 'z'}


# LLM-generated content at query #83
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=5, _factory_fields={'x'})
    assert obj.x == 5


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, ignore_extra=True, y=2)
    assert obj.x == 1
    assert not hasattr(obj, 'y')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field(initial=100)
    
    obj = TestClass(a=10, b=20)
    assert obj.a == 10
    assert obj.b == 20
    assert obj.c == 100


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert isinstance(obj, TestClass)


# LLM-generated content at query #84
#--------------------------

```python
def test_set_method_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    updated_instance = instance.set(x=10)
    
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert 'z' in TestClass._pclass_fields
    assert updated_instance.x == 10
    assert updated_instance.y == 2
    assert updated_instance.z == 3


# LLM-generated content at query #85
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.y' in e.missing_fields
        assert len(e.missing_fields) == 1


def test_pclass_raises_invariant_exception_with_invariant_errors():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    def check_x(obj):
        if obj.x < 0:
            return False, "x must be positive"
        return True, None
    
    TestClass._pclass_invariants = (check_x,)
    
    try:
        TestClass(x=-5)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.error_codes) > 0


# LLM-generated content at query #86
#--------------------------

```python
def test_pclass_hash_returns_consistent_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    
    hash_value1 = hash(instance1)
    hash_value2 = hash(instance2)
    
    assert isinstance(hash_value1, int)
    assert hash_value1 == hash_value2
    assert hash(instance1) == hash(instance1)


def test_pclass_hash_different_for_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    
    assert hash(instance1) != hash(instance2)


def test_pclass_hash_with_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    instance = TestClass(x=5)
    hash_value = hash(instance)
    
    assert isinstance(hash_value, int)
    assert hash_value == hash(instance)


def test_pclass_hash_with_nested_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=(1, 2), y={'a': 1})
    hash_value = hash(instance)
    
    assert isinstance(hash_value, int)


# LLM-generated content at query #87
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj = TestClass(x=5)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 5, 'y': 10}


def test_pclass_reduce_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=None)
    
    obj = TestClass()
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': None}


def test_pclass_reduce_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj = TestClass(a=1, b='test', c=[1, 2, 3], d={'key': 'value'})
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[1][1] == {'a': 1, 'b': 'test', 'c': [1, 2, 3], 'd': {'key': 'value'}}


# LLM-generated content at query #88
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    # The predicate at line 3 evaluates to True when isinstance(other, self.__class__) is True
    # This happens when comparing two instances of the same PClass
    result = obj1 == obj2
    
    assert result is True


# LLM-generated content at query #89
#--------------------------

```python
def test_pclass_meta_weakref_not_added_when_not_pclass_bases():
    from pyrsistent._pclass import PClassMeta
    
    # Create a metaclass instance with bases that are NOT PClass instances
    # This should result in _is_pclass(bases) evaluating to False
    name = 'TestClass'
    bases = (object,)
    dct = {'_pclass_fields': {}, '_pclass_invariants': ()}
    
    result = PClassMeta(name, bases, dct)
    
    # When _is_pclass(bases) is False, __weakref__ should NOT be added to __slots__
    assert '__weakref__' not in result.__slots__
    assert result.__slots__ == ('_pclass_frozen',)


# LLM-generated content at query #90
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
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


