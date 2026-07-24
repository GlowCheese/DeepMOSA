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
    assert evolver._pclass_evolver_data == {'new_key': 'new_value'}
    assert evolver._factory_fields == {'new_key'}

def test_set_existing_key_with_different_value_marks_dirty():
    evolver = _PClassEvolver(object(), {'key': 'old_value'})
    evolver.set('key', 'new_value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data == {'key': 'new_value'}
    assert evolver._factory_fields == {'key'}

def test_set_existing_key_with_same_value_does_not_mark_dirty():
    evolver = _PClassEvolver(object(), {'key': 'value'})
    evolver.set('key', 'value')
    assert evolver._pclass_evolver_data_is_dirty is False
    assert evolver._pclass_evolver_data == {'key': 'value'}
    assert evolver._factory_fields == set()

def test_set_returns_self():
    evolver = _PClassEvolver(object(), {})
    result = evolver.set('key', 'value')
    assert result is evolver


# LLM-generated content at query #2
#--------------------------

```python
def test_persistent_returns_original_when_not_dirty():
    original = object()
    evolver = _PClassEvolver(original, {})
    assert evolver.persistent() is original

def test_persistent_returns_new_instance_when_dirty():
    original = object()
    evolver = _PClassEvolver(original, {})
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert result is not original
    assert result._factory_fields == {'key'}
    assert result.key == 'value'

def test_persistent_includes_all_factory_fields():
    original = object()
    evolver = _PClassEvolver(original, {})
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    result = evolver.persistent()
    assert result._factory_fields == {'field1', 'field2'}
    assert result.field1 == 'value1'
    assert result.field2 == 'value2'

def test_persistent_excludes_removed_fields():
    original = object()
    evolver = _PClassEvolver(original, {'field1': 'value1', 'field2': 'value2'})
    evolver.remove('field1')
    result = evolver.persistent()
    assert 'field1' not in result._factory_fields
    assert result.field2 == 'value2'


# LLM-generated content at query #3
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

def test_set_with_mixed_arguments():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set('x', 10, y=20)
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_pclass_reduce_returns_correct_tuple():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    reduced = instance.__reduce__()
    assert reduced == (_restore_pickle, (TestClass, {'x': 1, 'y': 2}))


# LLM-generated content at query #5
#--------------------------

```python
def test_pclass_repr():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"


# LLM-generated content at query #6
#--------------------------

```python
def test_repr_returns_correct_format():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"


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

def test_pclass_new_with_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "must be positive"))

    instance = TestClass(x=1)
    assert instance.x == 1

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "must be positive" in str(e)

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._pclass import InvariantException

    def check_sum(instance):
        return instance.x + instance.y > 0, "sum must be positive"

    @invariant(check_sum)
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "sum must be positive" in str(e)

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

    instance = TestClass(x=5, y=2, _factory_fields={"x"})
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_type_check():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError

    class TestClass(PClass):
        x = field(type=int)

    instance = TestClass(x=1)
    assert instance.x == 1

    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert "Invalid type" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_pclassmeta_new_with_single_checkedtype_base():
    bases = (CheckedType,)
    dct = {'x': _PField(), 'y': _PField()}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert result._pclass_fields == {'x': dct['x'], 'y': dct['y']}
    assert result._pclass_invariants == ()
    assert result.__slots__ == ('_pclass_frozen', 'x', 'y', '__weakref__')

def test_pclassmeta_new_with_multiple_bases():
    class Base1(CheckedType):
        pass
    class Base2(CheckedType):
        pass
    bases = (Base1, Base2)
    dct = {'z': _PField()}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert 'z' not in result._pclass_fields
    assert result._pclass_invariants == ()
    assert result.__slots__ == ('_pclass_frozen', '__weakref__')

def test_pclassmeta_new_with_invariant():
    def test_invariant(obj):
        return True, ()
    bases = (CheckedType,)
    dct = {'__invariant__': test_invariant, 'x': _PField()}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert result._pclass_fields == {'x': dct['x']}
    assert len(result._pclass_invariants) == 1
    assert result._pclass_invariants[0](None) == (True, ())
    assert result.__slots__ == ('_pclass_frozen', 'x', '__weakref__')

def test_pclassmeta_new_with_inherited_fields():
    class Base(CheckedType):
        x = _PField()
    bases = (Base,)
    dct = {'y': _PField()}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert result._pclass_fields == {'x': Base.x, 'y': dct['y']}
    assert result._pclass_invariants == ()
    assert result.__slots__ == ('_pclass_frozen', 'x', 'y', '__weakref__')

def test_pclassmeta_new_with_inherited_invariant():
    def base_invariant(obj):
        return True, ()
    class Base(CheckedType):
        __invariant__ = base_invariant
    bases = (Base,)
    dct = {}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert result._pclass_fields == {}
    assert len(result._pclass_invariants) == 1
    assert result._pclass_invariants[0](None) == (True, ())
    assert result.__slots__ == ('_pclass_frozen', '__weakref__')

def test_pclassmeta_new_with_non_callable_invariant():
    bases = (CheckedType,)
    dct = {'__invariant__': 'not_callable'}
    try:
        PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test_pclassmeta_new_with_no_checkedtype_base():
    bases = (object,)
    dct = {'x': _PField()}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert result._pclass_fields == {'x': dct['x']}
    assert result._pclass_invariants == ()
    assert result.__slots__ == ('_pclass_frozen', 'x')


# LLM-generated content at query #9
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
        assert ("TestClass.x",) == e.missing_fields

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z' are not among the specified fields for TestClass" == str(e)

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

def test_pclass_new_with_invariant_error():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

    def invariant(value):
        return (value > 0, "must_be_positive")

    class TestClass(PClass):
        x = field(invariant=invariant)
        y = field()

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert ("must_be_positive",) == e.error_codes

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field

    def factory(value):
        return value * 2

    class TestClass(PClass):
        x = field(factory=factory)
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 2
    assert instance.y == 2

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

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
        assert ("sum_must_be_positive",) == e.error_codes

def test_pclass_new_with_frozen_attribute():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "Can't set attribute, key=x, value=2" == str(e)


# LLM-generated content at query #10
#--------------------------

```python
def test_serialize_without_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="test")
    result = instance.serialize()
    assert result == {"x": 1, "y": "test"}

def test_serialize_with_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v: str(v))
        y = field()

    instance = TestClass(x=1, y="test")
    result = instance.serialize()
    assert result == {"x": "1", "y": "test"}

def test_serialize_with_format():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v, fmt: str(v) if fmt == "str" else v)
        y = field()

    instance = TestClass(x=1, y="test")
    result = instance.serialize(format="str")
    assert result == {"x": "1", "y": "test"}

def test_serialize_missing_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1)
    result = instance.serialize()
    assert result == {"x": 1}


# LLM-generated content at query #11
#--------------------------

```python
def test_pclass_hash_returns_consistent_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert hash(instance) == hash(instance)

def test_pclass_hash_different_instances_same_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert hash(instance1) == hash(instance2)

def test_pclass_hash_different_instances_different_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    assert hash(instance1) != hash(instance2)

def test_pclass_hash_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    assert hash(instance1) == hash(instance2)

def test_pclass_hash_with_none_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=None, y=None)
    instance2 = TestClass(x=None, y=None)
    assert hash(instance1) == hash(instance2)


# LLM-generated content at query #12
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


# LLM-generated content at query #13
#--------------------------

```python
def test__check_and_set_attr_with_valid_type_and_invariant():
    class TestClass:
        pass

    class TestField:
        type = (int,)
        def invariant(self, value):
            return True, None

    result = TestClass()
    invariant_errors = []

    _check_and_set_attr(TestClass, TestField(), "test_field", 10, result, invariant_errors)

    assert hasattr(result, "test_field")
    assert result.test_field == 10
    assert invariant_errors == []

def test__check_and_set_attr_with_invalid_type():
    class TestClass:
        pass

    class TestField:
        type = (int,)
        def invariant(self, value):
            return True, None

    result = TestClass()
    invariant_errors = []

    try:
        _check_and_set_attr(TestClass, TestField(), "test_field", "not_an_int", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert str(e) == "Invalid type for field TestClass.test_field, was str"

def test__check_and_set_attr_with_failed_invariant():
    class TestClass:
        pass

    class TestField:
        type = (int,)
        def invariant(self, value):
            return False, "INVALID"

    result = TestClass()
    invariant_errors = []

    _check_and_set_attr(TestClass, TestField(), "test_field", 10, result, invariant_errors)

    assert not hasattr(result, "test_field")
    assert invariant_errors == ["INVALID"]


# LLM-generated content at query #14
#--------------------------

```python
def test_pclass_pickling_returns_restore_pickle_tuple():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass(x=42)
    result = instance.__reduce__()

    assert result[0] is _restore_pickle
    assert result[1] == (TestClass, {'x': 42})


# LLM-generated content at query #15
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
    instance2 = TestClass(x=3, y=4)
    assert not (instance1 == instance2)

def test_eq_different_classes():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    assert not (instance1 == instance2)

def test_eq_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert not (instance == 1)
    assert not (instance == {"x": 1})


# LLM-generated content at query #16
#--------------------------

```python
def test_pclass_hash_consistency():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=1, y=3)

    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

```python
def test_weakref_in_slots_when_bases_are_pclass():
    class Base(metaclass=PClassMeta):
        pass

    class Child(Base):
        pass

    assert '__weakref__' in Child.__slots__


# LLM-generated content at query #19
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field, pvector_field, pmap_field, pset_field
    class TestClass(PClass):
        x = field()
        y = pvector_field(int)
        z = pmap_field(str, int)
        w = pset_field(str)

    instance = TestClass(x=1, y=[2, 3], z={'a': 1}, w={'b', 'c'})
    assert instance.x == 1
    assert instance.y == pvector([2, 3])
    assert instance.z == pmap({'a': 1})
    assert instance.w == pset({'b', 'c'})

def test_pclass_new_with_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('TestClass.x',)

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'y' are not among the specified fields for TestClass"

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

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
    from pyrsistent import PClass, field, pvector_field, PTypeError
    class TestClass(PClass):
        x = pvector_field(int)

    try:
        TestClass(x=[1, 'a'])
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.destination_cls == TestClass
        assert e.field_name == 'x'
        assert e.expected_type == (int,)
        assert e.actual_type == str

def test_pclass_new_with_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, 'must_be_positive'))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('must_be_positive',)

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException, invariant
    @invariant
    def check_sum(obj):
        return obj.x + obj.y > 0, 'sum_must_be_positive'

    class TestClass(PClass):
        x = field()
        y = field()

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('sum_must_be_positive',)

def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    def custom_factory(value):
        return value.upper()

    class TestClass(PClass):
        x = field(factory=custom_factory)

    instance = TestClass(x='hello')
    assert instance.x == 'HELLO'

def test_pclass_new_with_factory_field_and_ignore_extra():
    from pyrsistent import PClass, field
    def custom_factory(value, ignore_extra=False):
        return value.upper() if ignore_extra else value.lower()

    class TestClass(PClass):
        x = field(factory=custom_factory)

    instance = TestClass.create({'x': 'hello'}, ignore_extra=True)
    assert instance.x == 'HELLO'

def test_pclass_new_with_factory_fields_param():
    from pyrsistent import PClass, field
    def custom_factory(value):
        return value.upper()

    class TestClass(PClass):
        x = field(factory=custom_factory)
        y = field()

    instance = TestClass(x='hello', y='world', _factory_fields={'x'})
    assert instance.x == 'HELLO'
    assert instance.y == 'world'


# LLM-generated content at query #20
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


# LLM-generated content at query #21
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
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in str(e.missing_fields)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

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


# LLM-generated content at query #22
#--------------------------

```python
def test_pclass_fields_items_iteration():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    test_instance = TestClass(x=1)
    fields_dict = TestClass._pclass_fields
    assert list(fields_dict.items()) == [('x', fields_dict['x'])]


# LLM-generated content at query #23
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
    except InvariantException as e:
        assert e.args[0] == () and e.args[1] == ('TestClass.x',) and e.args[2] == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #24
#--------------------------

```python
def test_serialize_returns_dict_with_all_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    result = instance.serialize()
    assert isinstance(result, dict)
    assert 'x' in result
    assert 'y' in result


# LLM-generated content at query #25
#--------------------------

```python
def test_pclass_hash_consistency():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_set_new_key_marks_dirty():
    evolver = _PClassEvolver(object(), {})
    evolver.set('new_key', 'value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data['new_key'] == 'value'
    assert 'new_key' in evolver._factory_fields

def test_set_existing_key_with_same_value():
    evolver = _PClassEvolver(object(), {'key': 'value'})
    evolver.set('key', 'value')
    assert evolver._pclass_evolver_data_is_dirty is False
    assert 'key' not in evolver._factory_fields

def test_set_existing_key_with_different_value():
    evolver = _PClassEvolver(object(), {'key': 'old_value'})
    evolver.set('key', 'new_value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data['key'] == 'new_value'
    assert 'key' in evolver._factory_fields

def test_set_returns_self():
    evolver = _PClassEvolver(object(), {})
    result = evolver.set('key', 'value')
    assert result is evolver


# LLM-generated content at query #2
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field, m, s
    class TestClass(PClass):
        x = field(type=int, mandatory=True)
        y = field(type=str, initial="default")
    instance = TestClass(x=1)
    assert instance.x == 1
    assert instance.y == "default"

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, mandatory=True)
    try:
        TestClass()
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=str)
    instance = TestClass(_factory_fields={"x"}, x=1, y="value")
    assert instance.x == 1
    assert instance.y == "value"

def test_pclass_constructor_with_invariant_check():
    from pyrsistent import PClass, field, invariant
    def check_positive(instance):
        return instance.x > 0
    class TestClass(PClass):
        x = field(type=int, invariant=check_positive)
    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)


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
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

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

def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field, PTypeError

    class TestClass(PClass):
        x = field(type=int)
        y = field()

    try:
        TestClass(x="not an int", y=2)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert "Invalid type for field TestClass.x" in str(e)

def test_pclass_new_with_factory():
    from pyrsistent import PClass, field

    def custom_factory(value):
        return value * 2

    class TestClass(PClass):
        x = field(factory=custom_factory)
        y = field()

    instance = TestClass(x=5, y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_invariant_failure():
    from pyrsistent import PClass, field, InvariantException

    def invariant(value):
        return (value > 0, "Value must be positive")

    class TestClass(PClass):
        x = field(invariant=invariant)
        y = field()

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in e.error_codes

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException

    def global_invariant(instance):
        return (instance.x + instance.y > 0, "Sum must be positive")

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_invariant]

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

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

    def custom_factory(value):
        return value * 2

    class TestClass(PClass):
        x = field(factory=custom_factory)
        y = field()

    instance = TestClass(x=5, y=2, _factory_fields={"x"})
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_ignore_extra_and_factory():
    from pyrsistent import PClass, field

    def custom_factory(value, ignore_extra=False):
        if ignore_extra:
            return value * 3
        return value * 2

    class TestClass(PClass):
        x = field(factory=custom_factory)
        y = field()

    instance = TestClass.create({"x": 5, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 15
    assert instance.y == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_remove_existing_item():
    original = object()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.remove('a')
    assert result is evolver
    assert 'a' not in evolver._pclass_evolver_data
    assert 'b' in evolver._pclass_evolver_data
    assert 'a' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True

def test_remove_non_existing_item():
    original = object()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    try:
        evolver.remove('c')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == 'c'


# LLM-generated content at query #5
#--------------------------

```python
def test__is_pclass_with_single_checked_type_base():
    assert _is_pclass((CheckedType,)) == True

def test__is_pclass_with_multiple_bases():
    assert _is_pclass((CheckedType, object)) == False

def test__is_pclass_with_no_bases():
    assert _is_pclass(()) == False

def test__is_pclass_with_non_checked_type_base():
    assert _is_pclass((object,)) == False


# LLM-generated content at query #6
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
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

def test_pclass_new_with_invalid_field_type():
    from pyrsistent import PClass, field, PTypeError

    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
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

    def custom_factory(value):
        return value * 2

    class TestClass(PClass):
        x = field(factory=custom_factory)
        y = field()

    instance = TestClass(x=5, y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

def test_pclass_new_with_invariant():
    from pyrsistent import PClass, field, InvariantException

    def invariant(value):
        return value > 0, "Value must be positive"

    class TestClass(PClass):
        x = field(invariant=invariant)

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in e.error_codes

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, InvariantException

    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_invariant]

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Sum must be positive" in e.error_codes


# LLM-generated content at query #7
#--------------------------

```python
def test_pclassmeta_new_with_single_checkedtype_base():
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = field()
        field2 = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__
    assert TestClass._pclass_invariants == (wrap_invariant(lambda self: True),)
    assert TestClass._pclass_fields == {'field1': field(), 'field2': field()}

def test_pclassmeta_new_with_multiple_bases():
    class BaseClass:
        pass

    class TestClass(BaseClass, metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' not in TestClass.__slots__
    assert TestClass._pclass_invariants == (wrap_invariant(lambda self: True),)
    assert TestClass._pclass_fields == {'field1': field()}

def test_pclassmeta_new_with_inherited_invariants():
    class BaseClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = field()

    class TestClass(BaseClass):
        __invariant__ = lambda self: False
        field2 = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' not in TestClass.__slots__
    assert len(TestClass._pclass_invariants) == 2
    assert TestClass._pclass_fields == {'field1': field(), 'field2': field()}

def test_pclassmeta_new_with_non_callable_invariant():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = "not callable"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"
    else:
        assert False, "Expected TypeError"

def test_pclassmeta_new_with_no_invariant():
    class TestClass(metaclass=PClassMeta):
        field1 = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__
    assert TestClass._pclass_invariants == ()
    assert TestClass._pclass_fields == {'field1': field()}


# LLM-generated content at query #8
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


# LLM-generated content at query #10
#--------------------------

```python
def test_pclass_repr():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"

    empty_instance = TestClass()
    assert repr(empty_instance) == "TestClass()"

    class NestedClass(PClass):
        a = field()
        b = field()

    nested = NestedClass(a="hello", b=3.14)
    assert repr(nested) == "NestedClass(a='hello', b=3.14)"


# LLM-generated content at query #11
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
        return f"serialized_{value}"

    class TestClass(PClass):
        x = field(serializer=custom_serializer)
        y = field()

    instance = TestClass(x=1, y="test")
    assert instance.serialize() == {"x": "serialized_1", "y": "test"}

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


# LLM-generated content at query #12
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

def test_pclass_constructor_with_extra_field():
    class TestPClass(PClass):
        x = field()

    try:
        TestPClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_initial_value():
    class TestPClass(PClass):
        x = field(initial=5)
        y = field()

    instance = TestPClass(y=2)
    assert instance.x == 5
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    class TestPClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestPClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_ignore_extra():
    class TestPClass(PClass):
        x = field()

    instance = TestPClass.create({"x": 1, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_factory_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #13
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

def test_set_with_positional_arguments():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set('x', 10)
    assert new_instance.x == 10
    assert new_instance.y == 2

def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert new_instance.z == 3

def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    new_instance = instance.set(x=2)
    assert instance is not new_instance
    assert instance.x == 1
    assert new_instance.x == 2

def test_set_with_multiple_updates():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10, y=20)
    assert new_instance.x == 10
    assert new_instance.y == 20


# LLM-generated content at query #14
#--------------------------

```python
def test_pclass_equality_with_same_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

def test_pclass_equality_with_different_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    assert not (instance1 == instance2)

def test_pclass_equality_with_different_types():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()

    class TestClass2(PClass):
        x = field()

    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    assert not (instance1 == instance2)

def test_pclass_equality_with_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert not (instance == 1)
    assert not (instance == {"x": 1})


# LLM-generated content at query #15
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_exception():
    from pyrsistent import PClass, field, mandatory, invariant

    class TestClass(PClass):
        x = field(type=int, invariant=lambda x: (True, None) if x > 0 else (False, "x must be positive"))
        y = field(mandatory=True)

    with pytest.raises(InvariantException):
        TestClass(x=-1, y=1)

    with pytest.raises(InvariantException):
        TestClass(x=1)


# LLM-generated content at query #16
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

def test_pclass_hash_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)

    assert hash(instance1) == hash(instance2)

def test_pclass_hash_with_different_field_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)

    assert hash(instance1) != hash(instance2)


# LLM-generated content at query #17
#--------------------------

```python
def test_pclass_pickle_reduce_returns_correct_tuple():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    result = instance.__reduce__()
    assert result == (_restore_pickle, (TestPClass, {'x': 1, 'y': 2}))


# LLM-generated content at query #18
#--------------------------

```python
def test_persistent_with_dirty_data():
    original = type('MockClass', (), {})()
    evolver = _PClassEvolver(original, {'a': 1, 'b': 2})
    evolver.set('a', 3)
    result = evolver.persistent()
    assert isinstance(result, original.__class__)
    assert result.a == 3
    assert result.b == 2
    assert result._factory_fields == {'a'}

def test_persistent_with_clean_data():
    original = type('MockClass', (), {})()
    evolver = _PClassEvolver(original, {'a': 1, 'b': 2})
    result = evolver.persistent()
    assert result is original

def test_persistent_after_removal():
    original = type('MockClass', (), {})()
    evolver = _PClassEvolver(original, {'a': 1, 'b': 2})
    evolver.remove('a')
    result = evolver.persistent()
    assert isinstance(result, original.__class__)
    assert not hasattr(result, 'a')
    assert result.b == 2
    assert result._factory_fields == set()


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_persistent_returns_original_when_not_dirty():
    original = object()
    evolver = _PClassEvolver(original, {})
    assert evolver.persistent() is original

def test_persistent_returns_new_instance_when_dirty():
    original = object()
    evolver = _PClassEvolver(original, {})
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert result is not original
    assert result._factory_fields == {'key'}
    assert result.key == 'value'

def test_persistent_includes_all_factory_fields():
    original = object()
    evolver = _PClassEvolver(original, {})
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    result = evolver.persistent()
    assert result._factory_fields == {'field1', 'field2'}
    assert result.field1 == 'value1'
    assert result.field2 == 'value2'

def test_persistent_excludes_removed_fields():
    original = object()
    evolver = _PClassEvolver(original, {'field1': 'value1', 'field2': 'value2'})
    evolver.remove('field1')
    result = evolver.persistent()
    assert 'field1' not in result._factory_fields
    assert result.field2 == 'value2'


# LLM-generated content at query #21
#--------------------------

```python
def test_check_and_set_attr_with_valid_type_and_invariant():
    class MockField:
        type = (int,)
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
        type = (int,)
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
        assert e.field_name == "attr"
        assert e.field_type == (int,)
        assert e.actual_type == str

def test_check_and_set_attr_with_failing_invariant():
    class MockField:
        type = (int,)
        def invariant(self, value):
            return False, "INVALID"

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField(), "attr", 42, result, invariant_errors)
    assert not hasattr(result, "attr")
    assert invariant_errors == ["INVALID"]

def test_check_and_set_attr_with_string_type():
    class MockField:
        type = ("builtins.int",)
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField(), "attr", 42, result, invariant_errors)
    assert result.attr == 42
    assert invariant_errors == []

def test_check_and_set_attr_with_custom_type():
    class CustomType:
        pass

    class MockField:
        type = (CustomType,)
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField(), "attr", CustomType(), result, invariant_errors)
    assert isinstance(result.attr, CustomType)
    assert invariant_errors == []


# LLM-generated content at query #22
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

    instance = TestClass.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')


# LLM-generated content at query #23
#--------------------------

```python
def test_hash_returns_consistent_value():
    from pyrsistent import PClass, field
    class TestPClass(PClass):
        x = field()
        y = field()

    instance1 = TestPClass(x=1, y=2)
    instance2 = TestPClass(x=1, y=2)
    assert hash(instance1) == hash(instance2)

def test_hash_different_instances_different_values():
    from pyrsistent import PClass, field
    class TestPClass(PClass):
        x = field()
        y = field()

    instance1 = TestPClass(x=1, y=2)
    instance2 = TestPClass(x=3, y=4)
    assert hash(instance1) != hash(instance2)

def test_hash_with_missing_optional_field():
    from pyrsistent import PClass, field
    class TestPClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance1 = TestPClass(x=1)
    instance2 = TestPClass(x=1)
    assert hash(instance1) == hash(instance2)

def test_hash_with_initial_value():
    from pyrsistent import PClass, field
    class TestPClass(PClass):
        x = field(initial=0)
        y = field()

    instance1 = TestPClass(y=2)
    instance2 = TestPClass(y=2)
    assert hash(instance1) == hash(instance2)


# LLM-generated content at query #24
#--------------------------

```python
def test_weakref_in_slots_when_bases_are_pclass():
    class Base(metaclass=PClassMeta):
        pass

    class Child(Base):
        pass

    assert '__weakref__' in Child.__slots__


# LLM-generated content at query #25
#--------------------------

```python
def test_eq_with_same_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance == instance

def test_eq_with_equivalent_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

def test_eq_with_different_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    assert not (instance1 == instance2)

def test_eq_with_different_class():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    assert not (instance1 == instance2)

def test_eq_with_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert not (instance == "not a PClass")

def test_eq_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1, y=2)
    assert not (instance1 == instance2)


# LLM-generated content at query #26
#--------------------------

```python
def test_invariant_errors_or_missing_fields_predicate():
    from pyrsistent import PClass, field, invariant, mandatory
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


# LLM-generated content at query #27
#--------------------------

```python
def test_pclass_fields_iteration():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    test_instance = TestClass(x=1)
    fields = TestClass._pclass_fields
    assert len(fields) > 0
    assert all(isinstance(name, str) for name, _ in fields.items())
    assert all(hasattr(field, 'type') for _, field in fields.items())


# LLM-generated content at query #28
#--------------------------

```python
def test_weakref_in_slots_when_bases_are_pclass():
    class Parent(metaclass=PClassMeta):
        pass

    class Child(Parent):
        pass

    assert '__weakref__' in Child.__slots__


# LLM-generated content at query #29
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

def test_pclass_eq_with_non_pclass():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert not (instance == 1)
    assert not (instance == {"x": 1})


# LLM-generated content at query #30
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._field_common import PFIELD_NO_INITIAL

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(initial=0)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('TestClass.x',)


# LLM-generated content at query #31
#--------------------------

```python
def test_pclass_hash_consistency():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


# LLM-generated content at query #32
#--------------------------

```python
def test_weakref_in_slots_when_bases_are_pclass():
    class Base(metaclass=PClassMeta):
        pass

    class Derived(Base):
        pass

    assert '__weakref__' in Derived.__slots__


# LLM-generated content at query #33
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


# LLM-generated content at query #34
#--------------------------

```python
def test_repr_returns_correct_format():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert repr(instance) == "TestPClass(x=1, y=2)"


# LLM-generated content at query #35
#--------------------------

```python
def test_pclass_pickling_support():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    reduced = instance.__reduce__()
    assert reduced[0] is _restore_pickle
    assert reduced[1][0] is TestClass
    assert reduced[1][1] == {'x': 1, 'y': 2}


# LLM-generated content at query #36
#--------------------------

```python
def test_serialize_returns_dict_with_serialized_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    result = instance.serialize()
    assert isinstance(result, dict)
    assert result == {'x': 1, 'y': 2}


# LLM-generated content at query #37
#--------------------------

```python
def test_check_and_set_attr_with_valid_type_and_invariant():
    class TestClass:
        pass

    class Field:
        def __init__(self, type, invariant):
            self.type = type
            self.invariant = invariant

    field = Field(type=int, invariant=lambda x: (True, None))
    result = TestClass()
    invariant_errors = []

    _check_and_set_attr(TestClass, field, "test_field", 42, result, invariant_errors)

    assert result.test_field == 42
    assert invariant_errors == []

def test_check_and_set_attr_with_invalid_type():
    class TestClass:
        pass

    class Field:
        def __init__(self, type, invariant):
            self.type = type
            self.invariant = invariant

    field = Field(type=int, invariant=lambda x: (True, None))
    result = TestClass()
    invariant_errors = []

    try:
        _check_and_set_attr(TestClass, field, "test_field", "not_an_int", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert str(e) == "Invalid type for field TestClass.test_field, was str"

def test_check_and_set_attr_with_failed_invariant():
    class TestClass:
        pass

    class Field:
        def __init__(self, type, invariant):
            self.type = type
            self.invariant = invariant

    field = Field(type=int, invariant=lambda x: (False, "INVALID") if x < 0 else (True, None))
    result = TestClass()
    invariant_errors = []

    _check_and_set_attr(TestClass, field, "test_field", -1, result, invariant_errors)

    assert not hasattr(result, "test_field")
    assert invariant_errors == ["INVALID"]


# LLM-generated content at query #38
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


# LLM-generated content at query #39
#--------------------------

```python
def test__is_pclass_returns_false_for_non_pclass_bases():
    assert not _is_pclass((object,))


# LLM-generated content at query #40
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


# LLM-generated content at query #41
#--------------------------

```python
def test_equality_with_same_class_and_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2


# LLM-generated content at query #42
#--------------------------

```python
def test_repr_format_matches_expected():
    assert PClass.__repr__(PClass()) == "{0}({1})".format(PClass.__name__, ', '.join('{0}={1}'.format(k, repr(v)) for k, v in PClass._to_dict(PClass()).items()))


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


# LLM-generated content at query #44
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
        x = field(initial=5)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 5
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


# LLM-generated content at query #45
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    invariant_errors = ["error1"]
    missing_fields = ["field1"]
    assert invariant_errors or missing_fields


# LLM-generated content at query #46
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, mandatory, invariant
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(type=int, mandatory=True)
        y = field(type=int, invariant=lambda v: (v > 0, "y must be positive"))

    try:
        TestClass()  # Missing mandatory field x
    except InvariantException as e:
        assert e.args[1] == ('TestClass.x',)  # missing_fields is not empty
        assert e.args[0] == ()  # invariant_errors is empty
        assert e.args[2] == 'Field invariant failed'

    try:
        TestClass(x=1, y=-1)  # y invariant fails
    except InvariantException as e:
        assert e.args[0] == ('y must be positive',)  # invariant_errors is not empty
        assert e.args[1] == ()  # missing_fields is empty
        assert e.args[2] == 'Field invariant failed'


# LLM-generated content at query #47
#--------------------------

```python
def test_check_and_set_attr_with_valid_invariant():
    cls = type('TestClass', (), {})
    field = type('Field', (), {'type': None, 'invariant': lambda x: (True, None)})()
    name = 'test_field'
    value = 'test_value'
    result = object()
    invariant_errors = []

    _check_and_set_attr(cls, field, name, value, result, invariant_errors)

    assert getattr(result, name) == value
    assert invariant_errors == []


# LLM-generated content at query #48
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

    instance = TestClass._create({'x': 1, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

def test_pclass_constructor_with_invariant_check():
    from pyrsistent import PClass, field, invariant
    def check_positive(instance):
        if instance.x <= 0:
            raise ValueError("x must be positive")

    class TestClass(PClass):
        x = field()
        _pclass_invariants = [check_positive]

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)


# LLM-generated content at query #49
#--------------------------

```python
def test__is_pclass_returns_false_for_non_pclass_bases():
    class NonPClassBase:
        pass

    bases = (NonPClassBase,)
    assert not _is_pclass(bases)


# LLM-generated content at query #50
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


# LLM-generated content at query #51
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


# LLM-generated content at query #52
#--------------------------

```python
def test_weakref_added_to_slots_when_bases_is_pclass():
    class Base(metaclass=PClassMeta):
        pass

    class Child(Base):
        pass

    assert '__weakref__' in Child.__slots__


# LLM-generated content at query #53
#--------------------------

```python
def test_pclass_fields_iteration():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    assert list(TestClass._pclass_fields.items()) == [('x', TestClass._pclass_fields['x']), ('y', TestClass._pclass_fields['y'])]


# LLM-generated content at query #54
#--------------------------

```python
def test_serialize_includes_all_fields():
    class TestPClass(PClass):
        field1 = field()
        field2 = field()

    instance = TestPClass(field1=1, field2=2)
    serialized = instance.serialize()
    assert 'field1' in serialized
    assert 'field2' in serialized


# LLM-generated content at query #55
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, v, invariant
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(invariant=lambda v: (v > 0, "y must be positive"))

    try:
        TestClass(x=1, y=-1)
    except InvariantException as e:
        assert e.args[0] == ("y must be positive",)
        assert e.args[1] == ()
        assert e.args[2] == 'Field invariant failed'

    try:
        TestClass(y=1)
    except InvariantException as e:
        assert e.args[0] == ()
        assert e.args[1] == ("TestClass.x",)
        assert e.args[2] == 'Field invariant failed'


# LLM-generated content at query #56
#--------------------------

```python
def test_serialize_with_no_serializer():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="hello")
    assert instance.serialize() == {'x': 1, 'y': "hello"}

def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field

    def custom_serializer(value):
        return str(value).upper()

    class TestClass(PClass):
        x = field(serializer=custom_serializer)
        y = field()

    instance = TestClass(x="hello", y=1)
    assert instance.serialize() == {'x': "HELLO", 'y': 1}

def test_serialize_with_format():
    from pyrsistent import PClass, field

    def format_serializer(format, value):
        if format == "json":
            return f'"{value}"'
        return value

    class TestClass(PClass):
        x = field(serializer=format_serializer)
        y = field()

    instance = TestClass(x="hello", y=1)
    assert instance.serialize(format="json") == {'x': '"hello"', 'y': 1}

def test_serialize_with_missing_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1)
    assert instance.serialize() == {'x': 1}

def test_serialize_with_none_values():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=None, y=1)
    assert instance.serialize() == {'x': None, 'y': 1}


# LLM-generated content at query #57
#--------------------------

```python
def test__is_pclass_bases_returns_false():
    class NonPClass:
        pass

    bases = (NonPClass,)
    assert not _is_pclass(bases)


# LLM-generated content at query #58
#--------------------------

```python
def test_pclass_hash_consistency():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    instance1 = TestPClass(x=1, y=2)
    instance2 = TestPClass(x=1, y=2)
    instance3 = TestPClass(x=1, y=3)

    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)


# LLM-generated content at query #59
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_exception():
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "must be positive"))

    with pytest.raises(InvariantException):
        TestClass(x=-1)

    class TestClass2(PClass):
        y = field(mandatory=True)

    with pytest.raises(InvariantException):
        TestClass2()


# LLM-generated content at query #60
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


# LLM-generated content at query #61
#--------------------------

```python
def test_check_and_set_attr_with_invalid_invariant():
    class MockField:
        type = None
        def invariant(self, value):
            return (False, "error_code")

    cls = type('MockClass', (), {})
    field = MockField()
    name = "test_field"
    value = "test_value"
    result = object()
    invariant_errors = []

    _check_and_set_attr(cls, field, name, value, result, invariant_errors)

    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "error_code"


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

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2

def test_pclass_constructor_with_invariant_check():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

    def invariant_func(instance):
        if instance.x < 0:
            raise ValueError("x must be non-negative")

    class TestClass(PClass):
        x = field(invariant=invariant_func)
        y = field()

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)


# LLM-generated content at query #63
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field, v, s, invariant
    class TestClass(PClass):
        x = field(type=int, invariant=lambda x: (x > 0, 'x must be positive'))
        y = field(type=str, initial='default')
        z = field(mandatory=True)

    instance = TestClass(x=1, z=3)
    assert instance.x == 1
    assert instance.y == 'default'
    assert instance.z == 3

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
        TestClass(x='not an int')
        assert False, "Expected PTypeError"
    except Exception as e:
        assert "Invalid type" in str(e)

def test_pclass_new_with_invalid_invariant():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(invariant=lambda x: (x > 0, 'x must be positive'))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "x must be positive" in str(e)

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)

    instance = TestClass(x=2)
    assert instance.x == 4

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v, ignore_extra=False: v if not ignore_extra else v.upper(), type=str)

    instance = TestClass.create({'x': 'test', 'y': 'extra'}, ignore_extra=True)
    assert instance.x == 'TEST'
    assert not hasattr(instance, 'y')

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    def global_inv(obj):
        return (obj.x + obj.y == 10, 'sum must be 10')

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_inv]

    try:
        TestClass(x=5, y=4)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Global invariant failed" in str(e)

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)

    instance = TestClass()
    assert instance.x == 42


# LLM-generated content at query #64
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
    _check_and_set_attr(TestClass, Field(), "attr", 42, result, [])
    assert result.attr == 42

def test_check_and_set_attr_with_invalid_type():
    class TestClass:
        pass

    class Field:
        type = int
        def invariant(self, value):
            return True, None

    result = TestClass()
    try:
        _check_and_set_attr(TestClass, Field(), "attr", "not_an_int", result, [])
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert str(e) == "Invalid type for field TestClass.attr, was str"

def test_check_and_set_attr_with_failed_invariant():
    class TestClass:
        pass

    class Field:
        type = int
        def invariant(self, value):
            return False, "INVALID"

    result = TestClass()
    errors = []
    _check_and_set_attr(TestClass, Field(), "attr", 42, result, errors)
    assert errors == ["INVALID"]
    assert not hasattr(result, "attr")

def test_check_and_set_attr_with_string_type():
    class TestClass:
        pass

    class Field:
        type = "builtins.int"
        def invariant(self, value):
            return True, None

    result = TestClass()
    _check_and_set_attr(TestClass, Field(), "attr", 42, result, [])
    assert result.attr == 42

def test_check_and_set_attr_with_multiple_types():
    class TestClass:
        pass

    class Field:
        type = [int, str]
        def invariant(self, value):
            return True, None

    result = TestClass()
    _check_and_set_attr(TestClass, Field(), "attr", "string", result, [])
    assert result.attr == "string"


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


# LLM-generated content at query #2
#--------------------------

```python
def test_pclass_repr():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"

    empty_instance = TestClass(x=0, y=None)
    assert repr(empty_instance) == "TestClass(x=0, y=None)"

    single_field_instance = TestClass(x="test")
    assert repr(single_field_instance) == "TestClass(x='test')"


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restored_obj = _restore_pickle(*obj.__reduce__())
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #5
#--------------------------

```python
def test_serialize_without_custom_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize() == {'x': 1, 'y': 2}

def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    def custom_serializer(value):
        return str(value)

    class TestClass(PClass):
        x = field(serializer=custom_serializer)
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize() == {'x': '1', 'y': 2}

def test_serialize_with_missing_optional_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance = TestClass(x=1)
    assert instance.serialize() == {'x': 1}

def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize(format='json') == {'x': 1, 'y': 2}


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field, serialize

    class CustomClass(PClass):
        x = field(serializer=lambda x: x * 2)

    obj = CustomClass(x=5)
    assert obj.serialize() == {'x': 10}


# LLM-generated content at query #7
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


# LLM-generated content at query #8
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

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass._factory_fields={'x'}, x=1
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


# LLM-generated content at query #9
#--------------------------

```python
def test_pclass_hash_equality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert hash(obj1) == hash(obj2)

def test_pclass_hash_inequality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=3, y=4)
    assert hash(obj1) != hash(obj2)

def test_pclass_hash_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    assert hash(obj1) == hash(obj2)

def test_pclass_hash_with_different_types():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()

    class TestClass2(PClass):
        x = field()

    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    assert hash(obj1) != hash(obj2)


# LLM-generated content at query #10
#--------------------------

```python
def test_eq_returns_true_for_equal_instances():
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

def test_eq_returns_false_for_different_instances():
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    assert not (instance1 == instance2)

def test_eq_returns_not_implemented_for_non_pclass():
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert instance.__eq__("not a PClass") is NotImplemented


# LLM-generated content at query #11
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

def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except Exception as e:
        assert "Invalid type for field" in str(e)

def test_pclass_new_with_factory():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
    instance = TestClass(x=5)
    assert instance.x == 10

def test_pclass_new_with_invariant():
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
    def global_inv(obj):
        return obj.x + obj.y > 0, "sum must be positive"
    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_inv]
    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Global invariant failed" in str(e)

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
        x = field(factory=lambda v, ignore_extra=False: v * 2)
    instance = TestClass(x=5, _factory_fields={"x"})
    assert instance.x == 10


# LLM-generated content at query #12
#--------------------------

```python
def test_pclassmeta_new_with_checkedtype_base():
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = _PField()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'field1' in TestClass.__slots__
    assert isinstance(TestClass._pclass_invariants, tuple)
    assert len(TestClass._pclass_invariants) == 1

def test_pclassmeta_new_without_checkedtype_base():
    class BaseClass:
        pass

    class TestClass(BaseClass, metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = _PField()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' not in TestClass.__slots__
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'field1' in TestClass.__slots__
    assert isinstance(TestClass._pclass_invariants, tuple)
    assert len(TestClass._pclass_invariants) == 1

def test_pclassmeta_new_with_inherited_fields():
    class BaseClass(metaclass=PClassMeta):
        field1 = _PField()

    class TestClass(BaseClass):
        field2 = _PField()

    assert 'field1' in TestClass._pclass_fields
    assert 'field2' in TestClass._pclass_fields
    assert 'field1' in TestClass.__slots__
    assert 'field2' in TestClass.__slots__

def test_pclassmeta_new_with_inherited_invariants():
    class BaseClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True

    class TestClass(BaseClass):
        __invariant__ = lambda self: True

    assert len(TestClass._pclass_invariants) == 2
    assert all(callable(inv) for inv in TestClass._pclass_invariants)

def test_pclassmeta_new_with_non_callable_invariant():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = "not callable"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

def test_pclassmeta_new_with_multiple_bases():
    class Base1(metaclass=PClassMeta):
        field1 = _PField()

    class Base2(metaclass=PClassMeta):
        field2 = _PField()

    class TestClass(Base1, Base2):
        field3 = _PField()

    assert 'field1' in TestClass._pclass_fields
    assert 'field2' in TestClass._pclass_fields
    assert 'field3' in TestClass._pclass_fields
    assert 'field1' in TestClass.__slots__
    assert 'field2' in TestClass.__slots__
    assert 'field3' in TestClass.__slots__


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_evaluates_to_false():
    dct = {'_pclass_fields': {}}
    bases = (object,)
    result = _is_pclass(bases)
    assert result is False


# LLM-generated content at query #14
#--------------------------

```python
def test_check_and_set_attr_with_valid_type_and_invariant():
    class TestClass:
        pass

    class Field:
        type = (int,)
        def invariant(self, value):
            return True, None

    result = TestClass()
    invariant_errors = []
    _check_and_set_attr(TestClass, Field(), 'attr', 42, result, invariant_errors)
    assert result.attr == 42
    assert invariant_errors == []

def test_check_and_set_attr_with_invalid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)
        def invariant(self, value):
            return True, None

    result = TestClass()
    invariant_errors = []
    try:
        _check_and_set_attr(TestClass, Field(), 'attr', 'not an int', result, invariant_errors)
    except PTypeError as e:
        assert e.message == "Invalid type for field TestClass.attr, was str"
    else:
        assert False, "Expected PTypeError"

def test_check_and_set_attr_with_failed_invariant():
    class TestClass:
        pass

    class Field:
        type = (int,)
        def invariant(self, value):
            return False, "INVALID"

    result = TestClass()
    invariant_errors = []
    _check_and_set_attr(TestClass, Field(), 'attr', 42, result, invariant_errors)
    assert not hasattr(result, 'attr')
    assert invariant_errors == ["INVALID"]


# LLM-generated content at query #15
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

def test_pclass_constructor_with_invalid_field_value():
    class TestClass(PClass):
        x = field(invariant=lambda x: x > 0)
        y = field()

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)


# LLM-generated content at query #16
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    invariant_errors = ["error1"]
    missing_fields = ["field1"]
    assert invariant_errors or missing_fields


# LLM-generated content at query #17
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

def test_serialize_with_format():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v, fmt: f"{v}_{fmt}" if fmt else str(v))
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize(format='json') == {'x': '1_json', 'y': 2}

def test_serialize_with_missing_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1)
    assert instance.serialize() == {'x': 1}


# LLM-generated content at query #18
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

def test_serialize_with_format():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v, fmt: v if fmt is None else v * 2)
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize(format='custom') == {'x': 2, 'y': 2}

def test_serialize_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1)
    assert instance.serialize() == {'x': 1}


# LLM-generated content at query #19
#--------------------------

```python
def test_equality_with_same_class_and_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2


# LLM-generated content at query #20
#--------------------------

```python
def test_weakref_added_to_slots_for_pclass_bases():
    class Base(metaclass=PClassMeta):
        pass

    class Derived(Base):
        pass

    assert '__weakref__' in Derived.__slots__


# LLM-generated content at query #21
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._pclass import _check_and_set_attr

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    # Test with missing mandatory field
    try:
        TestClass(y=1)
    except InvariantException as e:
        assert e.args[1] == ('TestClass.x',)

    # Test with invariant error
    class TestClassWithInvariant(PClass):
        x = field()
        _pclass_invariants = [lambda self: (False, 'error')]

    try:
        TestClassWithInvariant(x=1)
    except InvariantException as e:
        assert e.args[0] == ('error',)


# LLM-generated content at query #22
#--------------------------

```python
def test_eq_returns_true_for_identical_pclass_instances():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance1 = TestPClass(x=1, y=2)
    instance2 = TestPClass(x=1, y=2)
    assert instance1 == instance2


# LLM-generated content at query #23
#--------------------------

```python
def test_check_and_set_attr_with_invalid_invariant():
    cls = type('MockClass', (), {})
    field = type('MockField', (), {'invariant': lambda self, value: (False, "error")})()
    name = "test_field"
    value = "test_value"
    result = object()
    invariant_errors = []

    _check_and_set_attr(cls, field, name, value, result, invariant_errors)

    assert invariant_errors == ["error"]
    assert not hasattr(result, name)


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


# LLM-generated content at query #25
#--------------------------

```python
def test_pclass_reduce_returns_restore_pickle_and_class_data_tuple():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    result = instance.__reduce__()

    assert result[0] == _restore_pickle
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1}


# LLM-generated content at query #26
#--------------------------

```python
def test_pclass_hash_returns_consistent_value():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert hash(instance) == hash(instance)

def test_pclass_hash_different_instances_same_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert hash(instance1) == hash(instance2)

def test_pclass_hash_different_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=2, y=1)
    assert hash(instance1) != hash(instance2)

def test_pclass_hash_missing_optional_field():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(initial=0)

    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1, y=0)
    assert hash(instance1) == hash(instance2)

def test_pclass_hash_with_no_fields():
    from pyrsistent import PClass

    class EmptyClass(PClass):
        pass

    instance = EmptyClass()
    assert isinstance(hash(instance), int)


# LLM-generated content at query #27
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
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing

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
    from pyrsistent import PClass, field, InvariantException

    def positive_invariant(value):
        return value > 0, "must_be_positive"

    class TestClass(PClass):
        x = field(invariant=positive_invariant)

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "must_be_positive" in e.errors

def test_pclass_new_with_factory_and_ignore_extra():
    from pyrsistent import PClass, field

    def factory(value, ignore_extra=False):
        return value * 2

    class TestClass(PClass):
        x = field(type=int, factory=factory)

    instance = TestClass.create({"x": 5, "z": 10}, ignore_extra=True)
    assert instance.x == 10

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException

    def global_invariant(instance):
        return instance.x < instance.y, "x_must_be_less_than_y"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_invariant]

    try:
        TestClass(x=5, y=3)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x_must_be_less_than_y" in e.errors

def test_pclass_new_with_valid_type():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(type=int)

    instance = TestClass(x=42)
    assert instance.x == 42

def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field, PTypeError

    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not_an_int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert "Invalid type" in str(e)

def test_pclass_new_with_multiple_types():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(type={int, str})

    instance1 = TestClass(x=42)
    assert instance1.x == 42

    instance2 = TestClass(x="hello")
    assert instance2.x == "hello"

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field

    def factory(value):
        return value.upper()

    class TestClass(PClass):
        x = field(factory=factory)
        y = field()

    instance = TestClass(_factory_fields={"x"}, x="hello", y=2)
    assert instance.x == "HELLO"
    assert instance.y == 2


# LLM-generated content at query #28
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

def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    def custom_factory(value):
        return value * 2

    class TestClass(PClass):
        x = field(factory=custom_factory)
        y = field()

    instance = TestClass(x=5, y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_invariant():
    from pyrsistent import PClass, field
    def invariant(value):
        return value > 0, "Value must be positive"

    class TestClass(PClass):
        x = field(invariant=invariant)
        y = field()

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    @invariant
    def check_sum(obj):
        return obj.x + obj.y > 0, "Sum must be positive"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [check_sum]

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Global invariant failed" in str(e)

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    def custom_factory(value, ignore_extra=False):
        return value.upper() if ignore_extra else value.lower()

    class TestClass(PClass):
        x = field(factory=custom_factory)
        y = field()

    instance = TestClass.create({'x': 'hello', 'y': 2}, _factory_fields={'x'}, ignore_extra=True)
    assert instance.x == 'HELLO'
    assert instance.y == 2

def test_pclass_new_with_type_check():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
        y = field()

    try:
        TestClass(x='not an int', y=2)
        assert False, "Expected PTypeError"
    except Exception as e:
        assert "Invalid type for field" in str(e)


# LLM-generated content at query #29
#--------------------------

```python
def test_repr_returns_correct_string():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y="test")
    assert repr(instance) == "TestClass(x=1, y='test')"


# LLM-generated content at query #30
#--------------------------

```python
def test_equality_with_same_class_and_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance1 = TestPClass(x=1, y=2)
    instance2 = TestPClass(x=1, y=2)
    assert instance1 == instance2


# LLM-generated content at query #31
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
    new_instance = instance.set(z=30)
    assert new_instance.x == 1
    assert new_instance.y == 2
    assert not hasattr(new_instance, 'z')

def test_set_with_empty_kwargs():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set()
    assert new_instance.x == 1
    assert new_instance.y == 2


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

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException

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

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e) and "TestClass" in str(e)

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

    instance = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


# LLM-generated content at query #33
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
        assert "TestClass.x" in e.missing_fields

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z'" in str(e)
        assert "TestClass" in str(e)

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

def test_pclass_new_with_invariant_violation():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

    def invariant(value):
        return (value > 0, "must_be_positive")

    class TestClass(PClass):
        x = field(invariant=invariant)

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "must_be_positive" in e.error_codes

def test_pclass_new_with_factory_and_ignore_extra():
    from pyrsistent import PClass, field

    def factory(value, ignore_extra=False):
        if ignore_extra:
            return value.upper()
        return value

    class TestClass(PClass):
        x = field(factory=factory)

    instance = TestClass.create({"x": "hello", "z": "extra"}, ignore_extra=True)
    assert instance.x == "HELLO"

def test_pclass_new_with_global_invariant_violation():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._pclass import InvariantException

    @invariant
    def global_inv(obj):
        return obj.x != obj.y, "x_and_y_must_differ"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_inv]

    try:
        TestClass(x=1, y=1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x_and_y_must_differ" in e.error_codes

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field

    def factory(value):
        return value * 2

    class TestClass(PClass):
        x = field(factory=factory)
        y = field()

    instance = TestClass(_factory_fields={"x"}, x=5, y=10)
    assert instance.x == 10
    assert instance.y == 10

def test_pclass_new_with_type_check():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError

    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not_an_int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert "Invalid type for field TestClass.x, was str" in str(e)

def test_pclass_new_with_multiple_types():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(type={int, str})

    instance1 = TestClass(x=1)
    assert instance1.x == 1

    instance2 = TestClass(x="hello")
    assert instance2.x == "hello"


# LLM-generated content at query #34
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


# LLM-generated content at query #35
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_exception():
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)

    try:
        TestClass()
    except InvariantException as e:
        assert e.args[1] == ('TestClass.x',)
        assert e.args[2] == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #36
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
        assert e.args[1] == ("TestClass.x",)

def test_pclass_new_with_invalid_field_type():
    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.args[2] == (int,)
        assert e.args[3] == str

def test_pclass_new_with_extra_fields():
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

def test_pclass_new_with_ignore_extra():
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

def test_pclass_new_with_factory_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_initial_value():
    class TestClass(PClass):
        x = field(initial=10)

    instance = TestClass()
    assert instance.x == 10

def test_pclass_new_with_callable_initial():
    class TestClass(PClass):
        x = field(initial=lambda: 20)

    instance = TestClass()
    assert instance.x == 20

def test_pclass_new_with_invariant_failure():
    def invariant(value):
        return (value > 0, "must be positive")

    class TestClass(PClass):
        x = field(invariant=invariant)

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("must be positive",)

def test_pclass_new_with_global_invariant_failure():
    def global_invariant(instance):
        return (instance.x != instance.y, "x and y must differ")

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_invariant]

    try:
        TestClass(x=1, y=1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[2] == "Global invariant failed"

def test_pclass_new_with_factory_and_ignore_extra():
    def factory(value, ignore_extra=False):
        return value if ignore_extra else value * 2

    class TestClass(PClass):
        x = field(factory=factory, type={int, str})

    instance = TestClass.create({"x": 5}, ignore_extra=True)
    assert instance.x == 5

    instance = TestClass(x=5)
    assert instance.x == 10


# LLM-generated content at query #37
#--------------------------

```python
def test_pclass_hash_consistency():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=1, y=3)

    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)

def test_pclass_hash_with_missing_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1, y=None)

    assert hash(instance1) == hash(instance2)

def test_pclass_hash_with_different_field_order():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(y=2, x=1)

    assert hash(instance1) == hash(instance2)


# LLM-generated content at query #38
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

def test_set_with_mandatory_field_missing():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(y=20)

    assert new_instance.x == 1
    assert new_instance.y == 20
    assert instance.x == 1
    assert instance.y == 2

def test_set_with_initial_value_field():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClass(y=2)
    new_instance = instance.set(x=10)

    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 0
    assert instance.y == 2

def test_set_with_callable_initial_value_field():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=lambda: 0)
        y = field()

    instance = TestClass(y=2)
    new_instance = instance.set(x=10)

    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 0
    assert instance.y == 2


# LLM-generated content at query #39
#--------------------------

```python
def test_pclass_hash_equality():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #40
#--------------------------

```python
def test_pclass_repr():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"

    empty_instance = TestClass()
    assert repr(empty_instance) == "TestClass()"


# LLM-generated content at query #41
#--------------------------

```python
def test_pclass_repr():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"

    instance_empty = TestClass()
    assert repr(instance_empty) == "TestClass()"


# LLM-generated content at query #42
#--------------------------

```python
def test_weakref_added_to_slots_when_bases_are_pclass():
    class Base(metaclass=PClassMeta):
        pass

    class Child(Base):
        pass

    assert '__weakref__' in Child.__slots__


# LLM-generated content at query #43
#--------------------------

```python
def test__is_pclass_bases_returns_false():
    dct = {'_pclass_fields': {}}
    bases = (object,)
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert '__weakref__' not in result.__slots__


# LLM-generated content at query #44
#--------------------------

```python
def test_check_and_set_attr_invariant_false():
    cls = object
    field = type('Field', (), {'invariant': lambda x: (False, 'error')})
    name = 'test_field'
    value = 'test_value'
    result = object()
    invariant_errors = []

    _check_and_set_attr(cls, field, name, value, result, invariant_errors)

    assert invariant_errors == ['error']
    assert not hasattr(result, name)


# LLM-generated content at query #45
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._field_common import PFIELD_NO_INITIAL

    class TestClass(PClass):
        x = field(type=int)
        y = field(type=int, mandatory=True)

    try:
        TestClass(x=1)
    except InvariantException:
        pass


# LLM-generated content at query #46
#--------------------------

```python
def test_set_preserves_existing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.y == 2


# LLM-generated content at query #47
#--------------------------

```python
def test_check_and_set_attr_with_valid_invariant():
    class MockField:
        type = None
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    _check_and_set_attr(MockClass, MockField(), "attr_name", "value", result, [])
    assert hasattr(result, "attr_name")
    assert result.attr_name == "value"


# LLM-generated content at query #48
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


# LLM-generated content at query #49
#--------------------------

```python
def test_pclass_meta_new_with_single_checked_type_base():
    bases = (CheckedType,)
    dct = {'a': 1, 'b': 2}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert result._pclass_fields == {'a': 1, 'b': 2}
    assert result.__slots__ == ('_pclass_frozen', 'a', 'b', '__weakref__')
    assert hasattr(result, '_pclass_invariants')

def test_pclass_meta_new_with_multiple_bases():
    class Base1(CheckedType):
        pass
    class Base2(CheckedType):
        pass
    bases = (Base1, Base2)
    dct = {'c': 3}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert result._pclass_fields == {'c': 3}
    assert result.__slots__ == ('_pclass_frozen', 'c')
    assert hasattr(result, '_pclass_invariants')

def test_pclass_meta_new_with_invariant():
    def test_invariant(obj):
        return True, "Test"
    bases = (CheckedType,)
    dct = {'__invariant__': test_invariant, 'x': 10}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert len(result._pclass_invariants) == 1
    assert callable(result._pclass_invariants[0])
    assert result._pclass_fields == {'x': 10}

def test_pclass_meta_new_with_inherited_fields_and_invariants():
    class Parent(CheckedType):
        __invariant__ = lambda self: (True, "Parent invariant")
        parent_field = 1
    bases = (Parent,)
    dct = {'child_field': 2}
    result = PClassMeta.__new__(PClassMeta, 'ChildClass', bases, dct)
    assert 'parent_field' in result._pclass_fields
    assert 'child_field' in result._pclass_fields
    assert len(result._pclass_invariants) == 1

def test_pclass_meta_new_with_non_pclass_bases():
    class RegularBase:
        pass
    bases = (RegularBase,)
    dct = {'field': 'value'}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert result._pclass_fields == {'field': 'value'}
    assert result.__slots__ == ('_pclass_frozen', 'field')
    assert '__weakref__' not in result.__slots__

def test_pclass_meta_new_with_pfield_in_dct():
    dct = {'regular': 1, '_pfield': _PField()}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', (CheckedType,), dct)
    assert 'regular' in result._pclass_fields
    assert '_pfield' in result._pclass_fields
    assert 'regular' not in result.__dict__
    assert '_pfield' not in result.__dict__

def test_pclass_meta_new_with_invalid_invariant_type():
    bases = (CheckedType,)
    dct = {'__invariant__': "not callable", 'field': 1}
    try:
        PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'


# LLM-generated content at query #50
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
    from pyrsistent._checks import InvariantException

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
    from pyrsistent import PClass, field
    from pyrsistent._checks import InvariantException

    def invariant_func(instance):
        if instance.x < 0:
            return "x must be non-negative"
        return None

    class TestClass(PClass):
        x = field(invariant=invariant_func)

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in e.invariant_errors

def test_pclass_constructor_with_valid_invariant():
    from pyrsistent import PClass, field

    def invariant_func(instance):
        if instance.x < 0:
            return "x must be non-negative"
        return None

    class TestClass(PClass):
        x = field(invariant=invariant_func)

    instance = TestClass(x=1)
    assert instance.x == 1


# LLM-generated content at query #51
#--------------------------

```python
def test_pickle_support_returns_correct_tuple():
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    result = instance.__reduce__()
    assert result == (_restore_pickle, (TestClass, {'x': 1}))


# LLM-generated content at query #52
#--------------------------

```python
def test_pclassmeta_new_with_checkedtype_base():
    class TestClass(metaclass=PClassMeta, bases=(CheckedType,)):
        pass
    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__

def test_pclassmeta_new_without_checkedtype_base():
    class TestClass(metaclass=PClassMeta, bases=(object,)):
        pass
    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' not in TestClass.__slots__

def test_pclassmeta_new_with_fields():
    class TestClass(metaclass=PClassMeta, bases=(CheckedType,)):
        field1 = _PField()
        field2 = _PField()
    assert 'field1' in TestClass._pclass_fields
    assert 'field2' in TestClass._pclass_fields
    assert 'field1' not in TestClass.__dict__
    assert 'field2' not in TestClass.__dict__

def test_pclassmeta_new_with_invariants():
    def invariant1(obj):
        return True, "OK"
    def invariant2(obj):
        return True, "OK"
    class TestClass(metaclass=PClassMeta, bases=(CheckedType,)):
        __invariant__ = invariant1
    class ChildClass(TestClass):
        __invariant__ = invariant2
    assert len(ChildClass._pclass_invariants) == 2
    assert all(callable(inv) for inv in ChildClass._pclass_invariants)

def test_pclassmeta_new_with_non_callable_invariant():
    try:
        class TestClass(metaclass=PClassMeta, bases=(CheckedType,)):
            __invariant__ = "not callable"
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #53
#--------------------------

```python
def test_pclassmeta_new_with_checkedtype_base():
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = field()
        field2 = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__
    assert TestClass._pclass_invariants == (wrap_invariant(lambda self: True),)
    assert TestClass._pclass_fields == {'field1': field(), 'field2': field()}

def test_pclassmeta_new_without_checkedtype_base():
    class BaseClass:
        pass

    class TestClass(BaseClass, metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' not in TestClass.__slots__
    assert TestClass._pclass_invariants == (wrap_invariant(lambda self: True),)
    assert TestClass._pclass_fields == {'field1': field()}

def test_pclassmeta_new_with_inherited_invariants():
    class BaseClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = field()

    class TestClass(BaseClass):
        __invariant__ = lambda self: False
        field2 = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__
    assert len(TestClass._pclass_invariants) == 2
    assert TestClass._pclass_fields == {'field1': field(), 'field2': field()}

def test_pclassmeta_new_with_non_callable_invariant():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = "not callable"
            field1 = field()
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"


# LLM-generated content at query #54
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
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2

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

def test_pclass_new_with_invariant_failure():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

    def invariant(value):
        return value > 0, "Value must be positive"

    class TestClass(PClass):
        x = field(invariant=invariant)

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in str(e)

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._pclass import InvariantException

    @invariant
    def check_sum(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [check_sum]

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test_pclass_new_with_factory_and_ignore_extra():
    from pyrsistent import PClass, field

    def custom_factory(value, ignore_extra=False):
        return value * 2 if ignore_extra else value

    class TestClass(PClass):
        x = field(factory=custom_factory)

    instance = TestClass.create({"x": 5}, ignore_extra=True)
    assert instance.x == 10

    instance = TestClass(x=5)
    assert instance.x == 5


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


# LLM-generated content at query #56
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
        x = field(factory=lambda v, ignore_extra=False: v.upper() if ignore_extra else v)

    instance = TestClass.create({"x": "hello"}, ignore_extra=True)
    assert instance.x == "HELLO"

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    @invariant
    def check_sum(obj):
        return obj.x + obj.y > 0, "sum_positive"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [check_sum]

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "sum_positive" in str(e)


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

def test_pclass_constructor_with_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)

    instance = TestClass(x=1)
    assert instance.x == 1

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in str(e)

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

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_invariant_check():
    from pyrsistent import PClass, field, invariant
    def check_positive(instance):
        if instance.x < 0:
            raise ValueError("x must be positive")

    class TestClass(PClass):
        x = field()
        _pclass_invariants = [check_positive]

    instance = TestClass(x=1)
    assert instance.x == 1

def test_pclass_constructor_with_invariant_violation():
    from pyrsistent import PClass, field, invariant, InvariantException
    def check_positive(instance):
        if instance.x < 0:
            raise ValueError("x must be positive")

    class TestClass(PClass):
        x = field()
        _pclass_invariants = [check_positive]

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be positive" in str(e)


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

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
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

def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=5)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 5
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

    instance = TestClass._factory_fields={"x"}, x=1, y=2
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_invariant_check():
    from pyrsistent import PClass, field, invariant

    def check_positive(instance):
        if instance.x < 0:
            raise ValueError("x must be positive")

    class TestClass(PClass):
        x = field()
        _pclass_invariants = [invariant(check_positive)]

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "InvariantException" in str(type(e).__name__)

def test_pclass_constructor_with_valid_invariant():
    from pyrsistent import PClass, field, invariant

    def check_positive(instance):
        if instance.x < 0:
            raise ValueError("x must be positive")

    class TestClass(PClass):
        x = field()
        _pclass_invariants = [invariant(check_positive)]

    instance = TestClass(x=1)
    assert instance.x == 1


# LLM-generated content at query #59
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
    _check_and_set_attr(MockClass, MockField(), 'attr', 42, result, invariant_errors)
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
        _check_and_set_attr(MockClass, MockField(), 'attr', 'not_an_int', result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field MockClass.attr, was str"

def test_check_and_set_attr_with_failing_invariant():
    class MockField:
        type = int
        def invariant(self, value):
            return False, "INVALID"

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField(), 'attr', 42, result, invariant_errors)
    assert not hasattr(result, 'attr')
    assert invariant_errors == ["INVALID"]

def test_check_and_set_attr_with_string_type():
    class MockField:
        type = "builtins.int"
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField(), 'attr', 42, result, invariant_errors)
    assert result.attr == 42
    assert invariant_errors == []

def test_check_and_set_attr_with_invalid_string_type():
    class MockField:
        type = "builtins.int"
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    try:
        _check_and_set_attr(MockClass, MockField(), 'attr', 'not_an_int', result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field MockClass.attr, was str"

def test_check_and_set_attr_with_multiple_types():
    class MockField:
        type = (int, str)
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField(), 'attr', 42, result, invariant_errors)
    assert result.attr == 42
    assert invariant_errors == []

    _check_and_set_attr(MockClass, MockField(), 'attr', 'test', result, invariant_errors)
    assert result.attr == 'test'
    assert invariant_errors == []

def test_check_and_set_attr_with_invalid_multiple_types():
    class MockField:
        type = (int, str)
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    try:
        _check_and_set_attr(MockClass, MockField(), 'attr', 3.14, result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field MockClass.attr, was float"


# LLM-generated content at query #60
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

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_initial_values():
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


# LLM-generated content at query #61
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
    def invariant_func(instance):
        if instance.x < 0:
            raise ValueError("x must be non-negative")

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [invariant_func]

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in str(e)

def test_pclass_constructor_with_valid_invariant():
    def invariant_func(instance):
        if instance.x < 0:
            raise ValueError("x must be non-negative")

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [invariant_func]

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #62
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_mandatory_field_missing():
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
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_initial_value():
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

    instance = TestClass.create({'x': 1, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

def test_pclass_constructor_with_invariant_check():
    def invariant(obj):
        if obj.x > 10:
            raise ValueError("x must be <= 10")

    class TestClass(PClass):
        x = field()
        _pclass_invariants = [invariant]

    instance = TestClass(x=5)
    assert instance.x == 5

    try:
        TestClass(x=15)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be <= 10" in str(e)


# LLM-generated content at query #63
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


# LLM-generated content at query #64
#--------------------------

```python
def test_check_and_set_attr_with_valid_type_and_invariant():
    class MockField:
        type = None
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []

    _check_and_set_attr(MockClass, MockField(), 'attr', 'value', result, invariant_errors)

    assert hasattr(result, 'attr')
    assert result.attr == 'value'
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
        x = field(initial=lambda: 42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
    assert instance.y == 2


