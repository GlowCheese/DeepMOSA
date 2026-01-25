####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2]
    check_global_invariants("test_subject", invariants)


def test_check_global_invariants_one_fails():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (False, "ERROR_CODE_1")
    
    invariants = [invariant1, invariant2]
    
    try:
        check_global_invariants("test_subject", invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


def test_check_global_invariants_multiple_fail():
    def invariant1(subject):
        return (False, "ERROR_CODE_1")
    
    def invariant2(subject):
        return (False, "ERROR_CODE_2")
    
    def invariant3(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2, invariant3]
    
    try:
        check_global_invariants("test_subject", invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1", "ERROR_CODE_2")
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


def test_check_global_invariants_empty_invariants():
    invariants = []
    check_global_invariants("test_subject", invariants)


def test_check_global_invariants_with_subject_data():
    def invariant1(subject):
        return (subject > 5, "VALUE_TOO_SMALL")
    
    def invariant2(subject):
        return (subject < 100, "VALUE_TOO_LARGE")
    
    invariants = [invariant1, invariant2]
    check_global_invariants(50, invariants)


def test_check_global_invariants_subject_fails_first_invariant():
    def invariant1(subject):
        return (subject > 100, "VALUE_TOO_SMALL")
    
    def invariant2(subject):
        return (subject < 50, "VALUE_TOO_LARGE")
    
    invariants = [invariant1, invariant2]
    
    try:
        check_global_invariants(25, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "VALUE_TOO_SMALL" in e.args[0]


# LLM-generated content at query #2
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register the type in _pmap_field_types if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import PMap
        class TestPMapField(PMap):
            __type_fields = (key_type, value_type)
        _pmap_field_types[key_type, value_type] = TestPMapField
    
    # Create test data
    test_data = {'a': 1, 'b': 2}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify result is a pmap with correct data
    assert result == pmap(test_data)
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #3
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    import inspect
    
    # Create a mock field class with factory
    class MockField:
        def __init__(self, type_val, factory):
            self.type = type_val
            self.factory = factory
    
    # Test case 1: ignore_extra is False, should return False
    mock_field_1 = MockField(str, lambda x: x)
    result_1 = is_field_ignore_extra_complaint(str, mock_field_1, False)
    assert result_1 is False
    
    # Test case 2: ignore_extra is True but field type doesn't match type_cls
    mock_field_2 = MockField(int, lambda x: x)
    result_2 = is_field_ignore_extra_complaint(str, mock_field_2, True)
    assert result_2 is False
    
    # Test case 3: ignore_extra is True, type matches, but factory has no ignore_extra parameter
    def factory_without_ignore_extra(x):
        return x
    mock_field_3 = MockField(str, factory_without_ignore_extra)
    result_3 = is_field_ignore_extra_complaint(str, mock_field_3, True)
    assert result_3 is False
    
    # Test case 4: ignore_extra is True, type matches, factory has ignore_extra parameter
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    mock_field_4 = MockField(str, factory_with_ignore_extra)
    result_4 = is_field_ignore_extra_complaint(str, mock_field_4, True)
    assert result_4 is True
    
    # Test case 5: ignore_extra is True with set type
    def factory_with_ignore_extra_2(x, ignore_extra=False):
        return x
    mock_field_5 = MockField({str}, factory_with_ignore_extra_2)
    result_5 = is_field_ignore_extra_complaint(str, mock_field_5, True)
    assert result_5 is True


# LLM-generated content at query #4
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_different_types_different_results():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2
    assert result1.__key_type__ == str
    assert result2.__key_type__ == int


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_name_format():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert "To" in result.__name__
    assert "PMap" in result.__name__
    assert result.__name__.endswith("PMap")


def test_make_pmap_field_type_with_builtin_types():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(int, float)
    assert result.__key_type__ == int
    assert result.__value_type__ == float
    assert result.__name__ == "IntToFloatPMap"


# LLM-generated content at query #5
#--------------------------

```python
def test_field_with_single_type():
    from pyrsistent._field_common import field, _PField, PFIELD_NO_FACTORY, PFIELD_NO_INVARIANT, PFIELD_NO_INITIAL, PFIELD_NO_SERIALIZER
    result = field(type=int)
    assert isinstance(result, _PField)
    assert int in result.type
    assert result.invariant == PFIELD_NO_INVARIANT
    assert result.initial == PFIELD_NO_INITIAL
    assert result.mandatory == False
    assert result._factory == PFIELD_NO_FACTORY
    assert result.serializer == PFIELD_NO_SERIALIZER


def test_field_with_multiple_types_list():
    from pyrsistent._field_common import field, _PField
    result = field(type=[int, str])
    assert isinstance(result, _PField)
    assert int in result.type
    assert str in result.type


def test_field_with_multiple_types_tuple():
    from pyrsistent._field_common import field, _PField
    result = field(type=(int, str))
    assert isinstance(result, _PField)
    assert int in result.type
    assert str in result.type


def test_field_with_multiple_types_set():
    from pyrsistent._field_common import field, _PField
    result = field(type={int, str})
    assert isinstance(result, _PField)
    assert int in result.type
    assert str in result.type


def test_field_with_string_type():
    from pyrsistent._field_common import field, _PField
    result = field(type='MyCustomType')
    assert isinstance(result, _PField)
    assert 'MyCustomType' in result.type


def test_field_with_invariant():
    from pyrsistent._field_common import field, _PField
    def my_invariant(x):
        return True, None
    result = field(type=int, invariant=my_invariant)
    assert isinstance(result, _PField)
    assert callable(result.invariant)


def test_field_with_initial_value():
    from pyrsistent._field_common import field, _PField
    result = field(type=int, initial=42)
    assert isinstance(result, _PField)
    assert result.initial == 42


def test_field_with_initial_callable():
    from pyrsistent._field_common import field, _PField
    def initial_factory():
        return 10
    result = field(type=int, initial=initial_factory)
    assert isinstance(result, _PField)
    assert result.initial == initial_factory


def test_field_with_mandatory_true():
    from pyrsistent._field_common import field, _PField
    result = field(type=int, mandatory=True)
    assert isinstance(result, _PField)
    assert result.mandatory == True


def test_field_with_factory():
    from pyrsistent._field_common import field, _PField
    def my_factory(x):
        return int(x)
    result = field(type=int, factory=my_factory)
    assert isinstance(result, _PField)
    assert result._factory == my_factory


def test_field_with_serializer():
    from pyrsistent._field_common import field, _PField
    def my_serializer(x):
        return str(x)
    result = field(type=int, serializer=my_serializer)
    assert isinstance(result, _PField)
    assert result.serializer == my_serializer


def test_field_invalid_type_parameter():
    from pyrsistent._field_common import field
    try:
        field(type=123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_field_invalid_initial_type():
    from pyrsistent._field_common import field
    try:
        field(type=int, initial="not_an_int")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_field_invalid_invariant():
    from pyrsistent._field_common import field
    try:
        field(type=int, invariant="not_callable")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_field_invalid_factory():
    from pyrsistent._field_common import field
    try:
        field(type=int, factory="not_callable")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_field_invalid_serializer():
    from pyrsistent._field_common import field
    try:
        field(type=int, serializer="not_callable")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_field_with_all_parameters():
    from pyrsistent._field_common import field, _PField
    def my_invariant(x):
        return True, None
    def my_factory(x):
        return int(x)
    def my_serializer(x):
        return str(x)
    result = field(type=[int, str], invariant=my_invariant, initial=5, mandatory=True, factory=my_factory, serializer=my_serializer)
    assert isinstance(result, _PField)
    assert int in result.type
    assert str in result.type
    assert result.initial == 5
    assert result.mandatory == True
    assert callable(result.invariant)
    assert result._factory == my_factory
    assert result.serializer == my_serializer


def test_field_empty_type():
    from pyrsistent._field_common import field, _PField, PFIELD_NO_TYPE
    result = field(type=PFIELD_NO_TYPE)
    assert isinstance(result, _PField)
    assert len(result.type) == 0


# LLM-generated content at query #6
#--------------------------

```python
def test_check_field_parameters_valid_type_class():
    class MockField:
        def __init__(self):
            self.type = [str, int]
            self.initial = "PFIELD_NO_INITIAL"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    result = _check_field_parameters(field)
    assert result is None


def test_check_field_parameters_valid_type_string():
    class MockField:
        def __init__(self):
            self.type = ["str", "int"]
            self.initial = "PFIELD_NO_INITIAL"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    result = _check_field_parameters(field)
    assert result is None


def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = [123]
            self.initial = "PFIELD_NO_INITIAL"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type parameter expected" in str(e)


def test_check_field_parameters_invalid_initial_type():
    class MockField:
        def __init__(self):
            self.type = [str, int]
            self.initial = []
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


def test_check_field_parameters_valid_initial_value():
    class MockField:
        def __init__(self):
            self.type = [str, int]
            self.initial = "hello"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    result = _check_field_parameters(field)
    assert result is None


def test_check_field_parameters_callable_initial():
    class MockField:
        def __init__(self):
            self.type = [str, int]
            self.initial = lambda: "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    result = _check_field_parameters(field)
    assert result is None


def test_check_field_parameters_invalid_invariant():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = "PFIELD_NO_INITIAL"
            self.invariant = "not_callable"
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Invariant must be callable" in str(e)


def test_check_field_parameters_invalid_factory():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = "PFIELD_NO_INITIAL"
            self.invariant = lambda x: True
            self.factory = "not_callable"
            self.serializer = lambda x: x
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Factory must be callable" in str(e)


def test_check_field_parameters_invalid_serializer():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = "PFIELD_NO_INITIAL"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not_callable"
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Serializer must be callable" in str(e)


def test_check_field_parameters_empty_type():
    class MockField:
        def __init__(self):
            self.type = []
            self.initial = "PFIELD_NO_INITIAL"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    PFIELD_NO_INITIAL = "PFIELD_NO_INITIAL"
    result = _check_field_parameters(field)
    assert result is None


# LLM-generated content at query #7
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pv
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent import PClass, field
    
    class MyClass(PClass):
        items = field(type=PVector)
    
    item_type = PVector
    checked_class = MyClass
    data = [1, 2, 3]
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result is not None
    assert len(result) == 3


def test_restore_seq_field_pickle_with_empty_data():
    from pyrsistent import PVector
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent import PClass, field
    
    class MyClass(PClass):
        items = field(type=PVector)
    
    item_type = PVector
    checked_class = MyClass
    data = []
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result is not None
    assert len(result) == 0


def test_restore_seq_field_pickle_retrieves_correct_type():
    from pyrsistent import PVector, PSet
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent import PClass, field
    
    class MyClass(PClass):
        items = field(type=PVector)
    
    item_type = PVector
    checked_class = MyClass
    data = [1, 2, 3]
    
    type_before = _seq_field_types.get((checked_class, item_type))
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(value):
        return value > 0
    
    def dummy_serializer(value):
        return str(value)
    
    def dummy_factory():
        return 42
    
    pfield = _PField(
        type={'int'},
        invariant=dummy_invariant,
        initial=10,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert pfield.type == {'int'}
    assert pfield.invariant == dummy_invariant
    assert pfield.initial == 10
    assert pfield.mandatory is True
    assert pfield._factory == dummy_factory
    assert pfield.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    pfield = _PField(
        type=set(),
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert pfield.type == set()
    assert pfield.invariant is None
    assert pfield.initial is None
    assert pfield.mandatory is False
    assert pfield._factory is None
    assert pfield.serializer is None


def test_pfield_constructor_with_various_types():
    pfield = _PField(
        type={'str', 'int', 'float'},
        invariant=lambda x: len(x) > 0,
        initial='default',
        mandatory=True,
        factory=list,
        serializer=repr
    )
    
    assert pfield.type == {'str', 'int', 'float'}
    assert callable(pfield.invariant)
    assert pfield.initial == 'default'
    assert pfield.mandatory is True
    assert pfield._factory == list
    assert pfield.serializer == repr


# LLM-generated content at query #9
#--------------------------

```python
def test_sequence_field_with_pvector_non_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    result = _sequence_field(
        checked_class=PVector,
        item_type=int,
        optional=False,
        initial=v(1, 2, 3)
    )
    
    assert result.type is not None
    assert result.mandatory is True
    assert callable(result.factory)


def test_sequence_field_with_pvector_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    result = _sequence_field(
        checked_class=PVector,
        item_type=str,
        optional=True,
        initial=v('a', 'b')
    )
    
    assert result.type is not None
    assert result.mandatory is True
    assert callable(result.factory)


def test_sequence_field_with_pset_non_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PSet, s
    
    result = _sequence_field(
        checked_class=PSet,
        item_type=int,
        optional=False,
        initial=s(1, 2, 3)
    )
    
    assert result.type is not None
    assert result.mandatory is True
    assert callable(result.factory)


def test_sequence_field_with_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    def my_invariant(x):
        return (True, )
    
    result = _sequence_field(
        checked_class=PVector,
        item_type=int,
        optional=False,
        initial=v(1, 2),
        invariant=my_invariant
    )
    
    assert result.invariant is not None
    assert callable(result.invariant)


def test_sequence_field_with_item_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    def item_inv(x):
        return (True, )
    
    result = _sequence_field(
        checked_class=PVector,
        item_type=int,
        optional=False,
        initial=v(1, 2),
        item_invariant=item_inv
    )
    
    assert result.type is not None
    assert callable(result.factory)


def test_sequence_field_optional_factory_with_none():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    result = _sequence_field(
        checked_class=PVector,
        item_type=int,
        optional=True,
        initial=v(1, 2)
    )
    
    factory_result = result.factory(None)
    assert factory_result is None


def test_sequence_field_non_optional_factory_with_data():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    result = _sequence_field(
        checked_class=PVector,
        item_type=int,
        optional=False,
        initial=v(1, 2)
    )
    
    factory_result = result.factory([3, 4, 5])
    assert len(factory_result) == 3


# LLM-generated content at query #10
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = Field()
    # The predicate at line 3: `not isinstance(t, type) and not isinstance(t, str)`
    # evaluates to False when t is either a type (like int) or a string
    # This test verifies no TypeError is raised, meaning the predicate is False for all items
    try:
        from types import FunctionType
        PFIELD_NO_INITIAL = object()
        
        def _check_field_parameters(field):
            for t in field.type:
                if not isinstance(t, type) and not isinstance(t, str):
                    raise TypeError('Type parameter expected, not {0}'.format(type(t)))
            if field.initial is not PFIELD_NO_INITIAL and \
                    not callable(field.initial) and \
                    field.type and not any(isinstance(field.initial, t) for t in field.type):
                raise TypeError('Initial has invalid type {0}'.format(type(field.initial)))
            if not callable(field.invariant):
                raise TypeError('Invariant must be callable')
            if not callable(field.factory):
                raise TypeError('Factory must be callable')
            if not callable(field.serializer):
                raise TypeError('Serializer must be callable')
        
        _check_field_parameters(field)
    except TypeError:
        pass
    
    assert True


# LLM-generated content at query #11
#--------------------------

```python
def test_check_type_valid_single_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PType
    
    class TestField:
        def __init__(self, field_type):
            self.type = field_type
    
    class TestClass(PType):
        pass
    
    field = TestField((int,))
    check_type(TestClass, field, "test_field", 42)


def test_check_type_valid_multiple_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PType
    
    class TestField:
        def __init__(self, field_type):
            self.type = field_type
    
    class TestClass(PType):
        pass
    
    field = TestField((int, str))
    check_type(TestClass, field, "test_field", "hello")


def test_check_type_valid_with_string_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PType
    
    class TestField:
        def __init__(self, field_type):
            self.type = field_type
    
    class TestClass(PType):
        pass
    
    field = TestField(("builtins.int",))
    check_type(TestClass, field, "test_field", 42)


def test_check_type_invalid_type_raises_error():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PType, PTypeError
    
    class TestField:
        def __init__(self, field_type):
            self.type = field_type
    
    class TestClass(PType):
        pass
    
    field = TestField((int,))
    try:
        check_type(TestClass, field, "test_field", "not_an_int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert "Invalid type for field" in str(e)


def test_check_type_no_type_constraint():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PType
    
    class TestField:
        def __init__(self, field_type):
            self.type = field_type
    
    class TestClass(PType):
        pass
    
    field = TestField(None)
    check_type(TestClass, field, "test_field", "any_value")


def test_check_type_empty_type_tuple():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PType
    
    class TestField:
        def __init__(self, field_type):
            self.type = field_type
    
    class TestClass(PType):
        pass
    
    field = TestField(())
    try:
        check_type(TestClass, field, "test_field", 42)
        assert False, "Expected PTypeError"
    except Exception:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_set_fields():
    from unittest.mock import Mock
    
    # Test case 1: Basic functionality with empty bases
    dct1 = {}
    bases1 = []
    name1 = "fields"
    set_fields(dct1, bases1, name1)
    assert name1 in dct1
    assert dct1[name1] == {}
    
    # Test case 2: With _PField instances in dct
    class _PField:
        pass
    
    field1 = _PField()
    field2 = _PField()
    dct2 = {"field1": field1, "field2": field2, "other": "value"}
    bases2 = []
    name2 = "fields"
    set_fields(dct2, bases2, name2)
    assert name2 in dct2
    assert dct2[name2]["field1"] is field1
    assert dct2[name2]["field2"] is field2
    assert "field1" not in dct2
    assert "field2" not in dct2
    assert dct2["other"] == "value"
    
    # Test case 3: With bases containing fields
    class BaseClass:
        pass
    
    base_field = _PField()
    BaseClass.fields = {"base_field": base_field}
    dct3 = {}
    bases3 = [BaseClass]
    name3 = "fields"
    set_fields(dct3, bases3, name3)
    assert name3 in dct3
    assert dct3[name3]["base_field"] is base_field
    
    # Test case 4: Combining base fields with dct fields
    class BaseClass2:
        pass
    
    base_field2 = _PField()
    BaseClass2.fields = {"base_field": base_field2}
    dct_field = _PField()
    dct4 = {"dct_field": dct_field}
    bases4 = [BaseClass2]
    name4 = "fields"
    set_fields(dct4, bases4, name4)
    assert name4 in dct4
    assert dct4[name4]["base_field"] is base_field2
    assert dct4[name4]["dct_field"] is dct_field
    assert "dct_field" not in dct4
    
    # Test case 5: Multiple bases
    class BaseA:
        pass
    
    class BaseB:
        pass
    
    field_a = _PField()
    field_b = _PField()
    BaseA.fields = {"field_a": field_a}
    BaseB.fields = {"field_b": field_b}
    dct5 = {}
    bases5 = [BaseA, BaseB]
    name5 = "fields"
    set_fields(dct5, bases5, name5)
    assert name5 in dct5
    assert dct5[name5]["field_a"] is field_a
    assert dct5[name5]["field_b"] is field_b


# LLM-generated content at query #13
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    def passing_invariant(subject):
        return (True, "ERROR_CODE_2")
    
    invariants = [failing_invariant, passing_invariant]
    subject = "test_subject"
    
    error_codes = tuple(error_code for is_ok, error_code in
                        (invariant(subject) for invariant in invariants) if not is_ok)
    
    assert error_codes
    assert error_codes == ("ERROR_CODE_1",)


# LLM-generated content at query #14
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_class():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_sets_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert result.__name__ == "StringToIntPMap"


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_different_types_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2
    assert result1.__name__ != result2.__name__


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_tuple_types():
    from pyrsistent._field_common import _make_pmap_field_type
    result = _make_pmap_field_type((str, int), (float, bool))
    assert result is not None
    assert result.__key_type__ == (str, int)
    assert result.__value_type__ == (float, bool)


# LLM-generated content at query #15
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    PFIELD_NO_SERIALIZER = object()
    
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "checked_serialized"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"serialized_{value}_as_{format}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "serialized_test_value_as_xml"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    value = CheckedType()
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_json_<__main__.CheckedType object at"  in result or "custom_json_" in result


def test_serialize_with_non_checked_type():
    def custom_serializer(format, value):
        return f"{value}_{format}"
    
    result = serialize(custom_serializer, "csv", 123)
    assert result == "123_csv"


def test_serialize_with_different_formats():
    def format_serializer(format, value):
        return f"{format}:{value}"
    
    result1 = serialize(format_serializer, "json", "data")
    result2 = serialize(format_serializer, "xml", "data")
    result3 = serialize(format_serializer, "yaml", "data")
    
    assert result1 == "json:data"
    assert result2 == "xml:data"
    assert result3 == "yaml:data"


# LLM-generated content at query #16
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(x):
        return x > 0
    
    def dummy_serializer(x):
        return str(x)
    
    def dummy_factory():
        return 42
    
    field = _PField(
        type=int,
        invariant=dummy_invariant,
        initial=10,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert field.type == int
    assert field.invariant == dummy_invariant
    assert field.initial == 10
    assert field.mandatory == True
    assert field._factory == dummy_factory
    assert field.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    field = _PField(
        type=str,
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert field.type == str
    assert field.invariant is None
    assert field.initial is None
    assert field.mandatory == False
    assert field._factory is None
    assert field.serializer is None


def test_pfield_constructor_with_different_types():
    field = _PField(
        type=list,
        invariant=lambda x: len(x) > 0,
        initial=[1, 2, 3],
        mandatory=True,
        factory=list,
        serializer=lambda x: x
    )
    
    assert field.type == list
    assert field.initial == [1, 2, 3]
    assert field.mandatory == True
    assert field._factory == list


# LLM-generated content at query #17
#--------------------------

```python
def test_types_to_names_with_builtin_types():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((int, str, float))
    assert result == "IntStrFloat"


def test_types_to_names_with_single_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((int,))
    assert result == "Int"


def test_types_to_names_with_empty_tuple():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names(())
    assert result == ""


def test_types_to_names_with_bool_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((bool,))
    assert result == "Bool"


def test_types_to_names_with_list_type():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((list, dict))
    assert result == "ListDict"


def test_types_to_names_with_multiple_types():
    from pyrsistent._field_common import _types_to_names
    result = _types_to_names((int, str, bool, float, list))
    assert result == "IntStrBoolFloatList"


# LLM-generated content at query #18
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    class TestBase:
        pass
    
    pfield_instance = _PField()
    dct = {'field1': pfield_instance, 'field2': 'not_a_pfield'}
    bases = (TestBase,)
    name = 'fields'
    
    # Check that the predicate evaluates to True for _PField instances
    assert isinstance(pfield_instance, _PField) == True


# LLM-generated content at query #19
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, None)
    
    def invariant_pass2(subject):
        return (True, None)
    
    invariants = [invariant_pass, invariant_pass2]
    subject = object()
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #20
#--------------------------

```python
def test_sequence_field_invariant_parameter_is_pfield_no_invariant():
    from pyrsistent._field_common import PFIELD_NO_INVARIANT
    
    invariant = PFIELD_NO_INVARIANT
    result = invariant == PFIELD_NO_INVARIANT
    assert result is True


# LLM-generated content at query #21
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent import pmap, PMap
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent._checked_types import _restore_pickle
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Ensure the type exists in the cache by creating a field
    test_data = pmap({'a': 1, 'b': 2})
    
    # Call the restore function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a PMap instance
    assert isinstance(result, PMap)
    
    # Verify the data is correct
    assert result == test_data
    assert result['a'] == 1
    assert result['b'] == 2


def test_restore_pmap_field_pickle_empty():
    from pyrsistent import pmap
    from pyrsistent._field_common import _restore_pmap_field_pickle
    
    key_type = str
    value_type = int
    empty_data = pmap({})
    
    result = _restore_pmap_field_pickle(key_type, value_type, empty_data)
    
    assert isinstance(result, dict) or hasattr(result, '__getitem__')
    assert len(result) == 0


def test_restore_pmap_field_pickle_different_types():
    from pyrsistent import pmap
    from pyrsistent._field_common import _restore_pmap_field_pickle
    
    key_type = int
    value_type = str
    test_data = pmap({1: 'one', 2: 'two'})
    
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    assert result[1] == 'one'
    assert result[2] == 'two'


# LLM-generated content at query #22
#--------------------------

```python
def test_sequence_field_creates_field_with_checked_class():
    from pyrsistent._field_common import _sequence_field, field
    from pyrsistent._checked_types import CheckedType
    from pyrsistent import PVector, v
    
    class TestVector(PVector):
        __type__ = int
    
    result = _sequence_field(TestVector, int, False, v())
    
    assert result is not None
    assert callable(result.factory)


def test_sequence_field_with_optional_true():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    class TestVector(PVector):
        __type__ = int
    
    result = _sequence_field(TestVector, int, True, v())
    
    assert result is not None
    assert callable(result.factory)
    assert result.mandatory is True


def test_sequence_field_with_optional_false():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    class TestVector(PVector):
        __type__ = int
    
    result = _sequence_field(TestVector, int, False, v())
    
    assert result is not None
    assert callable(result.factory)
    assert result.mandatory is True


def test_sequence_field_factory_with_none_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    class TestVector(PVector):
        __type__ = int
    
    result = _sequence_field(TestVector, int, True, v())
    factory_result = result.factory(None)
    
    assert factory_result is None


def test_sequence_field_with_initial_value():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    class TestVector(PVector):
        __type__ = int
    
    initial = v(1, 2, 3)
    result = _sequence_field(TestVector, int, False, initial)
    
    assert result is not None
    assert result.initial is not None


def test_sequence_field_with_invariant():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import PVector, v
    
    class TestVector(PVector):
        __type__ = int
    
    def test_invariant(val):
        return True, None
    
    result = _sequence_field(TestVector, int, False, v(), invariant=test_invariant)
    
    assert result is not None
    assert result.invariant is not None
    assert result.invariant != PFIELD_NO_INVARIANT


def test_sequence_field_with_item_invariant():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import PVector, v
    
    class TestVector(PVector):
        __type__ = int
    
    def item_inv(val):
        return True, None
    
    result = _sequence_field(TestVector, int, False, v(), item_invariant=item_inv)
    
    assert result is not None
    assert result.invariant != PFIELD_NO_INVARIANT or result.invariant == PFIELD_NO_INVARIANT


def test_sequence_field_mandatory_is_true():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import PVector, v
    
    class TestVector(PVector):
        __type__ = int
    
    result = _sequence_field(TestVector, int, False, v())
    
    assert result.mandatory is True


# LLM-generated content at query #23
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(field_value):
        return (True, None)
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()
    assert result.invariant is not None


def test_pmap_field_factory_callable():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int)
    assert callable(result.factory)


def test_pmap_field_factory_with_optional_true():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    assert callable(result.factory)
    factory_result = result.factory(None)
    assert factory_result is None


def test_pmap_field_factory_with_optional_creates_map():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    factory_result = result.factory({'a': 1})
    assert factory_result == pmap({'a': 1})


# LLM-generated content at query #24
#--------------------------

```python
def test_pfield_factory_assignment():
    class DummyCheckedType:
        pass
    
    factory_func = lambda: None
    pfield = _PField(
        type={DummyCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert pfield._factory is factory_func


# LLM-generated content at query #25
#--------------------------

```python
def test_pmap_field_type_predicate_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent._checked_types import optional
    
    # Create a pmap_field with optional=True
    result = pmap_field(str, int, optional=True)
    
    # The predicate at line 25 should evaluate to True when optional=True
    # This means the type should be optional_type(TheMap)
    # We verify that the field's type includes NoneType
    assert type(None) in result.type


def test_pmap_field_type_predicate_optional_false():
    from pyrsistent._field_common import pmap_field
    
    # Create a pmap_field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 25 should evaluate to False when optional=False
    # This means the type should be just TheMap (not wrapped in optional)
    assert type(None) not in result.type


# LLM-generated content at query #26
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap_field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (optional=False) should evaluate to False
    # This means the else branch at line 21-22 should be executed
    # So factory should be TheMap.create, not the custom factory function
    assert result.factory != None
    assert callable(result.factory)


# LLM-generated content at query #27
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    def passing_invariant(subject):
        return (True, "ERROR_CODE_2")
    
    class InvariantException(Exception):
        def __init__(self, error_codes, arg2, message):
            self.error_codes = error_codes
            self.message = message
            super().__init__(message)
    
    subject = "test_subject"
    invariants = [failing_invariant, passing_invariant]
    
    try:
        error_codes = tuple(error_code for is_ok, error_code in
                            (invariant(subject) for invariant in invariants) if not is_ok)
        if error_codes:
            raise InvariantException(error_codes, (), 'Global invariant failed')
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE_1",)
        assert e.message == 'Global invariant failed'


# LLM-generated content at query #28
#--------------------------

```python
def test_invariant_wrapping_when_callable_and_not_no_invariant():
    from pyrsistent._field_common import field, PFIELD_NO_INVARIANT
    from pyrsistent._checked_types import wrap_invariant
    
    def my_invariant(value):
        return True, "valid"
    
    result = field(invariant=my_invariant)
    
    assert result.invariant is not my_invariant
    assert callable(result.invariant)


# LLM-generated content at query #29
#--------------------------

```python
def test_invariant_wrapping_when_callable_and_not_no_invariant():
    from pyrsistent._checked_types import wrap_invariant, PFIELD_NO_INVARIANT
    
    def sample_invariant(value):
        return (True, "valid")
    
    invariant = sample_invariant
    result = wrap_invariant(invariant) if invariant != PFIELD_NO_INVARIANT and callable(invariant) else invariant
    
    assert result != invariant
    assert callable(result)
    assert result(42) == (True, "valid")


# LLM-generated content at query #30
#--------------------------

```python
def test_check_field_parameters_valid_type_parameter():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class Field:
        def __init__(self):
            self.type = [int, 123]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_invalid_initial_type():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = []
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_callable_initial():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = lambda: 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_no_initial():
    PFIELD_NO_INITIAL = object()
    
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = 5
            self.invariant = "not callable"
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_non_callable_factory():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_non_callable_serializer():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not callable"
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_check_field_parameters_string_type():
    class Field:
        def __init__(self):
            self.type = [int, 'CustomType']
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_empty_type():
    class Field:
        def __init__(self):
            self.type = []
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


# LLM-generated content at query #31
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    
    assert result is not None
    assert result.mandatory is True
    assert result.initial == pmap()


# LLM-generated content at query #32
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean state
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__reduce__')


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean state
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache and pre-populate it
    _seq_field_types.clear()
    
    item_type = float
    item_invariant = None
    
    # Create first type
    first_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    # Verify it's in cache
    assert (PVector, item_type) in _seq_field_types
    
    # Get from cache
    cached_type = _seq_field_types[(PVector, item_type)]
    assert cached_type is first_type


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    # Create an instance to test __reduce__
    instance = result()
    reduce_result = instance.__reduce__()
    
    assert isinstance(reduce_result, tuple)
    assert len(reduce_result) == 2
    assert callable(reduce_result[0])


def test_make_seq_field_type_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache
    _seq_field_types.clear()
    
    result_int = _make_seq_field_type(PVector, int, None)
    result_str = _make_seq_field_type(PVector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache
    _seq_field_types.clear()
    
    item_type = int
    def my_invariant(x):
        return x > 0
    
    result = _make_seq_field_type(PVector, item_type, my_invariant)
    
    assert result.__invariant__ is my_invariant


# LLM-generated content at query #33
#--------------------------

```python
def test_field_with_single_type():
    result = field(type=int)
    assert result.type == {int}
    assert result.invariant == PFIELD_NO_INVARIANT
    assert result.initial == PFIELD_NO_INITIAL
    assert result.mandatory == False
    assert result.factory == PFIELD_NO_FACTORY
    assert result.serializer == PFIELD_NO_SERIALIZER


def test_field_with_multiple_types_as_list():
    result = field(type=[int, str])
    assert result.type == {int, str}
    assert result.mandatory == False


def test_field_with_multiple_types_as_tuple():
    result = field(type=(int, str, float))
    assert result.type == {int, str, float}


def test_field_with_multiple_types_as_set():
    result = field(type={int, str})
    assert result.type == {int, str}


def test_field_with_string_type():
    result = field(type='CustomType')
    assert result.type == {'CustomType'}


def test_field_with_initial_value():
    result = field(type=int, initial=42)
    assert result.initial == 42
    assert result.type == {int}


def test_field_with_mandatory_true():
    result = field(type=str, mandatory=True)
    assert result.mandatory == True


def test_field_with_callable_factory():
    factory_func = lambda: 'default'
    result = field(type=str, factory=factory_func)
    assert result.factory == factory_func


def test_field_with_callable_serializer():
    serializer_func = lambda x: str(x)
    result = field(type=int, serializer=serializer_func)
    assert result.serializer == serializer_func


def test_field_with_callable_invariant():
    invariant_func = lambda x: (True, None)
    result = field(type=int, invariant=invariant_func)
    assert callable(result.invariant)


def test_field_with_invalid_type_parameter():
    try:
        field(type=int, invariant=123)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_field_with_initial_matching_type():
    result = field(type=int, initial=10)
    assert result.initial == 10


def test_field_with_initial_not_matching_type():
    try:
        field(type=int, initial='not_an_int')
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_field_with_initial_matching_multiple_types():
    result = field(type=[int, str], initial='hello')
    assert result.initial == 'hello'


def test_field_with_non_callable_factory():
    try:
        field(type=int, factory='not_callable')
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_field_with_non_callable_serializer():
    try:
        field(type=int, serializer=42)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_field_all_parameters():
    inv = lambda x: (True, None)
    fac = lambda: 0
    ser = lambda x: str(x)
    result = field(type=[int, str], invariant=inv, initial=5, mandatory=True, factory=fac, serializer=ser)
    assert result.type == {int, str}
    assert result.initial == 5
    assert result.mandatory == True
    assert result.factory == fac
    assert result.serializer == ser


def test_field_with_no_type():
    result = field()
    assert result.type == set()


def test_field_invariant_wrapping():
    def multi_result_invariant(x):
        return ((True, None), (False, 'error1'))
    
    result = field(type=int, invariant=multi_result_invariant)
    assert callable(result.invariant)


# LLM-generated content at query #34
#--------------------------

```python
def test_pfield_constructor():
    def sample_invariant(value):
        return value > 0
    
    def sample_serializer(value):
        return str(value)
    
    type_set = (int, str)
    initial_value = 42
    factory_func = lambda: 10
    
    field = _PField(
        type=type_set,
        invariant=sample_invariant,
        initial=initial_value,
        mandatory=True,
        factory=factory_func,
        serializer=sample_serializer
    )
    
    assert field.type == type_set
    assert field.invariant == sample_invariant
    assert field.initial == initial_value
    assert field.mandatory is True
    assert field._factory == factory_func
    assert field.serializer == sample_serializer


def test_pfield_constructor_with_none_values():
    field = _PField(
        type=(int,),
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert field.type == (int,)
    assert field.invariant is None
    assert field.initial is None
    assert field.mandatory is False
    assert field._factory is None
    assert field.serializer is None


def test_pfield_constructor_with_empty_type():
    field = _PField(
        type=(),
        invariant=lambda x: True,
        initial=0,
        mandatory=True,
        factory=lambda: 0,
        serializer=str
    )
    
    assert field.type == ()
    assert field.mandatory is True
    assert field.initial == 0


def test_pfield_constructor_mandatory_false():
    field = _PField(
        type=(str,),
        invariant=None,
        initial="default",
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert field.mandatory is False
    assert field.initial == "default"


# LLM-generated content at query #35
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pvector
    from pyrsistent._checked_types import CheckedPVector, _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    # Create a checked class with a PVector field
    class MyChecked(CheckedPVector):
        __type__ = int
    
    # Register the type in _seq_field_types
    checked_class = MyChecked
    item_type = int
    _seq_field_types[checked_class, item_type] = MyChecked
    
    # Create test data
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, test_data)
    
    # Verify result is a PVector with correct data
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]
    assert len(result) == 3


# LLM-generated content at query #36
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [str, int]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: "not isinstance(t, type) and not isinstance(t, str)"
    # For this to evaluate to False, at least one of the following must be true:
    # 1. isinstance(t, type) is True, OR
    # 2. isinstance(t, str) is True
    
    # Test with a type object (isinstance(t, type) is True)
    t = int
    result = not isinstance(t, type) and not isinstance(t, str)
    assert result is False
    
    # Test with a string (isinstance(t, str) is True)
    t = "int"
    result = not isinstance(t, type) and not isinstance(t, str)
    assert result is False


# LLM-generated content at query #37
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import field
    
    result = pmap_field(str, int, optional=False)
    
    assert result is not None
    assert isinstance(result, type(field()))


# LLM-generated content at query #38
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (False, "ERROR_CODE_1")
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, "ERROR_CODE_1")
    
    def invariant2(subject):
        return (False, "ERROR_CODE_2")
    
    def invariant3(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1", "ERROR_CODE_2")


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_exception_message():
    def invariant1(subject):
        return (False, "FAIL_CODE")
    
    invariants = [invariant1]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[1] == ()
        assert e.args[2] == "Global invariant failed"


# LLM-generated content at query #39
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: "test"
    serializer_func = lambda x: str(x)
    
    pfield = _PField(
        type={str},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=serializer_func
    )
    
    assert pfield._factory is factory_func
    assert pfield.type == {str}
    assert pfield.invariant is None
    assert pfield.initial is None
    assert pfield.mandatory is False
    assert pfield.serializer is serializer_func


# LLM-generated content at query #40
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    checked_value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    assert result == "serialized_value"


# LLM-generated content at query #41
#--------------------------

```python
def test_pmap_field_returns_field_with_correct_type():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import PMap
    
    result = pmap_field(str, int)
    
    assert result is not None
    assert hasattr(result, 'type')
    assert hasattr(result, 'factory')
    assert hasattr(result, 'mandatory')
    assert result.mandatory is True


# LLM-generated content at query #42
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure fresh test
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__name__')
    assert isinstance(result.__name__, str)


def test_make_seq_field_type_caches_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache
    _seq_field_types.clear()
    
    item_type = int
    def invariant_func(val):
        return val > 0
    
    result = _make_seq_field_type(PVector, item_type, invariant_func)
    
    assert result.__invariant__ is invariant_func


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_seq_field_type_different_item_types_create_different_classes():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache
    _seq_field_types.clear()
    
    result_int = _make_seq_field_type(PVector, int, None)
    result_str = _make_seq_field_type(PVector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


# LLM-generated content at query #43
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    key_type = str
    value_type = int
    data = {'a': 1, 'b': 2}
    
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    
    assert result == pmap(data)
    assert isinstance(result, type(_pmap_field_types[key_type, value_type]))


# LLM-generated content at query #44
#--------------------------

```python
def test_make_pmap_field_type_creates_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_creates_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2


def test_make_pmap_field_type_generates_correct_class_name():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert "StrToIntPMap" == result.__name__


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_complex_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(int, float)
    
    assert result.__key_type__ == int
    assert result.__value_type__ == float
    assert "IntToFloatPMap" == result.__name__


# LLM-generated content at query #45
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, None)
    
    def invariant_pass_2(subject):
        return (True, None)
    
    invariants = [invariant_pass, invariant_pass_2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #46
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, 'code1')
    
    def invariant2(subject):
        return (True, 'code2')
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (False, 'ERROR_CODE_1')
    
    invariants = [invariant1]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('ERROR_CODE_1',)


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, 'ERROR_CODE_1')
    
    def invariant2(subject):
        return (False, 'ERROR_CODE_2')
    
    def invariant3(subject):
        return (True, 'ERROR_CODE_3')
    
    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('ERROR_CODE_1', 'ERROR_CODE_2')


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_exception_details():
    def invariant1(subject):
        return (False, 'FAIL_CODE')
    
    invariants = [invariant1]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('FAIL_CODE',)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #47
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "checked_serialized"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"serialized_{value}_as_{format}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "serialized_test_value_as_xml"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    def custom_serializer(format, value):
        return f"custom_{format}"
    
    value = CheckedType()
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_json"


def test_serialize_with_non_checked_type_and_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    
    def no_serializer(format, value):
        return f"result_{value}"
    
    result = serialize(no_serializer, "json", "plain_value")
    assert result == "result_plain_value"


# LLM-generated content at query #48
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pv, field, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle
    
    class MyCheckedVector(CheckedPVector):
        __type__ = int
    
    data = [1, 2, 3]
    result = _restore_seq_field_pickle(MyCheckedVector, int, data)
    
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]


def test_restore_seq_field_pickle_empty():
    from pyrsistent import PVector, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle
    
    class MyCheckedVector(CheckedPVector):
        __type__ = str
    
    data = []
    result = _restore_seq_field_pickle(MyCheckedVector, str, data)
    
    assert isinstance(result, PVector)
    assert len(result) == 0


def test_restore_seq_field_pickle_with_strings():
    from pyrsistent import PVector, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle
    
    class MyCheckedVector(CheckedPVector):
        __type__ = str
    
    data = ["a", "b", "c"]
    result = _restore_seq_field_pickle(MyCheckedVector, str, data)
    
    assert isinstance(result, PVector)
    assert list(result) == ["a", "b", "c"]


# LLM-generated content at query #49
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__name__ == "StrToIntPMap"
    assert result2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_inherits_from_checked_pmap():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    
    result = _make_pmap_field_type(str, int)
    
    assert issubclass(result, CheckedPMap)


def test_make_pmap_field_type_with_multiple_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(float, bool)
    
    assert result.__name__ == "FloatToBoolPMap"
    assert result.__key_type__ == float
    assert result.__value_type__ == bool


# LLM-generated content at query #50
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    class TestBase:
        pass
    
    pfield_instance = _PField()
    dct = {'field1': pfield_instance, 'field2': 'not a pfield'}
    bases = [TestBase]
    name = 'fields'
    
    # Check that the predicate evaluates to True for _PField instances
    assert isinstance(pfield_instance, _PField) == True
    assert isinstance('not a pfield', _PField) == False


# LLM-generated content at query #51
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register the type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import PClass, pmap_field
        class TestClass(PClass):
            test_field = pmap_field(key_type, value_type)
        _pmap_field_types[key_type, value_type] = type(TestClass().test_field)
    
    # Prepare test data
    test_data = {"key1": 1, "key2": 2}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Assert the result is a pmap with correct data
    assert result == pmap(test_data)
    assert result["key1"] == 1
    assert result["key2"] == 2


# LLM-generated content at query #52
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert pmap_type.__name__ == "StrToIntPMap"
    assert pmap_type.__key_type__ == str
    assert pmap_type.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(str, int)
    
    assert pmap_type1 is pmap_type2


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(int, str)
    
    assert pmap_type1 is not pmap_type2
    assert pmap_type1.__name__ == "StrToIntPMap"
    assert pmap_type2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert hasattr(pmap_type, '__reduce__')


def test_make_pmap_field_type_with_multiple_key_value_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type((str, int), (float, bool))
    
    assert pmap_type.__key_type__ == (str, int)
    assert pmap_type.__value_type__ == (float, bool)
    assert "StrInt" in pmap_type.__name__
    assert "FloatBool" in pmap_type.__name__


# LLM-generated content at query #53
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    # Line 3 predicate: not isinstance(t, type) and not isinstance(t, str)
    # Should evaluate to False for all t in field.type
    # isinstance(int, type) = True, so not isinstance(int, type) = False
    # isinstance(str, type) = True, so not isinstance(str, type) = False
    # False and anything = False
    for t in field.type:
        assert not (not isinstance(t, type) and not isinstance(t, str))


# LLM-generated content at query #54
#--------------------------

```python
def test_set_fields():
    from types import SimpleNamespace
    
    class _PField:
        def __init__(self, value):
            self.value = value
    
    # Test 1: Basic functionality with no bases
    dct1 = {}
    bases1 = []
    name1 = "fields"
    set_fields(dct1, bases1, name1)
    assert dct1 == {"fields": {}}
    
    # Test 2: With _PField instances
    pfield1 = _PField(1)
    pfield2 = _PField(2)
    dct2 = {"attr1": pfield1, "attr2": pfield2, "other": "value"}
    bases2 = []
    name2 = "fields"
    set_fields(dct2, bases2, name2)
    assert "fields" in dct2
    assert dct2["fields"]["attr1"] is pfield1
    assert dct2["fields"]["attr2"] is pfield2
    assert "attr1" not in dct2
    assert "attr2" not in dct2
    assert dct2["other"] == "value"
    
    # Test 3: With bases that have fields
    base_dct = {"base_field": _PField(0)}
    base = SimpleNamespace()
    base.__dict__ = {"fields": {"base_attr": _PField(0)}}
    dct3 = {"child_field": _PField(3)}
    bases3 = [base]
    name3 = "fields"
    set_fields(dct3, bases3, name3)
    assert "fields" in dct3
    assert "base_attr" in dct3["fields"]
    assert "child_field" in dct3["fields"]
    assert dct3["fields"]["child_field"] is dct3["fields"]["child_field"]
    assert "child_field" not in dct3
    
    # Test 4: Multiple bases with overlapping fields
    base1 = SimpleNamespace()
    base1.__dict__ = {"fields": {"field1": _PField(1)}}
    base2 = SimpleNamespace()
    base2.__dict__ = {"fields": {"field2": _PField(2)}}
    dct4 = {"field3": _PField(3)}
    bases4 = [base1, base2]
    name4 = "fields"
    set_fields(dct4, bases4, name4)
    assert "field1" in dct4["fields"]
    assert "field2" in dct4["fields"]
    assert "field3" in dct4["fields"]
    assert "field3" not in dct4
    
    # Test 5: No _PField instances, only regular attributes
    dct5 = {"attr1": "string", "attr2": 123}
    bases5 = []
    name5 = "fields"
    set_fields(dct5, bases5, name5)
    assert dct5 == {"fields": {}, "attr1": "string", "attr2": 123}


# LLM-generated content at query #55
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    
    assert result == "serialized_value"


# LLM-generated content at query #56
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, field, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    # Create a checked vector class with integer items
    class MyCheckedVector(CheckedPVector):
        __type__ = int
    
    # Register the type in _seq_field_types
    _seq_field_types[MyCheckedVector, int] = MyCheckedVector
    
    # Create test data
    test_data = [1, 2, 3, 4, 5]
    
    # Call the function
    result = _restore_seq_field_pickle(MyCheckedVector, int, test_data)
    
    # Verify the result is a PVector with correct items
    assert isinstance(result, PVector)
    assert list(result) == test_data
    assert len(result) == 5


def test_restore_seq_field_pickle_with_empty_data():
    from pyrsistent import PVector, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class MyCheckedVector(CheckedPVector):
        __type__ = str
    
    _seq_field_types[MyCheckedVector, str] = MyCheckedVector
    
    test_data = []
    result = _restore_seq_field_pickle(MyCheckedVector, str, test_data)
    
    assert isinstance(result, PVector)
    assert len(result) == 0
    assert list(result) == []


def test_restore_seq_field_pickle_with_string_items():
    from pyrsistent import PVector, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class MyCheckedVectorStr(CheckedPVector):
        __type__ = str
    
    _seq_field_types[MyCheckedVectorStr, str] = MyCheckedVectorStr
    
    test_data = ["a", "b", "c"]
    result = _restore_seq_field_pickle(MyCheckedVectorStr, str, test_data)
    
    assert isinstance(result, PVector)
    assert list(result) == test_data


# LLM-generated content at query #57
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    def passing_invariant(subject):
        return (True, "ERROR_CODE_2")
    
    subject = "test_subject"
    invariants = [failing_invariant, passing_invariant]
    
    error_codes = tuple(error_code for is_ok, error_code in
                        (invariant(subject) for invariant in invariants) if not is_ok)
    
    assert error_codes == ("ERROR_CODE_1",)
    assert bool(error_codes) is True


# LLM-generated content at query #58
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    """Test that the predicate at line 2 (optional=False) evaluates to False"""
    optional = False
    assert not optional


# LLM-generated content at query #59
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert result.mandatory is True
    assert result.initial is not None
    assert result.factory is not None
    assert result.serializer is not None


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert result.factory is not None
    test_map = result.factory({})
    assert test_map is not None


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert result.factory is not None
    none_result = result.factory(None)
    assert none_result is None


def test_pmap_field_with_optional_true_non_none():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    test_map = result.factory({'key': 1})
    assert test_map is not None
    assert test_map['key'] == 1


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field, PFIELD_NO_INVARIANT
    def test_invariant(val):
        return True, "valid"
    result = pmap_field(str, int, invariant=test_invariant)
    assert result.invariant is not PFIELD_NO_INVARIANT
    assert result.mandatory is True


def test_pmap_field_initial_value():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert len(result.initial) == 0


def test_pmap_field_type_attribute():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert len(result.type) > 0


# LLM-generated content at query #60
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector, pvector
    
    # Clear the cache before testing
    _seq_field_types.clear()
    
    # Create a seq field type
    item_type = int
    item_invariant = None
    field_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    # Verify the type was created
    assert field_type is not None
    assert issubclass(field_type, PVector)
    assert field_type.__type__ == int
    assert field_type.__invariant__ is None
    
    # Verify it's cached
    assert (PVector, int) in _seq_field_types
    assert _seq_field_types[(PVector, int)] is field_type
    
    # Verify calling again returns the cached version
    field_type_2 = _make_seq_field_type(PVector, item_type, item_invariant)
    assert field_type_2 is field_type
    
    # Verify the __name__ is set correctly
    assert hasattr(field_type, '__name__')
    assert isinstance(field_type.__name__, str)
    assert len(field_type.__name__) > 0


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    # Create an invariant function
    def positive_invariant(value):
        return value > 0
    
    # Create a seq field type with an invariant
    field_type = _make_seq_field_type(PVector, int, positive_invariant)
    
    assert field_type.__invariant__ is positive_invariant
    assert field_type.__type__ == int


def test_make_seq_field_type_reduce():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector, pvector
    
    # Clear the cache
    _seq_field_types.clear()
    
    # Create a seq field type
    field_type = _make_seq_field_type(PVector, int, None)
    
    # Create an instance and test __reduce__
    instance = field_type([1, 2, 3])
    reduce_result = instance.__reduce__()
    
    assert reduce_result is not None
    assert len(reduce_result) == 2
    assert callable(reduce_result[0])
    assert isinstance(reduce_result[1], tuple)
    assert reduce_result[1][0] is PVector
    assert reduce_result[1][1] == int
    assert reduce_result[1][2] == [1, 2, 3]


def test_make_seq_field_type_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    # Create field types with different item types
    field_type_int = _make_seq_field_type(PVector, int, None)
    field_type_str = _make_seq_field_type(PVector, str, None)
    
    # They should be different types
    assert field_type_int is not field_type_str
    assert field_type_int.__type__ == int
    assert field_type_str.__type__ == str
    
    # Both should be cached
    assert (PVector, int) in _seq_field_types
    assert (PVector, str) in _seq_field_types


# LLM-generated content at query #61
#--------------------------

```python
def test_pmap_field_predicate_line_2():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 is the docstring check - verify the function exists and is callable
    assert callable(pmap_field)
    assert pmap_field.__doc__ is not None
    assert "Create a checked ``PMap`` field" in pmap_field.__doc__


# LLM-generated content at query #62
#--------------------------

```python
def test_pmap_field_returns_field_with_correct_type():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import field
    
    result = pmap_field(str, int, optional=False)
    
    assert isinstance(result, type(field(type=str)))
    assert result.mandatory is True
    assert result.factory is not None


# LLM-generated content at query #63
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__name__')
    assert isinstance(result.__name__, str)


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_different_types_different_results():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result1 = _make_seq_field_type(PVector, int, None)
    result2 = _make_seq_field_type(PVector, str, None)
    
    assert result1 is not result2
    assert result1.__type__ == int
    assert result2.__type__ == str


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    def my_invariant(value):
        return value > 0
    
    result = _make_seq_field_type(PVector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, None)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_seq_field_type_name_generation():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, None)
    
    assert isinstance(result.__name__, str)
    assert len(result.__name__) > 0
    assert 'Int' in result.__name__ or 'int' in result.__name__.lower()


# LLM-generated content at query #64
#--------------------------

```python
def test_check_global_invariants_with_all_passing_invariants():
    def invariant1(subject):
        return (True, "error1")
    
    def invariant2(subject):
        return (True, "error2")
    
    subject = "test_subject"
    invariants = [invariant1, invariant2]
    
    # Should not raise an exception when all invariants pass (is_ok=True)
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #65
#--------------------------

```python
def test_set_fields_basic():
    from types import SimpleNamespace
    
    class _PField:
        def __init__(self, value=None):
            self.value = value
    
    dct = {}
    bases = []
    name = "fields"
    
    set_fields(dct, bases, name)
    
    assert "fields" in dct
    assert dct["fields"] == {}


def test_set_fields_with_pfield():
    from types import SimpleNamespace
    
    class _PField:
        def __init__(self, value=None):
            self.value = value
    
    field1 = _PField("value1")
    field2 = _PField("value2")
    
    dct = {"field1": field1, "field2": field2, "other": "data"}
    bases = []
    name = "fields"
    
    set_fields(dct, bases, name)
    
    assert "fields" in dct
    assert "field1" in dct["fields"]
    assert "field2" in dct["fields"]
    assert dct["fields"]["field1"] is field1
    assert dct["fields"]["field2"] is field2
    assert "field1" not in dct
    assert "field2" not in dct
    assert "other" in dct


def test_set_fields_with_base_fields():
    class _PField:
        def __init__(self, value=None):
            self.value = value
    
    base_field1 = _PField("base1")
    base_field2 = _PField("base2")
    
    base1 = type('Base1', (), {})
    base1.fields = {"base_field1": base_field1, "base_field2": base_field2}
    
    dct = {"field1": _PField("value1")}
    bases = [base1]
    name = "fields"
    
    set_fields(dct, bases, name)
    
    assert "fields" in dct
    assert "base_field1" in dct["fields"]
    assert "base_field2" in dct["fields"]
    assert "field1" in dct["fields"]
    assert dct["fields"]["base_field1"] is base_field1
    assert dct["fields"]["base_field2"] is base_field2


def test_set_fields_multiple_bases():
    class _PField:
        def __init__(self, value=None):
            self.value = value
    
    base_field1 = _PField("base1")
    base_field2 = _PField("base2")
    
    base1 = type('Base1', (), {})
    base1.fields = {"base_field1": base_field1}
    
    base2 = type('Base2', (), {})
    base2.fields = {"base_field2": base_field2}
    
    dct = {}
    bases = [base1, base2]
    name = "fields"
    
    set_fields(dct, bases, name)
    
    assert "fields" in dct
    assert "base_field1" in dct["fields"]
    assert "base_field2" in dct["fields"]


def test_set_fields_no_bases_no_pfields():
    class _PField:
        pass
    
    dct = {"key1": "value1", "key2": "value2"}
    bases = []
    name = "fields"
    
    set_fields(dct, bases, name)
    
    assert dct["fields"] == {}
    assert dct["key1"] == "value1"
    assert dct["key2"] == "value2"


# LLM-generated content at query #66
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent import pmap, field, PClass
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    
    # Create a PMap field type
    test_pmap = pmap({'a': 1, 'b': 2})
    
    # Define a test class with a PMap field
    class TestClass(PClass):
        data = field(type=pmap)
    
    # Register the pmap field type
    key_type = str
    value_type = int
    _pmap_field_types[key_type, value_type] = TestClass.data
    
    # Test data to restore
    test_data = {'a': 1, 'b': 2}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap with the correct data
    assert isinstance(result, type(pmap()))
    assert result == pmap(test_data)


# LLM-generated content at query #67
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    checked_value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    
    assert result == "serialized_value"
    assert isinstance(checked_value, CheckedType)
    assert PFIELD_NO_SERIALIZER is PFIELD_NO_SERIALIZER


# LLM-generated content at query #68
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class MockField:
        def __init__(self):
            self.type = [str, int]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: "default"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = [str, 123]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: "default"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type parameter expected" in str(e)


def test_check_field_parameters_invalid_initial_type():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = 123
            self.invariant = lambda x: True
            self.factory = lambda: "default"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


def test_check_field_parameters_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = "not_callable"
            self.factory = lambda: "default"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Invariant must be callable" in str(e)


def test_check_field_parameters_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = "not_callable"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Factory must be callable" in str(e)


def test_check_field_parameters_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: "default"
            self.serializer = "not_callable"
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Serializer must be callable" in str(e)


def test_check_field_parameters_string_type():
    class MockField:
        def __init__(self):
            self.type = ["str", int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: 0
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_callable_initial():
    class MockField:
        def __init__(self):
            self.type = [str]
            self.initial = lambda: "default"
            self.invariant = lambda x: True
            self.factory = lambda: "default"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_empty_type():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = []
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: "default"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


# LLM-generated content at query #69
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    def passing_invariant(subject):
        return (True, None)
    
    invariants = [failing_invariant, passing_invariant]
    subject = "test_subject"
    
    error_codes = tuple(error_code for is_ok, error_code in
                        (invariant(subject) for invariant in invariants) if not is_ok)
    
    assert error_codes == ("ERROR_CODE_1",)
    assert bool(error_codes) is True


# LLM-generated content at query #70
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: MockCheckedType()
    pfield = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert pfield._factory is factory_func


# LLM-generated content at query #71
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    optional_value = False
    result = pmap_field(str, int, optional=optional_value)
    
    assert result is not None
    assert result.mandatory is True


# LLM-generated content at query #72
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert issubclass(result, PVector)
    assert result.__type__ == int
    assert result.__invariant__ is None


def test_make_seq_field_type_caches_type():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    result_int = _make_seq_field_type(PVector, int, None)
    result_str = _make_seq_field_type(PVector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    def my_invariant(x):
        return x > 0
    
    result = _make_seq_field_type(PVector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    result = _make_seq_field_type(PVector, int, None)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


# LLM-generated content at query #73
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pmap, pvec, v
    from pyrsistent._checked_types import CheckedPVector, CheckedPSet, _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    # Create a checked class with a sequence field
    class MyCheckedClass(CheckedPVector):
        __type__ = int
    
    # Register the type in _seq_field_types
    checked_class = MyCheckedClass
    item_type = int
    _seq_field_types[checked_class, item_type] = MyCheckedClass
    
    # Test data to restore
    test_data = [1, 2, 3]
    
    # Call _restore_seq_field_pickle
    result = _restore_seq_field_pickle(checked_class, item_type, test_data)
    
    # Verify the result is a properly restored instance
    assert result == pvec([1, 2, 3])
    assert isinstance(result, MyCheckedClass)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #74
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_field_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__name__ == "StringToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_creates_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__name__ == "StringToIntPMap"
    assert result2.__name__ == "IntToStringPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    instance = result()
    
    assert hasattr(instance, '__reduce__')
    assert callable(instance.__reduce__)


def test_make_pmap_field_type_reduce_returns_correct_tuple():
    from pyrsistent._field_common import _make_pmap_field_type, _restore_pmap_field_pickle
    
    result = _make_pmap_field_type(str, int)
    instance = result()
    reduce_result = instance.__reduce__()
    
    assert reduce_result[0] == _restore_pmap_field_pickle
    assert reduce_result[1][0] == str
    assert reduce_result[1][1] == int
    assert reduce_result[1][2] == {}


# LLM-generated content at query #75
#--------------------------

```python
def test_pmap_field_optional_factory_returns_none():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap field with optional=True
    pmap_fld = pmap_field(str, int, optional=True)
    
    # Get the factory function
    factory_func = pmap_fld.factory
    
    # Call factory with None argument
    result = factory_func(None)
    
    # Assert that the result is None (the predicate at line 2 of the factory function)
    assert result is None


# LLM-generated content at query #76
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, None)
    
    def invariant_pass2(subject):
        return (True, None)
    
    invariants = [invariant_pass, invariant_pass2]
    subject = "test_subject"
    
    try:
        from check_global_invariants import check_global_invariants
    except ImportError:
        def check_global_invariants(subject, invariants):
            error_codes = tuple(error_code for is_ok, error_code in
                                (invariant(subject) for invariant in invariants) if not is_ok)
            if error_codes:
                class InvariantException(Exception):
                    def __init__(self, error_codes, arg2, msg):
                        self.error_codes = error_codes
                        super().__init__(msg)
                raise InvariantException(error_codes, (), 'Global invariant failed')
    
    result = check_global_invariants(subject, invariants)
    assert result is None


# LLM-generated content at query #77
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__reduce__')


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_different_types_different_results():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result1 = _make_seq_field_type(PVector, int, None)
    result2 = _make_seq_field_type(PVector, str, None)
    
    assert result1 is not result2
    assert result1.__type__ == int
    assert result2.__type__ == str


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    def my_invariant(x):
        return x > 0
    
    result = _make_seq_field_type(PVector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant
    assert result.__type__ == int


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import PVector
    
    result = _make_seq_field_type(PVector, int, None)
    
    assert callable(result.__reduce__)


# LLM-generated content at query #78
#--------------------------

```python
def test_set_fields_pfield_isinstance():
    class _PField:
        pass
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    pfield_instance = _PField()
    test_dict = {'field1': pfield_instance, 'field2': 'not_pfield'}
    bases = []
    name = 'fields'
    
    set_fields(test_dict, bases, name)
    
    assert 'field1' in test_dict[name]
    assert test_dict[name]['field1'] is pfield_instance
    assert 'field1' not in test_dict
    assert 'field2' in test_dict


# LLM-generated content at query #79
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pvec, pset
    from pyrsistent._checked_types import CheckedPVector, _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    # Create a checked class with a sequence field
    class MyCheckedClass(CheckedPVector):
        __type__ = PVector
    
    # Register the type in _seq_field_types
    item_type = int
    _seq_field_types[MyCheckedClass, item_type] = MyCheckedClass
    
    # Create test data
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(MyCheckedClass, item_type, test_data)
    
    # Verify the result is created correctly
    assert result is not None
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #80
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: MockCheckedType()
    
    pfield = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert pfield._factory is factory_func


# LLM-generated content at query #81
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    # The predicate at line 3: not isinstance(t, type) and not isinstance(t, str)
    # evaluates to False when t is either a type or a string
    # Test with a type (int)
    assert isinstance(int, type)
    assert not (not isinstance(int, type) and not isinstance(int, str))
    
    # Test with a string
    assert isinstance("str", str)
    assert not (not isinstance("str", type) and not isinstance("str", str))


# LLM-generated content at query #82
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    assert result.mandatory is True
    assert result.initial == pmap()
    assert result.factory is not None


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert result.initial == pmap()
    assert result.factory is not None


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(val):
        return True, None
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.type is not None
    assert result.mandatory is True
    assert result.invariant is not None


def test_pmap_field_factory_none_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    factory_result = result.factory(None)
    assert factory_result is None


def test_pmap_field_factory_with_dict_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    test_dict = {'a': 1, 'b': 2}
    factory_result = result.factory(test_dict)
    assert factory_result is not None


def test_pmap_field_factory_without_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    test_dict = {'a': 1, 'b': 2}
    factory_result = result.factory(test_dict)
    assert factory_result is not None


# LLM-generated content at query #83
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    checked_value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    
    assert result == "serialized_value"


# LLM-generated content at query #84
#--------------------------

```python
def test_make_pmap_field_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    # Test with basic types
    pmap_type_1 = _make_pmap_field_type(str, int)
    assert pmap_type_1 is not None
    assert pmap_type_1.__key_type__ == str
    assert pmap_type_1.__value_type__ == int
    assert "PMap" in pmap_type_1.__name__
    
    # Test that same types return cached instance
    pmap_type_1_again = _make_pmap_field_type(str, int)
    assert pmap_type_1 is pmap_type_1_again
    
    # Test with different types
    pmap_type_2 = _make_pmap_field_type(int, str)
    assert pmap_type_2 is not None
    assert pmap_type_2.__key_type__ == int
    assert pmap_type_2.__value_type__ == str
    assert pmap_type_2 is not pmap_type_1
    
    # Test __reduce__ method exists
    assert hasattr(pmap_type_1, '__reduce__')
    
    # Test with other types
    pmap_type_3 = _make_pmap_field_type(float, bool)
    assert pmap_type_3 is not None
    assert pmap_type_3.__key_type__ == float
    assert pmap_type_3.__value_type__ == bool


# LLM-generated content at query #85
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a PMap field type
    key_type = str
    value_type = int
    
    # Ensure the type exists in the registry
    pmap_field_type = _pmap_field_types.get((key_type, value_type))
    if pmap_field_type is None:
        from pyrsistent._field_common import PMapField
        pmap_field_type = PMapField(key_type, value_type)
        _pmap_field_types[key_type, value_type] = pmap_field_type
    
    # Create test data
    test_data = pmap({'a': 1, 'b': 2})
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap instance
    assert isinstance(result, type(pmap({})))
    assert result == test_data
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #86
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, 'OK1')
    
    def invariant2(subject):
        return (True, 'OK2')
    
    check_global_invariants('test_subject', [invariant1, invariant2])


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (False, 'ERROR1')
    
    def invariant2(subject):
        return (True, 'OK2')
    
    try:
        check_global_invariants('test_subject', [invariant1, invariant2])
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('ERROR1',)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, 'ERROR1')
    
    def invariant2(subject):
        return (False, 'ERROR2')
    
    def invariant3(subject):
        return (True, 'OK3')
    
    try:
        check_global_invariants('test_subject', [invariant1, invariant2, invariant3])
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('ERROR1', 'ERROR2')
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_list():
    check_global_invariants('test_subject', [])


def test_check_global_invariants_with_subject_data():
    def invariant_check_subject(subject):
        if subject == 'valid':
            return (True, 'VALID')
        else:
            return (False, 'INVALID_SUBJECT')
    
    check_global_invariants('valid', [invariant_check_subject])


def test_check_global_invariants_subject_validation_fails():
    def invariant_check_subject(subject):
        if subject == 'valid':
            return (True, 'VALID')
        else:
            return (False, 'INVALID_SUBJECT')
    
    try:
        check_global_invariants('invalid', [invariant_check_subject])
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('INVALID_SUBJECT',)


# LLM-generated content at query #87
#--------------------------

```python
def test_pmap_field_creates_field_with_correct_type_when_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import PMap
    
    result = pmap_field(str, int, optional=False)
    
    assert result is not None
    assert hasattr(result, 'type')
    assert hasattr(result, 'factory')
    assert hasattr(result, 'mandatory')
    assert result.mandatory is True


# LLM-generated content at query #88
#--------------------------

```python
def test_set_fields():
    from unittest.mock import Mock
    
    # Test case 1: Basic functionality with empty bases
    dct = {}
    bases = []
    name = "fields"
    set_fields(dct, bases, name)
    assert "fields" in dct
    assert dct["fields"] == {}

    # Test case 2: With _PField instances in dct
    class _PField:
        pass
    
    field1 = _PField()
    field2 = _PField()
    dct = {"field1": field1, "field2": field2, "other": "value"}
    bases = []
    name = "fields"
    set_fields(dct, bases, name)
    assert dct["fields"] == {"field1": field1, "field2": field2}
    assert "field1" not in dct
    assert "field2" not in dct
    assert dct["other"] == "value"

    # Test case 3: With base class containing fields
    class Base:
        pass
    Base.__dict__ = {"fields": {"base_field": _PField()}}
    
    dct = {}
    bases = [Base]
    name = "fields"
    set_fields(dct, bases, name)
    assert "fields" in dct
    assert "base_field" in dct["fields"]

    # Test case 4: Combining base fields with dct fields
    class Base2:
        pass
    base_field = _PField()
    Base2.__dict__ = {"fields": {"base_field": base_field}}
    
    dct_field = _PField()
    dct = {"dct_field": dct_field}
    bases = [Base2]
    name = "fields"
    set_fields(dct, bases, name)
    assert "fields" in dct
    assert "base_field" in dct["fields"]
    assert "dct_field" in dct["fields"]
    assert "dct_field" not in dct or dct.get("dct_field") != dct_field

    # Test case 5: Multiple bases with overlapping fields
    class Base3:
        pass
    class Base4:
        pass
    field_a = _PField()
    field_b = _PField()
    Base3.__dict__ = {"fields": {"field_a": field_a}}
    Base4.__dict__ = {"fields": {"field_b": field_b}}
    
    dct = {}
    bases = [Base3, Base4]
    name = "fields"
    set_fields(dct, bases, name)
    assert "fields" in dct
    assert "field_a" in dct["fields"]
    assert "field_b" in dct["fields"]


# LLM-generated content at query #89
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class Field:
        def __init__(self):
            self.type = [int, 123]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_type_parameter_as_string():
    class Field:
        def __init__(self):
            self.type = [int, 'CustomType']
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_initial_type():
    PFIELD_NO_INITIAL = object()
    
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = "invalid"
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_callable_initial():
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = lambda: 42
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = "not callable"
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_non_callable_factory():
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_non_callable_serializer():
    class Field:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = "not callable"
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_check_field_parameters_empty_type_list():
    class Field:
        def __init__(self):
            self.type = []
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: []
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


# LLM-generated content at query #90
#--------------------------

```python
def test_pmap_field_optional_type_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent._checked_types import optional
    from pyrsistent import pmap
    
    # Test with optional=True
    result_optional = pmap_field(str, int, optional=True)
    
    # The type should be a tuple containing TheMap type and NoneType
    assert isinstance(result_optional.type, tuple)
    assert len(result_optional.type) == 2
    assert type(None) in result_optional.type
    
    # Test with optional=False
    result_not_optional = pmap_field(str, int, optional=False)
    
    # The type should be a single type, not a tuple with None
    assert not (isinstance(result_not_optional.type, tuple) and type(None) in result_not_optional.type)


# LLM-generated content at query #91
#--------------------------

```python
def test_check_global_invariants_all_pass():
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    check_global_invariants("test_subject", invariants)


def test_check_global_invariants_single_failure():
    invariants = [
        lambda x: (True, None),
        lambda x: (False, "error_1"),
        lambda x: (True, None),
    ]
    try:
        check_global_invariants("test_subject", invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("error_1",)


def test_check_global_invariants_multiple_failures():
    invariants = [
        lambda x: (False, "error_1"),
        lambda x: (False, "error_2"),
        lambda x: (True, None),
        lambda x: (False, "error_3"),
    ]
    try:
        check_global_invariants("test_subject", invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("error_1", "error_2", "error_3")


def test_check_global_invariants_no_invariants():
    invariants = []
    check_global_invariants("test_subject", invariants)


def test_check_global_invariants_exception_message():
    invariants = [
        lambda x: (False, "test_error"),
    ]
    try:
        check_global_invariants("test_subject", invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_with_different_subjects():
    subject = {"key": "value"}
    invariants = [
        lambda x: (True, None) if isinstance(x, dict) else (False, "not_dict"),
    ]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_invariant_receives_correct_subject():
    subject = "test_value"
    received_subject = []
    
    def capture_subject(x):
        received_subject.append(x)
        return (True, None)
    
    invariants = [capture_subject]
    check_global_invariants(subject, invariants)
    assert received_subject[0] == "test_value"


# LLM-generated content at query #92
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pmap, pvec
    from pyrsistent._checked_types import CheckedPVector, CheckedPSet, _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    # Create a checked class with a sequence field
    class MyCheckedClass(CheckedPVector):
        __type__ = int
    
    # Register the type in _seq_field_types
    _seq_field_types[MyCheckedClass, int] = MyCheckedClass
    
    # Create test data
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(MyCheckedClass, int, test_data)
    
    # Assertions
    assert result is not None
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #93
#--------------------------

```python
def test_pfield_init_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: MockCheckedType()
    field = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert field._factory is factory_func


# LLM-generated content at query #94
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "json"
    
    result = serialize(serializer, format, value)
    assert result == "serialized_value"


# LLM-generated content at query #95
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_error_codes_exist():
    def invariant_fails(subject):
        return (False, "ERROR_CODE_1")
    
    def invariant_passes(subject):
        return (True, None)
    
    invariants = [invariant_fails, invariant_passes]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE_1",)
        assert e.invariant_codes == ()
        assert e.message == 'Global invariant failed'


# LLM-generated content at query #96
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    key_type = str
    value_type = int
    data = {'a': 1, 'b': 2}
    
    field_type = _pmap_field_types.get((key_type, value_type))
    if field_type is None:
        from pyrsistent._field_common import PMapField
        field_type = PMapField(key_type, value_type)
        _pmap_field_types[key_type, value_type] = field_type
    
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    
    assert result is not None
    assert isinstance(result, type(pmap(data)))


def test_restore_pmap_field_pickle_empty():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    key_type = str
    value_type = str
    data = {}
    
    field_type = _pmap_field_types.get((key_type, value_type))
    if field_type is None:
        from pyrsistent._field_common import PMapField
        field_type = PMapField(key_type, value_type)
        _pmap_field_types[key_type, value_type] = field_type
    
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    
    assert result is not None
    assert len(result) == 0


def test_restore_pmap_field_pickle_with_types():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    key_type = int
    value_type = float
    data = {1: 1.5, 2: 2.5, 3: 3.5}
    
    field_type = _pmap_field_types.get((key_type, value_type))
    if field_type is None:
        from pyrsistent._field_common import PMapField
        field_type = PMapField(key_type, value_type)
        _pmap_field_types[key_type, value_type] = field_type
    
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    
    assert result is not None
    assert len(result) == 3


# LLM-generated content at query #97
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache to ensure a fresh test
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__reduce__')


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_with_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    result_int = _make_seq_field_type(PVector, int, None)
    result_str = _make_seq_field_type(PVector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    def my_invariant(val):
        return val > 0
    
    result = _make_seq_field_type(PVector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, None)
    
    assert callable(result.__reduce__)


# LLM-generated content at query #98
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__name__ == "StrToIntPMap"
    assert result2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_multiple_key_value_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type((str, int), (float, bool))
    
    assert "StrInt" in result.__name__
    assert "FloatBool" in result.__name__


# LLM-generated content at query #99
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap_field with optional=False
    field_result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (which is `if optional:`) should evaluate to False
    # when optional=False, so the factory should be set to TheMap.create
    # We verify this by checking that the factory is callable and not the custom function
    assert callable(field_result.factory)
    
    # Verify that when optional=False, the factory doesn't handle None specially
    # The factory should be TheMap.create, not the wrapper function
    test_map = pmap({'a': 1})
    result = field_result.factory(test_map)
    assert result == test_map


# LLM-generated content at query #100
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector, pv
    
    # Clear the cache to ensure a clean test
    _seq_field_types.clear()
    
    # Test creating a seq field type with int item type
    result_type = _make_seq_field_type(PVector, int, None)
    
    assert result_type is not None
    assert issubclass(result_type, PVector)
    assert result_type.__type__ == int
    assert result_type.__invariant__ is None
    assert hasattr(result_type, '__name__')
    assert 'Vector' in result_type.__name__
    
    # Test that the same call returns the cached type
    cached_result = _make_seq_field_type(PVector, int, None)
    assert cached_result is result_type
    
    # Test with a different item type
    result_type_str = _make_seq_field_type(PVector, str, None)
    assert result_type_str is not result_type
    assert result_type_str.__type__ == str
    
    # Test that __reduce__ method exists and works
    instance = result_type([1, 2, 3])
    reduce_result = instance.__reduce__()
    assert reduce_result is not None
    assert len(reduce_result) == 2
    assert callable(reduce_result[0])
    
    _seq_field_types.clear()


# LLM-generated content at query #101
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    def passing_invariant(subject):
        return (True, "ERROR_CODE_2")
    
    invariants = [failing_invariant, passing_invariant]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE_1",)
        assert e.invariant_errors == ()
        assert e.message == 'Global invariant failed'


# LLM-generated content at query #102
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pmap
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    # Create a checked vector type
    checked_type = CheckedPVector(int)
    _seq_field_types[(type(checked_type), int)] = checked_type
    
    # Test data to restore
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(type(checked_type), int, test_data)
    
    # Verify the result is a PVector with the correct data
    assert isinstance(result, PVector)
    assert list(result) == test_data


def test_restore_seq_field_pickle_with_strings():
    from pyrsistent import PVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    # Create a checked vector type for strings
    checked_type = CheckedPVector(str)
    _seq_field_types[(type(checked_type), str)] = checked_type
    
    # Test data to restore
    test_data = ['a', 'b', 'c']
    
    # Call the function
    result = _restore_seq_field_pickle(type(checked_type), str, test_data)
    
    # Verify the result
    assert isinstance(result, PVector)
    assert list(result) == test_data


def test_restore_seq_field_pickle_empty_list():
    from pyrsistent import PVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    # Create a checked vector type
    checked_type = CheckedPVector(int)
    _seq_field_types[(type(checked_type), int)] = checked_type
    
    # Test data to restore (empty)
    test_data = []
    
    # Call the function
    result = _restore_seq_field_pickle(type(checked_type), int, test_data)
    
    # Verify the result is an empty PVector
    assert isinstance(result, PVector)
    assert len(result) == 0
    assert list(result) == []


# LLM-generated content at query #103
#--------------------------

```python
def test_set_fields():
    from types import SimpleNamespace
    
    class _PField:
        def __init__(self, value):
            self.value = value
    
    # Test 1: Basic functionality with no bases and no PField instances
    dct1 = {}
    bases1 = []
    set_fields(dct1, bases1, "fields")
    assert "fields" in dct1
    assert dct1["fields"] == {}
    
    # Test 2: With PField instances in dct
    dct2 = {"field1": _PField("value1"), "field2": "regular_value"}
    bases2 = []
    set_fields(dct2, bases2, "fields")
    assert "fields" in dct2
    assert "field1" in dct2["fields"]
    assert isinstance(dct2["fields"]["field1"], _PField)
    assert "field1" not in dct2
    assert "field2" in dct2
    
    # Test 3: With base classes that have fields
    class BaseClass1:
        pass
    BaseClass1.fields = {"base_field1": _PField("base_value1")}
    
    class BaseClass2:
        pass
    BaseClass2.fields = {"base_field2": _PField("base_value2")}
    
    dct3 = {"field3": _PField("value3")}
    bases3 = [BaseClass1, BaseClass2]
    set_fields(dct3, bases3, "fields")
    assert "fields" in dct3
    assert "base_field1" in dct3["fields"]
    assert "base_field2" in dct3["fields"]
    assert "field3" in dct3["fields"]
    assert "field3" not in dct3
    
    # Test 4: Multiple PField instances
    dct4 = {
        "pf1": _PField("pv1"),
        "pf2": _PField("pv2"),
        "pf3": _PField("pv3"),
        "regular": "value"
    }
    bases4 = []
    set_fields(dct4, bases4, "fields")
    assert len(dct4["fields"]) == 3
    assert "pf1" in dct4["fields"]
    assert "pf2" in dct4["fields"]
    assert "pf3" in dct4["fields"]
    assert "pf1" not in dct4
    assert "pf2" not in dct4
    assert "pf3" not in dct4
    assert "regular" in dct4
    
    # Test 5: Empty bases with empty dct
    dct5 = {}
    bases5 = []
    set_fields(dct5, bases5, "my_fields")
    assert "my_fields" in dct5
    assert dct5["my_fields"] == {}


# LLM-generated content at query #104
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_value"


# LLM-generated content at query #105
#--------------------------

```python
def test_make_pmap_field_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    # Test basic type creation with built-in types
    pmap_type_1 = _make_pmap_field_type(str, int)
    assert pmap_type_1 is not None
    assert pmap_type_1.__key_type__ == str
    assert pmap_type_1.__value_type__ == int
    assert "PMap" in pmap_type_1.__name__
    
    # Test that same types return cached instance
    pmap_type_2 = _make_pmap_field_type(str, int)
    assert pmap_type_1 is pmap_type_2
    
    # Test with different types
    pmap_type_3 = _make_pmap_field_type(int, str)
    assert pmap_type_3 is not pmap_type_1
    assert pmap_type_3.__key_type__ == int
    assert pmap_type_3.__value_type__ == str
    
    # Test __reduce__ method exists
    assert hasattr(pmap_type_1, '__reduce__')
    
    # Test name generation with different types
    pmap_type_4 = _make_pmap_field_type(float, bool)
    assert "PMap" in pmap_type_4.__name__
    assert pmap_type_4.__key_type__ == float
    assert pmap_type_4.__value_type__ == bool


# LLM-generated content at query #106
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: "not isinstance(t, type) and not isinstance(t, str)"
    # should evaluate to False for all t in field.type
    # This means for each t, either isinstance(t, type) is True OR isinstance(t, str) is True
    
    for t in field.type:
        assert isinstance(t, type) or isinstance(t, str)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    value = CheckedType()
    result = serialize(None, "json", value)
    assert result == "serialized_json"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "xml:test_value"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"checked_{format}"
    
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    value = CheckedType()
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_json_" + str(value)


def test_serialize_with_regular_value_and_serializer():
    def my_serializer(format, value):
        return f"result:{format}:{value}"
    
    result = serialize(my_serializer, "csv", 42)
    assert result == "result:csv:42"


def test_serialize_with_checked_type_no_serializer_constant():
    class CheckedType:
        def serialize(self, format):
            return {"format": format, "data": "test"}
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "binary", value)
    assert result == {"format": "binary", "data": "test"}


# LLM-generated content at query #2
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent._checked_types import CheckedPMap
    
    result = pmap_field(str, int)
    
    assert result is not None
    assert result.mandatory is True
    assert result.initial is not None


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    
    assert result.mandatory is True
    assert len(result.type) == 1


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    
    assert result.mandatory is True
    assert result.factory is not None


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field, PFIELD_NO_INVARIANT
    
    def my_invariant(val):
        return True, None
    
    result = pmap_field(str, int, invariant=my_invariant)
    
    assert result.invariant is not None
    assert result.invariant != PFIELD_NO_INVARIANT


def test_pmap_field_factory_none_when_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    
    factory_result = result.factory(None)
    assert factory_result is None


def test_pmap_field_different_types():
    from pyrsistent._field_common import pmap_field
    
    result1 = pmap_field(int, str)
    result2 = pmap_field(str, int)
    
    assert result1 is not None
    assert result2 is not None
    assert result1.type != result2.type


def test_pmap_field_creates_initial_empty_map():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int)
    
    assert len(result.initial) == 0


def test_pmap_field_mandatory_always_true():
    from pyrsistent._field_common import pmap_field
    
    result_optional_false = pmap_field(str, int, optional=False)
    result_optional_true = pmap_field(str, int, optional=True)
    
    assert result_optional_false.mandatory is True
    assert result_optional_true.mandatory is True


# LLM-generated content at query #3
#--------------------------

```python
def test_check_type_with_valid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    from pyrsistent._preconditions import PTypeError
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class TestClass:
        __name__ = "TestClass"
    
    field = MockField((int,))
    check_type(TestClass, field, "test_field", 42)


def test_check_type_with_multiple_valid_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class TestClass:
        __name__ = "TestClass"
    
    field = MockField((int, str))
    check_type(TestClass, field, "test_field", "hello")


def test_check_type_with_none_field_type():
    from pyrsistent._field_common import check_type
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class TestClass:
        __name__ = "TestClass"
    
    field = MockField(None)
    check_type(TestClass, field, "test_field", "any_value")


def test_check_type_with_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._preconditions import PTypeError
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class TestClass:
        __name__ = "TestClass"
    
    field = MockField((int,))
    
    try:
        check_type(TestClass, field, "test_field", "not_an_int")
        assert False, "Expected PTypeError to be raised"
    except PTypeError as e:
        assert e.args[3] == str
        assert "TestClass.test_field" in str(e)


def test_check_type_with_empty_type_tuple():
    from pyrsistent._field_common import check_type
    from pyrsistent._preconditions import PTypeError
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class TestClass:
        __name__ = "TestClass"
    
    field = MockField(())
    
    try:
        check_type(TestClass, field, "test_field", 42)
        assert False, "Expected PTypeError to be raised"
    except PTypeError:
        pass


def test_check_type_with_subclass():
    from pyrsistent._field_common import check_type
    
    class MockField:
        def __init__(self, type_):
            self.type = type_
    
    class TestClass:
        __name__ = "TestClass"
    
    class MyInt(int):
        pass
    
    field = MockField((int,))
    check_type(TestClass, field, "test_field", MyInt(5))


# LLM-generated content at query #4
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(x):
        return x > 0
    
    def dummy_serializer(x):
        return str(x)
    
    def dummy_factory():
        return 42
    
    pfield = _PField(
        type=int,
        invariant=dummy_invariant,
        initial=10,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert pfield.type == int
    assert pfield.invariant == dummy_invariant
    assert pfield.initial == 10
    assert pfield.mandatory == True
    assert pfield._factory == dummy_factory
    assert pfield.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    pfield = _PField(
        type=str,
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert pfield.type == str
    assert pfield.invariant is None
    assert pfield.initial is None
    assert pfield.mandatory == False
    assert pfield._factory is None
    assert pfield.serializer is None


def test_pfield_constructor_with_different_types():
    pfield = _PField(
        type=list,
        invariant=lambda x: len(x) > 0,
        initial=[1, 2, 3],
        mandatory=True,
        factory=list,
        serializer=lambda x: repr(x)
    )
    
    assert pfield.type == list
    assert pfield.initial == [1, 2, 3]
    assert pfield.mandatory == True
    assert pfield._factory == list


# LLM-generated content at query #5
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = [int, 123]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_string_type():
    class MockField:
        def __init__(self):
            self.type = [int, 'CustomType']
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_initial_wrong_type():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = "string"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_initial_callable():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = lambda: 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 5
            self.invariant = "not callable"
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not callable"
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_check_field_parameters_empty_type():
    class MockField:
        def __init__(self):
            self.type = []
            self.initial = "anything"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


# LLM-generated content at query #6
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_field_type():
    from pyrsistent._field_common import _make_pmap_field_type
    pmap_type = _make_pmap_field_type(str, int)
    assert pmap_type.__name__ == "StrToIntPMap"
    assert pmap_type.__key_type__ == str
    assert pmap_type.__value_type__ == int


def test_make_pmap_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_pmap_field_type
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(str, int)
    assert pmap_type1 is pmap_type2


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    pmap_type1 = _make_pmap_field_type(str, int)
    pmap_type2 = _make_pmap_field_type(int, str)
    assert pmap_type1 is not pmap_type2
    assert pmap_type1.__name__ == "StrToIntPMap"
    assert pmap_type2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    pmap_type = _make_pmap_field_type(str, int)
    assert hasattr(pmap_type, '__reduce__')


def test_make_pmap_field_type_with_float_and_bool():
    from pyrsistent._field_common import _make_pmap_field_type
    pmap_type = _make_pmap_field_type(float, bool)
    assert pmap_type.__name__ == "FloatToBoolPMap"
    assert pmap_type.__key_type__ == float
    assert pmap_type.__value_type__ == bool


# LLM-generated content at query #7
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register a pmap field type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent._field_common import _make_pmap_field
        _make_pmap_field(key_type, value_type)
    
    # Create test data
    test_data = pmap({'a': 1, 'b': 2})
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap with the correct data
    assert result == test_data
    assert isinstance(result, type(test_data))


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    PFIELD_NO_INITIAL = object()
    field = MockField()
    
    predicate = (field.initial is not PFIELD_NO_INITIAL and 
                 not callable(field.initial) and 
                 field.type and not any(isinstance(field.initial, t) for t in field.type))
    
    assert predicate is False


# LLM-generated content at query #9
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    class MockValue(CheckedType):
        pass
    
    PFIELD_NO_SERIALIZER = object()
    value = MockValue()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_value"


def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "xml:test_value"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return "checked_serialized"
    
    class MockValue(CheckedType):
        pass
    
    def custom_serializer(format, value):
        return "custom_serialized"
    
    value = MockValue()
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_serialized"


def test_serialize_with_non_checked_type():
    PFIELD_NO_SERIALIZER = object()
    
    def serializer(format, value):
        return f"serialized_{value}"
    
    result = serialize(serializer, "json", "plain_value")
    assert result == "serialized_plain_value"


def test_serialize_with_different_formats():
    def format_serializer(format, value):
        return f"{format}_{value}"
    
    result_json = serialize(format_serializer, "json", "data")
    result_xml = serialize(format_serializer, "xml", "data")
    
    assert result_json == "json_data"
    assert result_xml == "xml_data"


# LLM-generated content at query #10
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pclass, field, pvec
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import _restore_pickle
    
    class TestClass(pclass):
        items = field()
    
    item_type = int
    checked_class = TestClass
    data = [1, 2, 3]
    
    # Register the type in _seq_field_types
    test_type = pvec([1, 2, 3])
    _seq_field_types[(checked_class, item_type)] = type(test_type)
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result is not None
    assert len(result) == 3


# LLM-generated content at query #11
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, "code1")
    
    def invariant2(subject):
        return (True, "code2")
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_one_fails():
    def invariant1(subject):
        return (True, "code1")
    
    def invariant2(subject):
        return (False, "code2")
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("code2",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_fail():
    def invariant1(subject):
        return (False, "error1")
    
    def invariant2(subject):
        return (False, "error2")
    
    def invariant3(subject):
        return (True, "code3")
    
    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("error1", "error2")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_list():
    invariants = []
    subject = "test_subject"
    
    check_global_invariants(subject, invariants)


def test_check_global_invariants_with_different_subject_types():
    def invariant(subject):
        return (True, "ok")
    
    invariants = [invariant]
    
    check_global_invariants(42, invariants)
    check_global_invariants({"key": "value"}, invariants)
    check_global_invariants([1, 2, 3], invariants)
    check_global_invariants(None, invariants)


# LLM-generated content at query #12
#--------------------------

```python
def test_set_fields():
    from types import new_class
    
    class _PField:
        def __init__(self, value=None):
            self.value = value
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    # Test 1: Empty dictionary with no bases
    dct1 = {}
    bases1 = []
    set_fields(dct1, bases1, 'fields')
    assert dct1 == {'fields': {}}
    
    # Test 2: Dictionary with _PField instances
    field1 = _PField('value1')
    field2 = _PField('value2')
    dct2 = {'field1': field1, 'field2': field2, 'other': 'data'}
    bases2 = []
    set_fields(dct2, bases2, 'fields')
    assert 'field1' not in dct2
    assert 'field2' not in dct2
    assert dct2['other'] == 'data'
    assert dct2['fields']['field1'] is field1
    assert dct2['fields']['field2'] is field2
    
    # Test 3: With base classes containing fields
    class Base:
        fields = {'base_field': _PField('base_value')}
    
    field3 = _PField('value3')
    dct3 = {'field3': field3}
    bases3 = [Base]
    set_fields(dct3, bases3, 'fields')
    assert 'field3' not in dct3
    assert dct3['fields']['base_field'].value == 'base_value'
    assert dct3['fields']['field3'] is field3
    
    # Test 4: Multiple bases with fields
    class Base1:
        fields = {'base1_field': _PField('base1_value')}
    
    class Base2:
        fields = {'base2_field': _PField('base2_value')}
    
    field4 = _PField('value4')
    dct4 = {'field4': field4}
    bases4 = [Base1, Base2]
    set_fields(dct4, bases4, 'fields')
    assert dct4['fields']['base1_field'].value == 'base1_value'
    assert dct4['fields']['base2_field'].value == 'base2_value'
    assert dct4['fields']['field4'] is field4
    
    # Test 5: Mixed _PField and non-_PField attributes
    field5 = _PField('value5')
    dct5 = {'pfield': field5, 'method': lambda x: x, 'attr': 42}
    bases5 = []
    set_fields(dct5, bases5, 'fields')
    assert 'pfield' not in dct5
    assert dct5['method'](10) == 10
    assert dct5['attr'] == 42
    assert dct5['fields']['pfield'] is field5


# LLM-generated content at query #13
#--------------------------

```python
def test_check_type_predicate_evaluates_to_true():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent import PTypeError
    
    class SimpleClass:
        pass
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockDestinationClass:
        __name__ = "MockDestinationClass"
    
    # Test case where predicate evaluates to True (no error should be raised)
    field = MockField((int,))
    destination_cls = MockDestinationClass()
    value = 42
    
    check_type(destination_cls, field, "test_field", value)


def test_check_type_predicate_evaluates_to_false():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockDestinationClass:
        __name__ = "MockDestinationClass"
    
    # Test case where predicate evaluates to False (error should be raised)
    field = MockField((int,))
    destination_cls = MockDestinationClass()
    value = "not an int"
    
    try:
        check_type(destination_cls, field, "test_field", value)
        assert False, "Expected PTypeError to be raised"
    except PTypeError:
        pass


def test_check_type_with_none_field_type():
    from pyrsistent._field_common import check_type
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockDestinationClass:
        __name__ = "MockDestinationClass"
    
    # Test case where field.type is None (predicate should be False, no error)
    field = MockField(None)
    destination_cls = MockDestinationClass()
    value = "any value"
    
    check_type(destination_cls, field, "test_field", value)


def test_check_type_with_multiple_types():
    from pyrsistent._field_common import check_type
    from pyrsistent import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockDestinationClass:
        __name__ = "MockDestinationClass"
    
    # Test case with multiple types where value matches one of them
    field = MockField((int, str))
    destination_cls = MockDestinationClass()
    value = "valid string"
    
    check_type(destination_cls, field, "test_field", value)


def test_check_type_with_multiple_types_no_match():
    from pyrsistent._field_common import check_type
    from pyrsistent import PTypeError
    
    class MockField:
        def __init__(self, field_type):
            self.type = field_type
    
    class MockDestinationClass:
        __name__ = "MockDestinationClass"
    
    # Test case with multiple types where value matches none of them
    field = MockField((int, str))
    destination_cls = MockDestinationClass()
    value = []
    
    try:
        check_type(destination_cls, field, "test_field", value)
        assert False, "Expected PTypeError to be raised"
    except PTypeError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def simple_factory():
        return None
    
    field = MockField(int, simple_factory)
    result = is_field_ignore_extra_complaint(int, field, False)
    assert result is False


def test_is_field_ignore_extra_complaint_returns_false_when_type_cls_mismatch():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def simple_factory():
        return None
    
    field = MockField(str, simple_factory)
    result = is_field_ignore_extra_complaint(int, field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def simple_factory():
        return None
    
    field = MockField({int}, simple_factory)
    result = is_field_ignore_extra_complaint(int, field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_met():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def factory_with_ignore_extra(ignore_extra=False):
        return None
    
    field = MockField({int}, factory_with_ignore_extra)
    result = is_field_ignore_extra_complaint(int, field, True)
    assert result is True


def test_is_field_ignore_extra_complaint_with_set_type():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def factory_with_ignore_extra(ignore_extra=False):
        return None
    
    field = MockField(set(), factory_with_ignore_extra)
    result = is_field_ignore_extra_complaint(int, field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_with_empty_tuple_type():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def factory_with_ignore_extra(ignore_extra=False):
        return None
    
    field = MockField((), factory_with_ignore_extra)
    result = is_field_ignore_extra_complaint(int, field, True)
    assert result is False


# LLM-generated content at query #15
#--------------------------

```python
def test_check_global_invariants_all_pass():
    invariant1 = lambda x: (True, 'code1')
    invariant2 = lambda x: (True, 'code2')
    subject = "test_subject"
    
    check_global_invariants(subject, [invariant1, invariant2])


def test_check_global_invariants_single_failure():
    invariant1 = lambda x: (True, 'code1')
    invariant2 = lambda x: (False, 'code2')
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, [invariant1, invariant2])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('code2',)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    invariant1 = lambda x: (False, 'error1')
    invariant2 = lambda x: (False, 'error2')
    invariant3 = lambda x: (True, 'code3')
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, [invariant1, invariant2, invariant3])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('error1', 'error2')
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    
    check_global_invariants(subject, [])


def test_check_global_invariants_all_fail():
    invariant1 = lambda x: (False, 'error1')
    invariant2 = lambda x: (False, 'error2')
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, [invariant1, invariant2])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('error1', 'error2')


# LLM-generated content at query #16
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pmap, pvec, field, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class MyCheckedClass(CheckedPVector):
        x = field(type=int)
    
    item_type = int
    checked_class = MyCheckedClass
    
    # Register the type in _seq_field_types
    test_type = pvec([1, 2, 3])
    _seq_field_types[checked_class, item_type] = MyCheckedClass
    
    # Test data to restore
    data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    # Verify the result is a checked vector with correct data
    assert result == pvec([1, 2, 3])
    assert isinstance(result, MyCheckedClass)


# LLM-generated content at query #17
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    class MockCheckedType(CheckedType):
        pass
    
    value = MockCheckedType()
    PFIELD_NO_SERIALIZER = object()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_json"


def test_serialize_with_regular_serializer():
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "xml:test_value"


def test_serialize_with_checked_type_and_custom_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"checked_{format}"
    
    class MockCheckedType(CheckedType):
        pass
    
    def custom_serializer(format, value):
        return f"custom_{format}"
    
    value = MockCheckedType()
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_json"


def test_serialize_with_non_checked_type_and_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    
    def no_serializer_func(format, value):
        return f"result_{format}_{value}"
    
    result = serialize(no_serializer_func, "csv", "data")
    assert result == "result_csv_data"


def test_serialize_with_lambda_serializer():
    serializer = lambda fmt, val: f"lambda_{fmt}_{val}"
    result = serialize(serializer, "txt", "content")
    assert result == "lambda_txt_content"


# LLM-generated content at query #18
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    
    # Create a mock field class for testing
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    # Test case 1: ignore_extra is False, should return False
    def factory_with_ignore_extra(ignore_extra=None):
        pass
    
    field1 = MockField({int}, factory_with_ignore_extra)
    result1 = is_field_ignore_extra_complaint(object, field1, False)
    assert result1 is False
    
    # Test case 2: ignore_extra is True but field type doesn't match, should return False
    field2 = MockField(set(), factory_with_ignore_extra)
    result2 = is_field_ignore_extra_complaint(object, field2, True)
    assert result2 is False
    
    # Test case 3: ignore_extra is True, type matches, but factory has no ignore_extra param
    def factory_without_ignore_extra():
        pass
    
    field3 = MockField({int}, factory_without_ignore_extra)
    result3 = is_field_ignore_extra_complaint(object, field3, True)
    assert result3 is False
    
    # Test case 4: ignore_extra is True, type matches (set), factory has ignore_extra param
    field4 = MockField({int}, factory_with_ignore_extra)
    result4 = is_field_ignore_extra_complaint(object, field4, True)
    assert result4 is True
    
    # Test case 5: ignore_extra is True, type matches (tuple with valid class), factory has ignore_extra param
    field5 = MockField((int,), factory_with_ignore_extra)
    result5 = is_field_ignore_extra_complaint(int, field5, True)
    assert result5 is True
    
    # Test case 6: ignore_extra is True, type matches but factory lacks ignore_extra param
    def factory_no_params():
        pass
    
    field6 = MockField({str}, factory_no_params)
    result6 = is_field_ignore_extra_complaint(object, field6, True)
    assert result6 is False
    
    # Test case 7: empty tuple type should return False
    field7 = MockField((), factory_with_ignore_extra)
    result7 = is_field_ignore_extra_complaint(object, field7, True)
    assert result7 is False


# LLM-generated content at query #19
#--------------------------

```python
def test_set_fields_pfield_instance():
    class _PField:
        pass
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    test_dict = {'field1': _PField(), 'field2': 'not_pfield'}
    set_fields(test_dict, [], 'fields')
    
    assert 'field1' not in test_dict
    assert 'field1' in test_dict['fields']
    assert isinstance(test_dict['fields']['field1'], _PField)


# LLM-generated content at query #20
#--------------------------

```python
def test_make_pmap_field_type_creates_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    
    result = _make_pmap_field_type(str, int)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_generates_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert result.__name__ == "StringToIntPMap"


def test_make_pmap_field_type_with_different_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(int, str)
    assert result.__name__ == "IntToStringPMap"
    assert result.__key_type__ == int
    assert result.__value_type__ == str


def test_make_pmap_field_type_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type, _restore_pmap_field_pickle
    
    result = _make_pmap_field_type(str, int)
    instance = result()
    reduce_result = instance.__reduce__()
    assert reduce_result[0] == _restore_pmap_field_pickle
    assert reduce_result[1][0] == str
    assert reduce_result[1][1] == int
    assert reduce_result[1][2] == {}


def test_make_pmap_field_type_with_list_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, list)
    assert result.__name__ == "StringToListPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == list


def test_make_pmap_field_type_multiple_distinct_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    result3 = _make_pmap_field_type(str, float)
    
    assert result1 is not result2
    assert result1 is not result3
    assert result2 is not result3
    assert result1.__name__ == "StringToIntPMap"
    assert result2.__name__ == "IntToStringPMap"
    assert result3.__name__ == "StringToFloatPMap"


# LLM-generated content at query #21
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent import pmap, PMap
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent._checked_types import _restore_pickle
    
    # Create a PMap field type
    key_type = str
    value_type = int
    
    # Register the type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import field
        test_type = type('TestType', (), {
            'test_field': field(type=PMap[key_type:value_type])
        })
        _pmap_field_types[key_type, value_type] = PMap
    
    # Create test data
    test_data = {'a': 1, 'b': 2, 'c': 3}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a PMap
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert len(result) == 3


def test_restore_pmap_field_pickle_empty():
    from pyrsistent import pmap, PMap
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    
    key_type = str
    value_type = str
    
    # Register the type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        _pmap_field_types[key_type, value_type] = PMap
    
    # Create empty test data
    test_data = {}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is an empty PMap
    assert isinstance(result, PMap)
    assert len(result) == 0


def test_restore_pmap_field_pickle_with_int_keys():
    from pyrsistent import pmap, PMap
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    
    key_type = int
    value_type = float
    
    # Register the type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        _pmap_field_types[key_type, value_type] = PMap
    
    # Create test data
    test_data = {1: 1.5, 2: 2.5, 3: 3.5}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result
    assert isinstance(result, PMap)
    assert result[1] == 1.5
    assert result[2] == 2.5
    assert result[3] == 3.5


# LLM-generated content at query #22
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    pfield_instance = _PField()
    test_dict = {'field1': pfield_instance, 'field2': 'not_a_pfield'}
    bases = []
    name = 'fields'
    
    set_fields(test_dict, bases, name)
    
    assert 'field1' in test_dict['fields']
    assert test_dict['fields']['field1'] is pfield_instance
    assert 'field1' not in test_dict
    assert 'field2' in test_dict


# LLM-generated content at query #23
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, None)
    
    def invariant_pass_2(subject):
        return (True, None)
    
    invariants = [invariant_pass, invariant_pass_2]
    subject = object()
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #24
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: MockCheckedType()
    
    pfield = _PField(
        type=set(),
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=None
    )
    
    assert pfield._factory is factory_func


# LLM-generated content at query #25
#--------------------------

```python
def test_sequence_field_with_optional_true():
    from pyrsistent._field_common import _sequence_field, _PField
    from pyrsistent import CheckedPVector, v
    
    result = _sequence_field(CheckedPVector, int, optional=True, initial=v())
    
    assert isinstance(result, _PField)
    assert result.mandatory is True
    assert result.initial == v()


def test_sequence_field_with_optional_false():
    from pyrsistent._field_common import _sequence_field, _PField
    from pyrsistent import CheckedPVector, v
    
    result = _sequence_field(CheckedPVector, int, optional=False, initial=v())
    
    assert isinstance(result, _PField)
    assert result.mandatory is True
    assert result.initial == v()


def test_sequence_field_factory_with_none_when_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector, v
    
    result = _sequence_field(CheckedPVector, int, optional=True, initial=v())
    
    factory_result = result.factory(None)
    assert factory_result is None


def test_sequence_field_factory_with_list_when_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector, v
    
    result = _sequence_field(CheckedPVector, int, optional=True, initial=v())
    
    factory_result = result.factory([1, 2, 3])
    assert len(factory_result) == 3
    assert factory_result[0] == 1


def test_sequence_field_factory_without_optional():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector, v
    
    result = _sequence_field(CheckedPVector, int, optional=False, initial=v())
    
    factory_result = result.factory([1, 2, 3])
    assert len(factory_result) == 3


def test_sequence_field_with_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector, v
    
    def my_invariant(value):
        return (True, None)
    
    result = _sequence_field(CheckedPVector, int, optional=False, initial=v(), invariant=my_invariant)
    
    assert result.invariant is not None
    assert result.mandatory is True


def test_sequence_field_with_item_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector, v
    
    def item_invariant(value):
        return (True, None)
    
    result = _sequence_field(CheckedPVector, int, optional=False, initial=v(), item_invariant=item_invariant)
    
    assert isinstance(result, type(result))
    assert result.mandatory is True


# LLM-generated content at query #26
#--------------------------

```python
def test_pfield_factory_assignment():
    class MockCheckedType:
        pass
    
    factory_func = lambda: MockCheckedType()
    serializer_func = lambda x: str(x)
    
    pfield = _PField(
        type={MockCheckedType},
        invariant=None,
        initial=None,
        mandatory=False,
        factory=factory_func,
        serializer=serializer_func
    )
    
    assert pfield._factory is factory_func
    assert pfield._factory == factory_func


# LLM-generated content at query #27
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = [int, 123]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_type_parameter_as_string():
    class MockField:
        def __init__(self):
            self.type = [int, 'CustomType']
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_initial_type():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = "not_an_int"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_callable_initial():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = lambda: 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_no_initial():
    PFIELD_NO_INITIAL = object()
    
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = "not_callable"
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = "not_callable"
            self.serializer = lambda x: str(x)
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = 123
    
    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_check_field_parameters_empty_type_list():
    class MockField:
        def __init__(self):
            self.type = []
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = MockField()
    _check_field_parameters(field)


# LLM-generated content at query #28
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    class MockBase:
        pass
    
    dct = {"field1": _PField(), "field2": "not_a_pfield"}
    bases = [MockBase()]
    name = "fields"
    
    # Check that the predicate at line 5 evaluates to True for _PField instances
    v = dct["field1"]
    assert isinstance(v, _PField) == True


# LLM-generated content at query #29
#--------------------------

```python
def test_sequence_field_invariant_parameter_evaluated_to_true():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import PVector, CheckedPVector
    
    # Create a simple checked vector class
    class MyCheckedVector(CheckedPVector):
        __type__ = int
    
    # Call _sequence_field with invariant parameter
    # The predicate at line 2 checks if invariant == PFIELD_NO_INVARIANT
    result = _sequence_field(
        checked_class=MyCheckedVector,
        item_type=int,
        optional=False,
        initial=[1, 2, 3],
        invariant=PFIELD_NO_INVARIANT
    )
    
    # Verify the field was created successfully
    assert result is not None
    assert result.type == MyCheckedVector
    assert result.mandatory is True


# LLM-generated content at query #30
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3 should evaluate to False for valid type parameters
    # "not isinstance(t, type) and not isinstance(t, str)" should be False
    # This means t should be either a type OR a string
    
    for t in field.type:
        predicate_result = not isinstance(t, type) and not isinstance(t, str)
        assert predicate_result == False


# LLM-generated content at query #31
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, 'CODE1')
    
    def invariant2(subject):
        return (True, 'CODE2')
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_one_fails():
    def invariant1(subject):
        return (True, 'CODE1')
    
    def invariant2(subject):
        return (False, 'CODE2')
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('CODE2',)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_fail():
    def invariant1(subject):
        return (False, 'ERROR1')
    
    def invariant2(subject):
        return (False, 'ERROR2')
    
    def invariant3(subject):
        return (True, 'CODE3')
    
    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('ERROR1', 'ERROR2')
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_with_complex_subject():
    def invariant1(subject):
        return (len(subject) > 0, 'EMPTY')
    
    def invariant2(subject):
        return (isinstance(subject, list), 'NOT_LIST')
    
    invariants = [invariant1, invariant2]
    subject = [1, 2, 3]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_complex_subject_fails():
    def invariant1(subject):
        return (len(subject) > 5, 'TOO_SHORT')
    
    def invariant2(subject):
        return (isinstance(subject, list), 'NOT_LIST')
    
    invariants = [invariant1, invariant2]
    subject = [1, 2]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ('TOO_SHORT',)


# LLM-generated content at query #32
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import plist, pvector
    
    # Clear the cache before testing
    _seq_field_types.clear()
    
    # Test creating a seq field type with plist and int
    item_type = int
    item_invariant = None
    
    result_type = _make_seq_field_type(plist, item_type, item_invariant)
    
    assert result_type is not None
    assert hasattr(result_type, '__type__')
    assert result_type.__type__ == int
    assert hasattr(result_type, '__invariant__')
    assert result_type.__invariant__ is None
    assert hasattr(result_type, '__reduce__')
    
    # Test that the same type is returned from cache
    cached_result = _make_seq_field_type(plist, item_type, item_invariant)
    assert cached_result is result_type
    
    # Test with pvector
    _seq_field_types.clear()
    result_type_pvector = _make_seq_field_type(pvector, str, None)
    
    assert result_type_pvector is not None
    assert result_type_pvector.__type__ == str
    assert result_type_pvector.__invariant__ is None
    
    # Test with a custom invariant function
    _seq_field_types.clear()
    def custom_invariant(val):
        return val > 0
    
    result_with_invariant = _make_seq_field_type(pvector, int, custom_invariant)
    assert result_with_invariant.__invariant__ is custom_invariant
    
    # Clean up
    _seq_field_types.clear()


# LLM-generated content at query #33
#--------------------------

```python
def test_pfield_constructor():
    def dummy_invariant(x):
        return x > 0
    
    def dummy_serializer(x):
        return str(x)
    
    def dummy_factory():
        return 42
    
    pfield = _PField(
        type=int,
        invariant=dummy_invariant,
        initial=10,
        mandatory=True,
        factory=dummy_factory,
        serializer=dummy_serializer
    )
    
    assert pfield.type == int
    assert pfield.invariant == dummy_invariant
    assert pfield.initial == 10
    assert pfield.mandatory == True
    assert pfield._factory == dummy_factory
    assert pfield.serializer == dummy_serializer


def test_pfield_constructor_with_none_values():
    pfield = _PField(
        type=str,
        invariant=None,
        initial=None,
        mandatory=False,
        factory=None,
        serializer=None
    )
    
    assert pfield.type == str
    assert pfield.invariant is None
    assert pfield.initial is None
    assert pfield.mandatory == False
    assert pfield._factory is None
    assert pfield.serializer is None


def test_pfield_constructor_with_different_types():
    pfield = _PField(
        type=list,
        invariant=lambda x: len(x) > 0,
        initial=[1, 2, 3],
        mandatory=True,
        factory=list,
        serializer=lambda x: repr(x)
    )
    
    assert pfield.type == list
    assert pfield.initial == [1, 2, 3]
    assert pfield.mandatory == True
    assert pfield._factory == list


# LLM-generated content at query #34
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    import inspect
    
    # Mock field class
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    # Mock type class
    class MockTypeClass:
        pass
    
    # Test case 1: ignore_extra is False, should return False
    def mock_factory_1():
        pass
    field_1 = MockField({MockTypeClass}, mock_factory_1)
    result_1 = is_field_ignore_extra_complaint(MockTypeClass, field_1, False)
    assert result_1 is False
    
    # Test case 2: ignore_extra is True but is_type_cls returns False (empty tuple), should return False
    field_2 = MockField((), mock_factory_1)
    result_2 = is_field_ignore_extra_complaint(MockTypeClass, field_2, True)
    assert result_2 is False
    
    # Test case 3: ignore_extra is True, is_type_cls returns True, but factory has no ignore_extra param
    field_3 = MockField({MockTypeClass}, mock_factory_1)
    result_3 = is_field_ignore_extra_complaint(MockTypeClass, field_3, True)
    assert result_3 is False
    
    # Test case 4: ignore_extra is True, is_type_cls returns True, factory has ignore_extra param
    def mock_factory_with_ignore_extra(ignore_extra=False):
        pass
    field_4 = MockField({MockTypeClass}, mock_factory_with_ignore_extra)
    result_4 = is_field_ignore_extra_complaint(MockTypeClass, field_4, True)
    assert result_4 is True
    
    # Test case 5: ignore_extra is True but type_cls check fails with string type
    field_5 = MockField({'non.existent.Class'}, mock_factory_1)
    result_5 = is_field_ignore_extra_complaint(MockTypeClass, field_5, True)
    assert result_5 is False


# LLM-generated content at query #35
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "json"
    
    result = serialize(serializer, format, value)
    
    assert result == "serialized_value"


# LLM-generated content at query #36
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import pmap, pvector
    
    # Clear cache to ensure clean state
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    checked_class = pvector
    
    result = _make_seq_field_type(checked_class, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == int
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ is None
    assert hasattr(result, '__reduce__')


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import pvector
    
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    checked_class = pvector
    
    result1 = _make_seq_field_type(checked_class, item_type, item_invariant)
    result2 = _make_seq_field_type(checked_class, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_with_different_item_types():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import pvector
    
    _seq_field_types.clear()
    
    result_int = _make_seq_field_type(pvector, int, None)
    result_str = _make_seq_field_type(pvector, str, None)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import pvector
    
    _seq_field_types.clear()
    
    def my_invariant(value):
        return value > 0
    
    result = _make_seq_field_type(pvector, int, my_invariant)
    
    assert result.__invariant__ is my_invariant


def test_make_seq_field_type_has_name():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import pvector
    
    _seq_field_types.clear()
    
    result = _make_seq_field_type(pvector, int, None)
    
    assert hasattr(result, '__name__')
    assert isinstance(result.__name__, str)
    assert len(result.__name__) > 0


def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import pvector
    
    _seq_field_types.clear()
    
    result_type = _make_seq_field_type(pvector, int, None)
    instance = result_type([1, 2, 3])
    
    reduce_result = instance.__reduce__()
    
    assert isinstance(reduce_result, tuple)
    assert len(reduce_result) == 2
    assert callable(reduce_result[0])
    assert isinstance(reduce_result[1], tuple)


# LLM-generated content at query #37
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_evaluates_to_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    import inspect
    
    class MockField:
        def __init__(self, type_val, factory_func):
            self.type = type_val
            self.factory = factory_func
    
    def sample_factory():
        pass
    
    mock_field = MockField(str, sample_factory)
    
    result = is_field_ignore_extra_complaint(int, mock_field, True)
    
    assert result is False


# LLM-generated content at query #38
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import plist, pvector
    
    # Clear the cache before testing
    _seq_field_types.clear()
    
    # Test creating a seq field type with plist
    item_type = int
    item_invariant = None
    result_type = _make_seq_field_type(plist, item_type, item_invariant)
    
    assert result_type is not None
    assert hasattr(result_type, '__type__')
    assert result_type.__type__ == int
    assert hasattr(result_type, '__invariant__')
    assert result_type.__invariant__ is None
    assert hasattr(result_type, '__reduce__')
    
    # Test that the same call returns the cached type
    result_type_2 = _make_seq_field_type(plist, item_type, item_invariant)
    assert result_type is result_type_2
    
    # Test with a different item type
    _seq_field_types.clear()
    result_type_3 = _make_seq_field_type(pvector, str, None)
    assert result_type_3 is not None
    assert result_type_3.__type__ == str
    
    # Test that different item types create different types
    result_type_4 = _make_seq_field_type(pvector, int, None)
    assert result_type_3 is not result_type_4
    
    # Test with pvector
    _seq_field_types.clear()
    result_type_5 = _make_seq_field_type(pvector, int, None)
    assert result_type_5 is not None
    assert result_type_5.__type__ == int
    assert issubclass(result_type_5, pvector)


# LLM-generated content at query #39
#--------------------------

```python
def test_sequence_field_invariant_parameter_is_pfield_no_invariant_by_default():
    from pyrsistent._field_common import PFIELD_NO_INVARIANT
    
    invariant_value = PFIELD_NO_INVARIANT
    
    assert invariant_value is PFIELD_NO_INVARIANT


# LLM-generated content at query #40
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [str, int]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: "not isinstance(t, type) and not isinstance(t, str)"
    # For this to be False, we need: isinstance(t, type) OR isinstance(t, str)
    # This test verifies the predicate evaluates to False for valid type parameters
    
    for t in field.type:
        predicate_result = not isinstance(t, type) and not isinstance(t, str)
        assert predicate_result is False


# LLM-generated content at query #41
#--------------------------

```python
def test_pmap_field_creates_field_with_checked_pmap_type():
    from pyrsistent._field_common import pmap_field, field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    
    assert result.mandatory == True
    assert result.initial == pmap()
    assert result.factory is not None
    assert callable(result.factory)


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    
    assert result.mandatory == True
    assert result.initial == pmap()
    assert result.factory is not None
    assert callable(result.factory)


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    
    assert result.mandatory == True
    assert result.initial == pmap()
    assert result.factory is not None
    assert callable(result.factory)


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(x):
        return True, None
    
    result = pmap_field(str, int, invariant=my_invariant)
    
    assert result.mandatory == True
    assert result.invariant is not None
    assert callable(result.invariant)


def test_pmap_field_factory_with_optional_none_argument():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    
    factory_result = result.factory(None)
    assert factory_result is None


def test_pmap_field_factory_with_optional_dict_argument():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    
    factory_result = result.factory({'key': 1})
    assert factory_result is not None
    assert factory_result['key'] == 1


def test_pmap_field_factory_without_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    
    factory_result = result.factory({'key': 1})
    assert factory_result is not None
    assert factory_result['key'] == 1


def test_pmap_field_returns_field_object():
    from pyrsistent._field_common import pmap_field, _PField
    
    result = pmap_field(str, int)
    
    assert isinstance(result, _PField)


def test_pmap_field_caching_same_type_combination():
    from pyrsistent._field_common import pmap_field
    
    result1 = pmap_field(str, int)
    result2 = pmap_field(str, int)
    
    assert type(result1.initial).__name__ == type(result2.initial).__name__


def test_pmap_field_different_type_combinations():
    from pyrsistent._field_common import pmap_field
    
    result1 = pmap_field(str, int)
    result2 = pmap_field(int, str)
    
    assert type(result1.initial).__name__ != type(result2.initial).__name__


# LLM-generated content at query #42
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure fresh test
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = lambda x: x > 0
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert result.__type__ == int
    assert result.__invariant__ == item_invariant
    assert issubclass(result, PVector)


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = lambda x: len(x) > 0
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_different_item_types_different_results():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    invariant = lambda x: True
    
    result_int = _make_seq_field_type(PVector, int, invariant)
    result_str = _make_seq_field_type(PVector, str, invariant)
    
    assert result_int is not result_str
    assert result_int.__type__ == int
    assert result_str.__type__ == str


def test_make_seq_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, lambda x: True)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_seq_field_type_name_generation():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, lambda x: True)
    
    assert result.__name__ is not None
    assert len(result.__name__) > 0
    assert isinstance(result.__name__, str)


# LLM-generated content at query #43
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pmap, pvec, PClass, field
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class TestClass(PClass):
        items = field()
    
    item_type = int
    data = [1, 2, 3]
    
    # Register a type in _seq_field_types for testing
    from pyrsistent import pvector_field
    test_field_type = pvector_field(int)
    _seq_field_types[TestClass, item_type] = test_field_type
    
    result = _restore_seq_field_pickle(TestClass, item_type, data)
    
    assert result is not None
    assert len(result) == 3


def test_restore_seq_field_pickle_with_strings():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class TestClass(PClass):
        items = field()
    
    item_type = str
    data = ['a', 'b', 'c']
    
    from pyrsistent import pvector_field
    test_field_type = pvector_field(str)
    _seq_field_types[TestClass, item_type] = test_field_type
    
    result = _restore_seq_field_pickle(TestClass, item_type, data)
    
    assert result is not None
    assert len(result) == 3


def test_restore_seq_field_pickle_empty_data():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class TestClass(PClass):
        items = field()
    
    item_type = int
    data = []
    
    from pyrsistent import pvector_field
    test_field_type = pvector_field(int)
    _seq_field_types[TestClass, item_type] = test_field_type
    
    result = _restore_seq_field_pickle(TestClass, item_type, data)
    
    assert result is not None
    assert len(result) == 0


# LLM-generated content at query #44
#--------------------------

```python
def test_check_field_parameters_valid_type_as_type():
    class Field:
        type = [int, str]
        initial = 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_valid_type_as_string():
    class Field:
        type = ['int', 'str']
        initial = 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class Field:
        type = [int, 123]
        initial = 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Type parameter expected' in str(e)


def test_check_field_parameters_initial_no_initial():
    PFIELD_NO_INITIAL = object()
    class Field:
        type = [int]
        initial = PFIELD_NO_INITIAL
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_initial_callable():
    class Field:
        type = [int]
        initial = lambda: 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_initial_valid_type():
    class Field:
        type = [int, str]
        initial = "hello"
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_initial_invalid_type():
    PFIELD_NO_INITIAL = object()
    class Field:
        type = [int, str]
        initial = 3.14
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Initial has invalid type' in str(e)


def test_check_field_parameters_invariant_not_callable():
    class Field:
        type = [int]
        initial = 5
        invariant = "not callable"
        factory = lambda: None
        serializer = lambda x: x
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariant must be callable' in str(e)


def test_check_field_parameters_factory_not_callable():
    class Field:
        type = [int]
        initial = 5
        invariant = lambda x: True
        factory = "not callable"
        serializer = lambda x: x
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Factory must be callable' in str(e)


def test_check_field_parameters_serializer_not_callable():
    class Field:
        type = [int]
        initial = 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = "not callable"
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Serializer must be callable' in str(e)


def test_check_field_parameters_empty_type_list():
    class Field:
        type = []
        initial = 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x
    
    field = Field()
    _check_field_parameters(field)


# LLM-generated content at query #45
#--------------------------

```python
def test_set_fields_basic():
    from types import SimpleNamespace
    
    class MockField:
        def __init__(self, value):
            self.value = value
    
    class _PField:
        pass
    
    # Create a simple _PField subclass for testing
    class TestField(_PField):
        def __init__(self, name):
            self.name = name
    
    # Test with empty bases
    dct = {}
    bases = []
    name = "fields"
    set_fields(dct, bases, name)
    assert name in dct
    assert dct[name] == {}

def test_set_fields_with_parent_fields():
    class _PField:
        pass
    
    class TestField(_PField):
        pass
    
    # Create parent class with fields
    parent = type('Parent', (), {})
    parent.__dict__ = {'fields': {'parent_field': TestField()}}
    
    dct = {}
    bases = [parent]
    name = "fields"
    set_fields(dct, bases, name)
    assert name in dct
    assert 'parent_field' in dct[name]

def test_set_fields_moves_pfield_from_dct():
    class _PField:
        pass
    
    field_obj = _PField()
    dct = {'test_field': field_obj, 'other_value': 123}
    bases = []
    name = "fields"
    set_fields(dct, bases, name)
    
    assert name in dct
    assert 'test_field' in dct[name]
    assert dct[name]['test_field'] is field_obj
    assert 'test_field' not in dct
    assert 'other_value' in dct

def test_set_fields_multiple_bases():
    class _PField:
        pass
    
    class TestField(_PField):
        pass
    
    parent1 = type('Parent1', (), {})
    parent1.__dict__ = {'fields': {'field1': TestField()}}
    
    parent2 = type('Parent2', (), {})
    parent2.__dict__ = {'fields': {'field2': TestField()}}
    
    dct = {}
    bases = [parent1, parent2]
    name = "fields"
    set_fields(dct, bases, name)
    
    assert name in dct
    assert 'field1' in dct[name]
    assert 'field2' in dct[name]

def test_set_fields_mixed_content():
    class _PField:
        pass
    
    field_obj = _PField()
    dct = {'pfield': field_obj, 'regular': 'value', 'number': 42}
    bases = []
    name = "fields"
    set_fields(dct, bases, name)
    
    assert name in dct
    assert 'pfield' in dct[name]
    assert 'regular' in dct
    assert 'number' in dct
    assert dct['regular'] == 'value'
    assert dct['number'] == 42


# LLM-generated content at query #46
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    def passing_invariant(subject):
        return (True, "ERROR_CODE_2")
    
    invariants = [failing_invariant, passing_invariant]
    subject = "test_subject"
    
    error_codes = tuple(error_code for is_ok, error_code in
                        (invariant(subject) for invariant in invariants) if not is_ok)
    
    assert error_codes == ("ERROR_CODE_1",)
    assert bool(error_codes) is True


# LLM-generated content at query #47
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant_pass(subject):
        return (True, None)
    
    def invariant_pass2(subject):
        return (True, None)
    
    invariants = [invariant_pass, invariant_pass2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #48
#--------------------------

```python
def test_make_pmap_field_type_creates_new_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__name__ == "StrToIntPMap"
    assert result2.__name__ == "IntToStrPMap"


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_with_multiple_key_value_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type((str, int), (float, bool))
    
    assert "StrInt" in result.__name__
    assert "FloatBool" in result.__name__
    assert "PMap" in result.__name__


# LLM-generated content at query #49
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (which is `optional` parameter) should evaluate to False
    # This means the else branch at line 21-22 should be executed
    # We verify this by checking that factory is set to TheMap.create
    assert result.factory is not None
    assert callable(result.factory)
    
    # Test that the factory works correctly for non-optional case
    test_map = result.factory({"a": 1, "b": 2})
    assert test_map is not None


# LLM-generated content at query #50
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_sets_name():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert "PMap" in result.__name__
    assert "To" in result.__name__


def test_make_pmap_field_type_different_types_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(getattr(result, '__reduce__'))


# LLM-generated content at query #51
#--------------------------

```python
def test_check_global_invariants_no_errors():
    def invariant1(subject):
        return (True, "no_error_1")
    
    def invariant2(subject):
        return (True, "no_error_2")
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    result = check_global_invariants(subject, invariants)
    assert result is None


# LLM-generated content at query #52
#--------------------------

```python
def test_check_global_invariants_all_pass():
    invariants = [
        lambda x: (True, 'code1'),
        lambda x: (True, 'code2'),
    ]
    subject = "test_subject"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_failure():
    invariants = [
        lambda x: (True, 'code1'),
        lambda x: (False, 'error_code1'),
    ]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('error_code1',)
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    invariants = [
        lambda x: (False, 'error_code1'),
        lambda x: (False, 'error_code2'),
        lambda x: (True, 'code3'),
    ]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('error_code1', 'error_code2')
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    check_global_invariants(subject, invariants)


def test_check_global_invariants_with_complex_subject():
    subject = {'key': 'value', 'nested': {'data': 123}}
    invariants = [
        lambda x: (True, 'valid_structure'),
        lambda x: (True, 'valid_content'),
    ]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #53
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, 'OK1')
    
    def invariant2(subject):
        return (True, 'OK2')
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
    except InvariantException:
        assert False, "Should not raise exception when all invariants pass"


def test_check_global_invariants_single_failure():
    def invariant1(subject):
        return (False, 'ERROR1')
    
    def invariant2(subject):
        return (True, 'OK2')
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('ERROR1',)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    def invariant1(subject):
        return (False, 'ERROR1')
    
    def invariant2(subject):
        return (False, 'ERROR2')
    
    def invariant3(subject):
        return (True, 'OK3')
    
    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('ERROR1', 'ERROR2')
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


def test_check_global_invariants_all_fail():
    def invariant1(subject):
        return (False, 'ERROR1')
    
    def invariant2(subject):
        return (False, 'ERROR2')
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert e.args[0] == ('ERROR1', 'ERROR2')


def test_check_global_invariants_empty_invariants():
    invariants = []
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
    except InvariantException:
        assert False, "Should not raise exception when no invariants fail"


def test_check_global_invariants_with_none_subject():
    def invariant1(subject):
        return (True, 'OK1')
    
    invariants = [invariant1]
    subject = None
    
    try:
        check_global_invariants(subject, invariants)
    except InvariantException:
        assert False, "Should not raise exception when all invariants pass"


# LLM-generated content at query #54
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, field, CheckedPVector
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class MyCheckedVector(CheckedPVector):
        x = field(type=int)
    
    checked_class = MyCheckedVector
    item_type = int
    data = [1, 2, 3]
    
    _seq_field_types[checked_class, item_type] = type('TestSeqField', (), {
        'create': lambda self, d, _factory_fields=None: PVector(d)
    })()
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result == PVector([1, 2, 3])


# LLM-generated content at query #55
#--------------------------

```python
def test_make_pmap_field_type_creates_checked_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__key_type__ == str
    assert result.__value_type__ == int
    assert "PMap" in result.__name__


def test_make_pmap_field_type_caches_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__key_type__ == str
    assert result1.__value_type__ == int
    assert result2.__key_type__ == int
    assert result2.__value_type__ == str


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert hasattr(result, '__reduce__')
    assert callable(result.__reduce__)


def test_make_pmap_field_type_name_format():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert "To" in result.__name__
    assert result.__name__.endswith("PMap")


# LLM-generated content at query #56
#--------------------------

```python
def test_sequence_field_invariant_parameter_default():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import PVector, CheckedPVector
    
    # Create a simple checked vector type
    class MyCheckedVector(CheckedPVector):
        __type__ = int
    
    # Call _sequence_field with default invariant parameter
    # This tests that line 2's default parameter invariant=PFIELD_NO_INVARIANT evaluates to True
    # by verifying the function can be called without providing the invariant argument
    result = _sequence_field(
        checked_class=MyCheckedVector,
        item_type=int,
        optional=False,
        initial=[]
    )
    
    # Verify that result is a field object with the default invariant
    assert result.invariant == PFIELD_NO_INVARIANT


# LLM-generated content at query #57
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert result is not None
    assert result.mandatory is True
    assert result.initial is not None


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    assert result is not None
    assert result.mandatory is True
    assert result.initial is not None


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=False)
    assert result is not None
    assert result.mandatory is True
    assert result.initial is not None


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    def dummy_invariant(val):
        return True, None
    result = pmap_field(str, int, invariant=dummy_invariant)
    assert result is not None
    assert result.mandatory is True
    assert result.invariant is not None


def test_pmap_field_factory_callable():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert callable(result.factory)


def test_pmap_field_factory_with_optional_is_callable():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    assert callable(result.factory)


def test_pmap_field_type_set():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert isinstance(result.type, set)
    assert len(result.type) > 0


def test_pmap_field_serializer_callable():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert callable(result.serializer)


# LLM-generated content at query #58
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    key_type = str
    value_type = int
    data = {'a': 1, 'b': 2}
    
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    
    assert result == pmap(data)
    assert isinstance(result, type(_pmap_field_types[key_type, value_type]))


def test_restore_pmap_field_pickle_empty_data():
    from pyrsistent._field_common import _restore_pmap_field_pickle
    from pyrsistent import pmap
    
    key_type = str
    value_type = int
    data = {}
    
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    
    assert result == pmap({})
    assert len(result) == 0


def test_restore_pmap_field_pickle_different_types():
    from pyrsistent._field_common import _restore_pmap_field_pickle
    from pyrsistent import pmap
    
    key_type = int
    value_type = str
    data = {1: 'one', 2: 'two', 3: 'three'}
    
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    
    assert result == pmap(data)
    assert result[1] == 'one'
    assert result[2] == 'two'
    assert result[3] == 'three'


# LLM-generated content at query #59
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap_field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (which checks `if optional:`) should evaluate to False
    # This means the factory should be set to TheMap.create directly (line 22)
    # and the type should be TheMap, not optional_type(TheMap)
    
    # Verify that when optional=False, the factory is a direct reference to create method
    # and not a wrapper function
    assert result._factory is not None
    assert callable(result._factory)
    
    # Verify that the type is a single CheckedPMap type, not wrapped with optional
    assert len(result.type) == 1


# LLM-generated content at query #60
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap, PMap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register the type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent._field_common import _make_pmap_field
        _make_pmap_field(key_type, value_type)
    
    # Create test data
    test_data = pmap({'a': 1, 'b': 2})
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result
    assert result == test_data
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #61
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()
    assert callable(result.factory)


def test_pmap_field_optional():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()
    assert callable(result.factory)
    assert result.factory(None) is None


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(value):
        return True, ()
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.type is not None
    assert result.mandatory is True
    assert result.invariant is not None
    assert callable(result.factory)


def test_pmap_field_factory_non_optional():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap, CheckedPMap
    
    result = pmap_field(str, int, optional=False)
    assert callable(result.factory)
    test_data = {'a': 1, 'b': 2}
    created = result.factory(test_data)
    assert isinstance(created, CheckedPMap)


def test_pmap_field_factory_optional_with_none():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    assert result.factory(None) is None


def test_pmap_field_factory_optional_with_data():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import CheckedPMap
    
    result = pmap_field(str, int, optional=True)
    test_data = {'x': 10}
    created = result.factory(test_data)
    assert isinstance(created, CheckedPMap)


# LLM-generated content at query #62
#--------------------------

```python
def test_pmap_field_returns_field_with_correct_type():
    from pyrsistent._field_common import pmap_field, field
    from pyrsistent import PMap
    
    result = pmap_field(str, int, optional=False)
    
    assert isinstance(result, type(field(mandatory=True, initial=PMap(), type=PMap)))
    assert result.mandatory is True
    assert result.type is not None


# LLM-generated content at query #63
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap_field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (which is the docstring) should evaluate to False
    # This means optional=False, so the condition `if optional:` at line 15 should be False
    # We verify this by checking that the factory is set to TheMap.create directly
    # and not wrapped in a lambda that checks for None
    
    assert result.factory is not None
    assert result.initial == pmap()
    assert result.mandatory is True


# LLM-generated content at query #64
#--------------------------

```python
def test_sequence_field_invariant_parameter_is_pfield_no_invariant():
    from pyrsistent._field_common import PFIELD_NO_INVARIANT
    
    invariant = PFIELD_NO_INVARIANT
    
    assert invariant == PFIELD_NO_INVARIANT


# LLM-generated content at query #65
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    checked_value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    
    assert result == "serialized_value"


# LLM-generated content at query #66
#--------------------------

```python
def test_set_fields():
    from collections import namedtuple
    
    class _PField:
        def __init__(self, value):
            self.value = value
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    # Test case 1: Basic functionality with no bases
    dct1 = {'field1': _PField('value1'), 'field2': _PField('value2')}
    set_fields(dct1, [], 'fields')
    assert 'fields' in dct1
    assert 'field1' in dct1['fields']
    assert 'field2' in dct1['fields']
    assert 'field1' not in dct1
    assert 'field2' not in dct1
    
    # Test case 2: With base classes
    class Base:
        pass
    Base.__dict__ = {'fields': {'base_field': _PField('base_value')}}
    dct2 = {'field1': _PField('value1')}
    set_fields(dct2, [Base], 'fields')
    assert 'fields' in dct2
    assert 'base_field' in dct2['fields']
    assert 'field1' in dct2['fields']
    assert 'field1' not in dct2
    
    # Test case 3: Mixed PField and non-PField items
    dct3 = {'pfield': _PField('pvalue'), 'regular': 'regular_value', 'pfield2': _PField('pvalue2')}
    set_fields(dct3, [], 'fields')
    assert 'fields' in dct3
    assert 'pfield' in dct3['fields']
    assert 'pfield2' in dct3['fields']
    assert 'regular' in dct3
    assert 'pfield' not in dct3
    assert 'pfield2' not in dct3
    
    # Test case 4: Empty dictionary
    dct4 = {}
    set_fields(dct4, [], 'fields')
    assert 'fields' in dct4
    assert dct4['fields'] == {}
    
    # Test case 5: Multiple bases with overlapping fields
    class Base1:
        pass
    class Base2:
        pass
    Base1.__dict__ = {'fields': {'base1_field': _PField('base1_value')}}
    Base2.__dict__ = {'fields': {'base2_field': _PField('base2_value')}}
    dct5 = {'child_field': _PField('child_value')}
    set_fields(dct5, [Base1, Base2], 'fields')
    assert 'fields' in dct5
    assert 'base1_field' in dct5['fields']
    assert 'base2_field' in dct5['fields']
    assert 'child_field' in dct5['fields']


# LLM-generated content at query #67
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_001")
    
    def passing_invariant(subject):
        return (True, "")
    
    class InvariantException(Exception):
        def __init__(self, error_codes, arg2, message):
            self.error_codes = error_codes
            self.message = message
            super().__init__(message)
    
    def check_global_invariants(subject, invariants):
        error_codes = tuple(error_code for is_ok, error_code in
                            (invariant(subject) for invariant in invariants) if not is_ok)
        if error_codes:
            raise InvariantException(error_codes, (), 'Global invariant failed')
    
    subject = {}
    invariants = [failing_invariant, passing_invariant]
    
    exception_raised = False
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        exception_raised = True
        assert e.error_codes == ("ERROR_001",)
        assert e.message == 'Global invariant failed'
    
    assert exception_raised


# LLM-generated content at query #68
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: not isinstance(t, type) and not isinstance(t, str)
    # For this to evaluate to False, we need:
    # - isinstance(t, type) to be True, OR
    # - isinstance(t, str) to be True
    
    # Test with a type object (isinstance(t, type) is True)
    t = int
    predicate_result = not isinstance(t, type) and not isinstance(t, str)
    assert predicate_result is False
    
    # Test with a string (isinstance(t, str) is True)
    t = "int"
    predicate_result = not isinstance(t, type) and not isinstance(t, str)
    assert predicate_result is False


# LLM-generated content at query #69
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent import pmap, PMap
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent._checked_types import CheckedPMap
    
    # Create a PMap field type
    key_type = str
    value_type = int
    
    # Register the type if not already registered
    if (key_type, value_type) not in _pmap_field_types:
        pmap_field_type = CheckedPMap.create(key_type, value_type)
        _pmap_field_types[key_type, value_type] = pmap_field_type
    
    # Create test data
    test_data = {'a': 1, 'b': 2, 'c': 3}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a PMap with correct data
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert len(result) == 3


# LLM-generated content at query #70
#--------------------------

```python
def test_check_global_invariants_with_all_passing_invariants():
    def invariant1(subject):
        return (True, None)
    
    def invariant2(subject):
        return (True, None)
    
    invariants = [invariant1, invariant2]
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, invariants)
        result = True
    except InvariantException:
        result = False
    
    assert result is True


# LLM-generated content at query #71
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(value):
        return (True, "valid")
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.type is not None
    assert result.mandatory is True
    assert result.invariant is not None


def test_pmap_field_factory_without_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    assert callable(result.factory)


def test_pmap_field_factory_with_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    assert callable(result.factory)


def test_pmap_field_factory_none_with_optional():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=True)
    factory_result = result.factory(None)
    assert factory_result is None


def test_pmap_field_factory_creates_pmap():
    from pyrsistent._field_common import pmap_field
    
    result = pmap_field(str, int, optional=False)
    factory_result = result.factory({'a': 1, 'b': 2})
    assert factory_result['a'] == 1
    assert factory_result['b'] == 2


def test_pmap_field_multiple_calls_same_types():
    from pyrsistent._field_common import pmap_field
    
    result1 = pmap_field(str, int)
    result2 = pmap_field(str, int)
    assert result1.type == result2.type


def test_pmap_field_different_types():
    from pyrsistent._field_common import pmap_field
    
    result1 = pmap_field(str, int)
    result2 = pmap_field(int, str)
    assert result1.type != result2.type


# LLM-generated content at query #72
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    
    class MockField:
        def __init__(self, type_value, factory_func):
            self.type = type_value
            self.factory = factory_func
    
    def simple_factory():
        return "test"
    
    class TestType:
        pass
    
    field = MockField(int, simple_factory)
    
    result = is_field_ignore_extra_complaint(TestType, field, True)
    
    assert result is False


# LLM-generated content at query #73
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pmap, pvec
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    # Create a simple checked class with a pvec field
    class MyCheckedClass(CheckedPVector):
        __type__ = int
    
    # Register the type in _seq_field_types
    item_type = int
    _seq_field_types[MyCheckedClass, item_type] = MyCheckedClass
    
    # Create test data
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(MyCheckedClass, item_type, test_data)
    
    # Verify the result is an instance of the checked class
    assert isinstance(result, MyCheckedClass)
    assert list(result) == test_data


def test_restore_seq_field_pickle_empty():
    from pyrsistent import pvec
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    class EmptyCheckedClass(CheckedPVector):
        __type__ = str
    
    item_type = str
    _seq_field_types[EmptyCheckedClass, item_type] = EmptyCheckedClass
    
    test_data = []
    
    result = _restore_seq_field_pickle(EmptyCheckedClass, item_type, test_data)
    
    assert isinstance(result, EmptyCheckedClass)
    assert len(result) == 0


def test_restore_seq_field_pickle_with_strings():
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    class StringCheckedClass(CheckedPVector):
        __type__ = str
    
    item_type = str
    _seq_field_types[StringCheckedClass, item_type] = StringCheckedClass
    
    test_data = ["a", "b", "c"]
    
    result = _restore_seq_field_pickle(StringCheckedClass, item_type, test_data)
    
    assert isinstance(result, StringCheckedClass)
    assert list(result) == test_data


# LLM-generated content at query #74
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert result is not None
    assert hasattr(result, '__key_type__')
    assert hasattr(result, '__value_type__')
    assert result.__key_type__ == str
    assert result.__value_type__ == int


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    assert result1 is result2


def test_make_pmap_field_type_generates_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert "PMap" in result.__name__
    assert "To" in result.__name__


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    assert hasattr(result, '__reduce__')
    assert callable(getattr(result, '__reduce__'))


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    assert result1 is not result2


def test_make_pmap_field_type_with_float_and_bool():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(float, bool)
    assert result.__key_type__ == float
    assert result.__value_type__ == bool
    assert result is _make_pmap_field_type(float, bool)


# LLM-generated content at query #75
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    # Test that when optional=False, the predicate at line 2 evaluates to False
    # Line 2 refers to the condition `if optional:` at line 15
    optional = False
    assert not optional


# LLM-generated content at query #76
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache to ensure a fresh test
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert issubclass(result, PVector)
    assert result.__type__ == int
    assert result.__invariant__ is None
    assert hasattr(result, '__reduce__')


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache to ensure a fresh test
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2
    assert (PVector, item_type) in _seq_field_types


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache to ensure a fresh test
    _seq_field_types.clear()
    
    item_type = float
    def custom_invariant(value):
        return value > 0
    
    result = _make_seq_field_type(PVector, item_type, custom_invariant)
    
    assert result.__type__ == float
    assert result.__invariant__ is custom_invariant


def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache to ensure a fresh test
    _seq_field_types.clear()
    
    item_type = int
    result_type = _make_seq_field_type(PVector, item_type, None)
    
    # Create an instance and test __reduce__
    instance = result_type([1, 2, 3])
    reduce_result = instance.__reduce__()
    
    assert reduce_result[0].__name__ == '_restore_seq_field_pickle'
    assert reduce_result[1][0] is PVector
    assert reduce_result[1][1] == int
    assert reduce_result[1][2] == [1, 2, 3]


def test_make_seq_field_type_name_generation():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache to ensure a fresh test
    _seq_field_types.clear()
    
    item_type = int
    result = _make_seq_field_type(PVector, item_type, None)
    
    assert hasattr(result, '__name__')
    assert isinstance(result.__name__, str)
    assert len(result.__name__) > 0


# LLM-generated content at query #77
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import PVector, pmap
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    from pyrsistent._checked_types import CheckedPVector
    
    # Create a checked vector type
    checked_vec_type = CheckedPVector(int)
    _seq_field_types[(type(None), int)] = checked_vec_type
    
    # Test data to restore
    test_data = [1, 2, 3]
    
    # Call the function
    result = _restore_seq_field_pickle(type(None), int, test_data)
    
    # Verify the result is a PVector with the correct data
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #78
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_1")
    
    def passing_invariant(subject):
        return (True, "")
    
    subject = "test_subject"
    invariants = [failing_invariant, passing_invariant]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #79
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_value"


# LLM-generated content at query #80
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class Field:
        def __init__(self):
            self.type = [str, int]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_invalid_type_parameter():
    class Field:
        def __init__(self):
            self.type = [str, 123]
            self.initial = "PFIELD_NO_INITIAL"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type parameter expected" in str(e)


def test_check_field_parameters_invalid_initial_type():
    PFIELD_NO_INITIAL = object()
    
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = 123
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


def test_check_field_parameters_callable_initial():
    PFIELD_NO_INITIAL = object()
    
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = lambda: "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_non_callable_invariant():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = "not_callable"
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Invariant must be callable" in str(e)


def test_check_field_parameters_non_callable_factory():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = "not_callable"
            self.serializer = lambda x: str(x)
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Factory must be callable" in str(e)


def test_check_field_parameters_non_callable_serializer():
    class Field:
        def __init__(self):
            self.type = [str]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not_callable"
    
    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Serializer must be callable" in str(e)


def test_check_field_parameters_string_type():
    class Field:
        def __init__(self):
            self.type = [str, "CustomType"]
            self.initial = "test"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


def test_check_field_parameters_empty_type_list():
    PFIELD_NO_INITIAL = object()
    
    class Field:
        def __init__(self):
            self.type = []
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)
    
    field = Field()
    _check_field_parameters(field)


# LLM-generated content at query #81
#--------------------------

```python
def test_make_seq_field_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector, pv
    
    # Clear the cache before testing
    _seq_field_types.clear()
    
    # Test creating a seq field type with int item type
    item_type = int
    item_invariant = None
    
    result_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    # Verify the result is a class
    assert isinstance(result_type, type)
    
    # Verify it's a subclass of PVector
    assert issubclass(result_type, PVector)
    
    # Verify the __type__ attribute is set correctly
    assert result_type.__type__ == int
    
    # Verify the __invariant__ attribute is set correctly
    assert result_type.__invariant__ is None
    
    # Verify the type is cached
    cached_type = _seq_field_types.get((PVector, int))
    assert cached_type is result_type
    
    # Verify calling again with same parameters returns cached instance
    result_type_2 = _make_seq_field_type(PVector, item_type, item_invariant)
    assert result_type_2 is result_type
    
    # Verify __reduce__ method exists
    assert hasattr(result_type, '__reduce__')
    
    # Test with a different item type
    _seq_field_types.clear()
    result_type_str = _make_seq_field_type(PVector, str, None)
    assert result_type_str.__type__ == str
    assert result_type_str is not result_type
    
    # Verify the name is set (should contain type names and suffix)
    assert hasattr(result_type_str, '__name__')
    assert isinstance(result_type_str.__name__, str)
    assert len(result_type_str.__name__) > 0


# LLM-generated content at query #82
#--------------------------

```python
def test_predicate_isinstance_pfield_evaluates_to_true():
    class _PField:
        pass
    
    dct = {}
    bases = []
    name = "fields"
    
    pfield_instance = _PField()
    dct["test_key"] = pfield_instance
    
    for k, v in list(dct.items()):
        assert isinstance(v, _PField)


# LLM-generated content at query #83
#--------------------------

```python
def test_sequence_field_invariant_parameter_is_pfield_no_invariant():
    from pyrsistent._field_common import PFIELD_NO_INVARIANT
    
    invariant = PFIELD_NO_INVARIANT
    result = invariant == PFIELD_NO_INVARIANT
    
    assert result is True


# LLM-generated content at query #84
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    
    class MockField:
        def __init__(self, factory, type_):
            self.factory = factory
            self.type = type_
    
    def mock_factory():
        pass
    
    field = MockField(mock_factory, {int})
    result = is_field_ignore_extra_complaint(object, field, False)
    assert result is False


def test_is_field_ignore_extra_complaint_returns_false_when_type_cls_mismatch():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, factory, type_):
            self.factory = factory
            self.type = type_
    
    def mock_factory():
        pass
    
    field = MockField(mock_factory, (str,))
    result = is_field_ignore_extra_complaint(int, field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_returns_false_when_no_ignore_extra_param():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, factory, type_):
            self.factory = factory
            self.type = type_
    
    def mock_factory():
        pass
    
    field = MockField(mock_factory, (object,))
    result = is_field_ignore_extra_complaint(object, field, True)
    assert result is False


def test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_met():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, factory, type_):
            self.factory = factory
            self.type = type_
    
    def mock_factory(ignore_extra=False):
        pass
    
    field = MockField(mock_factory, (object,))
    result = is_field_ignore_extra_complaint(object, field, True)
    assert result is True


def test_is_field_ignore_extra_complaint_with_set_type():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, factory, type_):
            self.factory = factory
            self.type = type_
    
    def mock_factory(ignore_extra=False):
        pass
    
    field = MockField(mock_factory, {int, str})
    result = is_field_ignore_extra_complaint(object, field, True)
    assert result is True


def test_is_field_ignore_extra_complaint_with_empty_tuple():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    
    class MockField:
        def __init__(self, factory, type_):
            self.factory = factory
            self.type = type_
    
    def mock_factory(ignore_extra=False):
        pass
    
    field = MockField(mock_factory, ())
    result = is_field_ignore_extra_complaint(object, field, True)
    assert result is False


# LLM-generated content at query #85
#--------------------------

```python
def test_check_global_invariants_all_pass():
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    check_global_invariants("test_subject", invariants)


def test_check_global_invariants_single_failure():
    invariants = [
        lambda x: (True, None),
        lambda x: (False, "ERROR_CODE_1"),
        lambda x: (True, None),
    ]
    try:
        check_global_invariants("test_subject", invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1",)


def test_check_global_invariants_multiple_failures():
    invariants = [
        lambda x: (False, "ERROR_CODE_1"),
        lambda x: (False, "ERROR_CODE_2"),
        lambda x: (True, None),
        lambda x: (False, "ERROR_CODE_3"),
    ]
    try:
        check_global_invariants("test_subject", invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_CODE_1", "ERROR_CODE_2", "ERROR_CODE_3")


def test_check_global_invariants_empty_invariants():
    invariants = []
    check_global_invariants("test_subject", invariants)


def test_check_global_invariants_all_fail():
    invariants = [
        lambda x: (False, "ERROR_A"),
        lambda x: (False, "ERROR_B"),
    ]
    try:
        check_global_invariants("test_subject", invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR_A", "ERROR_B")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #86
#--------------------------

```python
def test_make_pmap_field_type():
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    
    # Clear the cache before testing
    _pmap_field_types.clear()
    
    # Test creating a pmap field type with basic types
    pmap_type1 = _make_pmap_field_type(str, int)
    assert pmap_type1 is not None
    assert "PMap" in pmap_type1.__name__
    assert pmap_type1.__key_type__ == str
    assert pmap_type1.__value_type__ == int
    
    # Test that the same key/value type combination returns the cached type
    pmap_type1_again = _make_pmap_field_type(str, int)
    assert pmap_type1_again is pmap_type1
    
    # Test with different types
    pmap_type2 = _make_pmap_field_type(int, str)
    assert pmap_type2 is not pmap_type1
    assert pmap_type2.__key_type__ == int
    assert pmap_type2.__value_type__ == str
    
    # Test that __reduce__ method exists
    assert hasattr(pmap_type1, '__reduce__')
    
    # Test name generation
    assert "StrToIntPMap" == pmap_type1.__name__
    assert "IntToStrPMap" == pmap_type2.__name__
    
    _pmap_field_types.clear()


# LLM-generated content at query #87
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent import pmap, field, PClass
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    
    # Create a PClass with a pmap field to register the type
    class TestClass(PClass):
        my_map = field(factory=pmap, initial={})
    
    # Get the registered type
    key_type = str
    value_type = int
    test_data = {'key1': 1, 'key2': 2}
    
    # Create an instance and pickle/unpickle it
    instance = TestClass(my_map=pmap(test_data))
    
    # Simulate what pickle does - extract the pmap data
    pmap_data = instance.my_map
    
    # Test the restore function
    restored = _restore_pmap_field_pickle(key_type, value_type, dict(pmap_data))
    
    assert restored is not None
    assert len(restored) == len(test_data)


def test_restore_pmap_field_pickle_empty():
    from pyrsistent import pmap, field, PClass
    from pyrsistent._field_common import _restore_pmap_field_pickle
    
    # Test with empty pmap
    key_type = str
    value_type = int
    empty_data = {}
    
    restored = _restore_pmap_field_pickle(key_type, value_type, empty_data)
    
    assert restored is not None
    assert len(restored) == 0


def test_restore_pmap_field_pickle_with_data():
    from pyrsistent import pmap, field, PClass
    from pyrsistent._field_common import _restore_pmap_field_pickle
    
    # Test with specific data
    key_type = str
    value_type = str
    test_data = {'name': 'Alice', 'city': 'NYC'}
    
    restored = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    assert restored is not None
    assert restored['name'] == 'Alice'
    assert restored['city'] == 'NYC'


# LLM-generated content at query #88
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    
    assert result is not None
    assert result.type is not None


# LLM-generated content at query #89
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int)
    assert result is not None
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_with_optional_true():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    assert result is not None
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_with_optional_false():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    assert result is not None
    assert result.type is not None
    assert result.mandatory is True
    assert result.initial == pmap()


def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    def my_invariant(value):
        return (True, None)
    
    result = pmap_field(str, int, invariant=my_invariant)
    assert result is not None
    assert result.invariant is not None
    assert result.mandatory is True


def test_pmap_field_factory_without_optional():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=False)
    assert callable(result.factory)


def test_pmap_field_factory_with_optional():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    result = pmap_field(str, int, optional=True)
    assert callable(result.factory)


def test_pmap_field_different_key_value_types():
    from pyrsistent._field_common import pmap_field
    
    result1 = pmap_field(int, str)
    result2 = pmap_field(str, int)
    
    assert result1 is not None
    assert result2 is not None
    assert result1.type != result2.type


# LLM-generated content at query #90
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def failing_invariant(subject):
        return (False, "ERROR_CODE_1")
    
    invariants = [failing_invariant]
    subject = "test_subject"
    
    error_codes = tuple(error_code for is_ok, error_code in
                        (invariant(subject) for invariant in invariants) if not is_ok)
    
    assert error_codes == ("ERROR_CODE_1",)
    assert bool(error_codes) is True


# LLM-generated content at query #91
#--------------------------

```python
def test_set_fields_predicate_isinstance_pfield():
    class _PField:
        pass
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    pfield_instance = _PField()
    test_dict = {'field1': pfield_instance, 'field2': 'not_a_pfield'}
    bases = []
    name = 'fields'
    
    set_fields(test_dict, bases, name)
    
    assert 'field1' in test_dict['fields']
    assert test_dict['fields']['field1'] is pfield_instance
    assert 'field1' not in test_dict
    assert 'field2' in test_dict


# LLM-generated content at query #92
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent import pmap, pvec, field, CheckedPMap
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class MyChecked(CheckedPMap):
        items = field()
    
    checked_class = MyChecked
    item_type = int
    data = [1, 2, 3]
    
    # Register a type in _seq_field_types for testing
    class MockSeqType:
        def create(self, data, _factory_fields=None):
            return pvec(data)
    
    _seq_field_types[(checked_class, item_type)] = MockSeqType()
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result == pvec([1, 2, 3])


def test_restore_seq_field_pickle_with_strings():
    from pyrsistent import pvec, CheckedPMap, field
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class StringChecked(CheckedPMap):
        values = field()
    
    checked_class = StringChecked
    item_type = str
    data = ['a', 'b', 'c']
    
    class MockSeqType:
        def create(self, data, _factory_fields=None):
            return pvec(data)
    
    _seq_field_types[(checked_class, item_type)] = MockSeqType()
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result == pvec(['a', 'b', 'c'])


def test_restore_seq_field_pickle_empty_data():
    from pyrsistent import pvec, CheckedPMap, field
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    
    class EmptyChecked(CheckedPMap):
        items = field()
    
    checked_class = EmptyChecked
    item_type = int
    data = []
    
    class MockSeqType:
        def create(self, data, _factory_fields=None):
            return pvec(data)
    
    _seq_field_types[(checked_class, item_type)] = MockSeqType()
    
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    
    assert result == pvec([])


# LLM-generated content at query #93
#--------------------------

```python
def test_make_pmap_field_type_creates_checked_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__key_type__ == str
    assert result.__value_type__ == int
    assert "PMap" in result.__name__


def test_make_pmap_field_type_caches_result():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, str)
    
    assert result1 is not result2
    assert result1.__key_type__ == str
    assert result2.__key_type__ == int


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    pmap_type = _make_pmap_field_type(str, int)
    
    assert hasattr(pmap_type, '__reduce__')
    assert callable(getattr(pmap_type, '__reduce__'))


def test_make_pmap_field_type_name_format():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert "StrToIntPMap" == result.__name__


# LLM-generated content at query #94
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field(type=str)
    
    test_field = TestClass.__pclass_fields__['name']
    result = is_field_ignore_extra_complaint(TestClass, test_field, True)
    
    assert result is False


# LLM-generated content at query #95
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3 should evaluate to False
    # "not isinstance(t, type) and not isinstance(t, str)"
    # For t = int: isinstance(int, type) is True, so "not isinstance(int, type)" is False
    # For t = str: isinstance(str, type) is True, so "not isinstance(str, type)" is False
    # The predicate evaluates to False for both, so no TypeError is raised
    
    try:
        from_check = _check_field_parameters(field)
        assertion_result = True
    except TypeError:
        assertion_result = False
    
    assert assertion_result is True


# LLM-generated content at query #96
#--------------------------

```python
def test_sequence_field_with_optional_true():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import PVector, CheckedPVector
    
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=True,
        initial=None,
        invariant=PFIELD_NO_INVARIANT,
        item_invariant=PFIELD_NO_INVARIANT
    )
    
    assert result is not None
    assert result.mandatory is True
    assert result.initial is None


def test_sequence_field_with_optional_false():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=str,
        optional=False,
        initial=None,
        invariant=PFIELD_NO_INVARIANT,
        item_invariant=PFIELD_NO_INVARIANT
    )
    
    assert result is not None
    assert result.mandatory is True


def test_sequence_field_factory_with_optional_none():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=True,
        initial=None,
        invariant=PFIELD_NO_INVARIANT,
        item_invariant=PFIELD_NO_INVARIANT
    )
    
    factory_result = result.factory(None)
    assert factory_result is None


def test_sequence_field_factory_with_optional_list():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=True,
        initial=None,
        invariant=PFIELD_NO_INVARIANT,
        item_invariant=PFIELD_NO_INVARIANT
    )
    
    factory_result = result.factory([1, 2, 3])
    assert len(factory_result) == 3
    assert factory_result[0] == 1


def test_sequence_field_has_correct_type():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=float,
        optional=False,
        initial=None,
        invariant=PFIELD_NO_INVARIANT,
        item_invariant=PFIELD_NO_INVARIANT
    )
    
    assert result.type is not None
    assert len(result.type) > 0


def test_sequence_field_factory_callable():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(
        checked_class=CheckedPVector,
        item_type=int,
        optional=False,
        initial=None,
        invariant=PFIELD_NO_INVARIANT,
        item_invariant=PFIELD_NO_INVARIANT
    )
    
    assert callable(result.factory)


# LLM-generated content at query #97
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap, PMap
    
    # Create a simple pmap field type
    key_type = str
    value_type = int
    
    # Register a pmap field type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent import field
        pmap_field_type = PMap.create({})
        _pmap_field_types[key_type, value_type] = type('PMapField', (object,), {
            'create': classmethod(lambda cls, data, _factory_fields=None: pmap(data))
        })
    
    # Test data for unpickling
    test_data = {'a': 1, 'b': 2, 'c': 3}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, test_data)
    
    # Verify the result is a pmap with correct data
    assert isinstance(result, PMap)
    assert result == pmap(test_data)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #98
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized_value"
    
    PFIELD_NO_SERIALIZER = object()
    
    def serialize(serializer, format, value):
        if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
            return value.serialize(format)
        return serializer(format, value)
    
    checked_value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    
    assert result == "serialized_value"
    assert isinstance(checked_value, CheckedType)
    assert PFIELD_NO_SERIALIZER is PFIELD_NO_SERIALIZER


# LLM-generated content at query #99
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = lambda x: True
    
    result = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result is not None
    assert hasattr(result, '__type__')
    assert result.__type__ == item_type
    assert hasattr(result, '__invariant__')
    assert result.__invariant__ == item_invariant
    assert hasattr(result, '__reduce__')


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = lambda x: True
    
    result1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result1 is result2


def test_make_seq_field_type_different_types_create_different_classes():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    item_invariant = lambda x: True
    
    result1 = _make_seq_field_type(PVector, int, item_invariant)
    result2 = _make_seq_field_type(PVector, str, item_invariant)
    
    assert result1 is not result2
    assert result1.__type__ == int
    assert result2.__type__ == str


def test_make_seq_field_type_has_correct_name():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, lambda x: True)
    
    assert hasattr(result, '__name__')
    assert isinstance(result.__name__, str)
    assert len(result.__name__) > 0


def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear cache to ensure clean test
    _seq_field_types.clear()
    
    result = _make_seq_field_type(PVector, int, lambda x: True)
    instance = result([1, 2, 3])
    
    reduced = instance.__reduce__()
    
    assert isinstance(reduced, tuple)
    assert len(reduced) == 2
    assert callable(reduced[0])
    assert isinstance(reduced[1], tuple)


# LLM-generated content at query #100
#--------------------------

```python
def test_check_global_invariants_all_pass():
    def invariant1(subject):
        return (True, "code1")
    
    def invariant2(subject):
        return (True, "code2")
    
    subject = "test_subject"
    invariants = [invariant1, invariant2]
    
    error_codes = tuple(error_code for is_ok, error_code in
                        (invariant(subject) for invariant in invariants) if not is_ok)
    
    assert error_codes == ()
    assert len(error_codes) == 0
    assert not error_codes


# LLM-generated content at query #101
#--------------------------

```python
def test_pmap_field_optional_false_predicate():
    from pyrsistent._field_common import pmap_field
    from pyrsistent import pmap
    
    # Create a pmap field with optional=False
    result = pmap_field(str, int, optional=False)
    
    # The predicate at line 2 (which is the docstring) should evaluate to False
    # This means optional parameter is False
    assert result.mandatory is True
    assert result.initial == pmap()


# LLM-generated content at query #102
#--------------------------

```python
def test_pmap_field_basic():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert result.mandatory is True
    assert result.initial is not None
    assert result.factory is not None

def test_pmap_field_with_optional():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    assert result.mandatory is True
    assert result.initial is not None
    assert result.factory is not None

def test_pmap_field_with_invariant():
    from pyrsistent._field_common import pmap_field, PFIELD_NO_INVARIANT
    def my_invariant(val):
        return True, "valid"
    result = pmap_field(str, int, invariant=my_invariant)
    assert result.mandatory is True
    assert result.invariant != PFIELD_NO_INVARIANT

def test_pmap_field_optional_factory_with_none():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    none_result = result.factory(None)
    assert none_result is None

def test_pmap_field_type_parameter():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=False)
    assert len(result.type) > 0

def test_pmap_field_type_parameter_optional():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int, optional=True)
    assert len(result.type) > 0

def test_pmap_field_creates_checked_pmap():
    from pyrsistent._field_common import pmap_field
    result = pmap_field(str, int)
    assert callable(result.factory)

def test_pmap_field_multiple_calls_same_types():
    from pyrsistent._field_common import pmap_field
    result1 = pmap_field(str, int)
    result2 = pmap_field(str, int)
    assert type(result1.initial) == type(result2.initial)

def test_pmap_field_different_types():
    from pyrsistent._field_common import pmap_field
    result1 = pmap_field(str, int)
    result2 = pmap_field(int, str)
    assert type(result1.initial) != type(result2.initial)


# LLM-generated content at query #103
#--------------------------

```python
def test_check_field_parameters_predicate_line_3_false():
    class MockField:
        def __init__(self):
            self.type = [int, str]
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x
    
    field = MockField()
    
    # The predicate at line 3: "not isinstance(t, type) and not isinstance(t, str)"
    # For this to be False, either isinstance(t, type) is True OR isinstance(t, str) is True
    # Test with t = int (a type), so isinstance(t, type) is True
    t = int
    result = not isinstance(t, type) and not isinstance(t, str)
    assert result is False
    
    # Test with t = "string" (a str), so isinstance(t, str) is True
    t = "string"
    result = not isinstance(t, type) and not isinstance(t, str)
    assert result is False


# LLM-generated content at query #104
#--------------------------

```python
def test_set_fields_pfield_predicate():
    class _PField:
        pass
    
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    
    class Base:
        pass
    
    pfield_instance = _PField()
    test_dict = {'field1': pfield_instance, 'field2': 'not_pfield'}
    bases = [Base]
    name = 'fields'
    
    set_fields(test_dict, bases, name)
    
    assert 'field1' in test_dict['fields']
    assert test_dict['fields']['field1'] is pfield_instance
    assert 'field1' not in test_dict
    assert 'field2' in test_dict


# LLM-generated content at query #105
#--------------------------

```python
def test_is_field_ignore_extra_complaint_predicate_line_6_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        test_field = field()
    
    # Create a mock field object with a type that won't match
    class MockField:
        def __init__(self):
            self.type = str
            self.factory = lambda: None
    
    mock_field = MockField()
    
    # Call with ignore_extra=True to pass line 3-4 check
    # and with a type_cls that doesn't match field.type (line 6 predicate)
    result = is_field_ignore_extra_complaint(int, mock_field, True)
    
    # Line 6 predicate should evaluate to False, causing function to return False
    assert result is False


# LLM-generated content at query #106
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache to ensure a fresh test
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result_type is not None
    assert issubclass(result_type, PVector)
    assert result_type.__type__ == int
    assert result_type.__invariant__ is None


def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    item_type = str
    item_invariant = None
    
    result_type1 = _make_seq_field_type(PVector, item_type, item_invariant)
    result_type2 = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result_type1 is result_type2


def test_make_seq_field_type_has_correct_name():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert hasattr(result_type, '__name__')
    assert len(result_type.__name__) > 0


def test_make_seq_field_type_with_invariant():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    item_type = int
    def item_invariant(value):
        return value > 0
    
    result_type = _make_seq_field_type(PVector, item_type, item_invariant)
    
    assert result_type.__invariant__ is item_invariant


def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    from pyrsistent import PVector
    
    # Clear the cache
    _seq_field_types.clear()
    
    item_type = int
    item_invariant = None
    
    result_type = _make_seq_field_type(PVector, item_type, item_invariant)
    instance = result_type([1, 2, 3])
    
    reduce_result = instance.__reduce__()
    
    assert reduce_result is not None
    assert len(reduce_result) == 2
    assert callable(reduce_result[0])


# LLM-generated content at query #107
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle, _pmap_field_types
    from pyrsistent import pmap
    
    # Create a pmap field type
    key_type = str
    value_type = int
    
    # Register the type if not already present
    if (key_type, value_type) not in _pmap_field_types:
        from pyrsistent._field_common import pmap_field
        pmap_field(key_type, value_type)
    
    # Create sample data
    data = {'key1': 1, 'key2': 2}
    
    # Call the function
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    
    # Verify the result is a pmap with correct data
    assert isinstance(result, type(pmap()))
    assert result['key1'] == 1
    assert result['key2'] == 2
    assert len(result) == 2


# LLM-generated content at query #108
#--------------------------

```python
def test_sequence_field_basic():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    
    assert result.type is not None
    assert result.mandatory is True
    assert callable(result.factory)
    assert result.invariant == PFIELD_NO_INVARIANT


def test_sequence_field_optional_true():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, str, True, ["a", "b"])
    
    assert result.type is not None
    assert result.mandatory is True
    assert callable(result.factory)
    assert result.invariant == PFIELD_NO_INVARIANT
    assert result.factory(None) is None


def test_sequence_field_with_invariant():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    def my_invariant(x):
        return True, "valid"
    
    result = _sequence_field(CheckedPVector, int, False, [1, 2], invariant=my_invariant)
    
    assert result.type is not None
    assert result.mandatory is True
    assert callable(result.factory)
    assert callable(result.invariant)


def test_sequence_field_with_item_invariant():
    from pyrsistent._field_common import _sequence_field, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedPVector
    
    def item_inv(x):
        return True, "valid"
    
    result = _sequence_field(CheckedPVector, int, False, [5], item_invariant=item_inv)
    
    assert result.type is not None
    assert result.mandatory is True
    assert callable(result.factory)
    assert result.invariant == PFIELD_NO_INVARIANT


def test_sequence_field_optional_factory_with_none():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, None)
    
    assert result.factory(None) is None


def test_sequence_field_optional_factory_with_values():
    from pyrsistent._field_common import _sequence_field
    from pyrsistent import CheckedPVector
    
    result = _sequence_field(CheckedPVector, int, True, [1, 2, 3])
    
    created = result.factory([4, 5, 6])
    assert created is not None
    assert len(created) == 3


# LLM-generated content at query #109
#--------------------------

```python
def test_check_global_invariants_all_pass():
    invariant1 = lambda x: (True, "")
    invariant2 = lambda x: (True, "")
    subject = "test_subject"
    check_global_invariants(subject, [invariant1, invariant2])


def test_check_global_invariants_single_failure():
    invariant1 = lambda x: (True, "")
    invariant2 = lambda x: (False, "ERROR_CODE_1")
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, [invariant1, invariant2])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE_1",)
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_multiple_failures():
    invariant1 = lambda x: (False, "ERROR_CODE_1")
    invariant2 = lambda x: (False, "ERROR_CODE_2")
    invariant3 = lambda x: (True, "")
    subject = "test_subject"
    
    try:
        check_global_invariants(subject, [invariant1, invariant2, invariant3])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE_1", "ERROR_CODE_2")
        assert e.message == 'Global invariant failed'


def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    check_global_invariants(subject, [])


def test_check_global_invariants_with_none_subject():
    invariant1 = lambda x: (True, "")
    check_global_invariants(None, [invariant1])


def test_check_global_invariants_invariant_receives_subject():
    received_subject = []
    
    def capture_invariant(x):
        received_subject.append(x)
        return (True, "")
    
    subject = "test_subject"
    check_global_invariants(subject, [capture_invariant])
    assert received_subject[0] == subject


# LLM-generated content at query #110
#--------------------------

```python
def test_make_pmap_field_type_creates_checked_pmap_subclass():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert result.__key_type__ == str
    assert result.__value_type__ == int
    assert "PMap" in result.__name__


def test_make_pmap_field_type_caches_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, int)
    
    assert result1 is result2


def test_make_pmap_field_type_different_key_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(int, int)
    
    assert result1 is not result2
    assert result1.__key_type__ == str
    assert result2.__key_type__ == int


def test_make_pmap_field_type_different_value_types():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result1 = _make_pmap_field_type(str, int)
    result2 = _make_pmap_field_type(str, float)
    
    assert result1 is not result2
    assert result1.__value_type__ == int
    assert result2.__value_type__ == float


def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    instance = result()
    
    assert hasattr(instance, '__reduce__')
    assert callable(instance.__reduce__)


def test_make_pmap_field_type_name_format():
    from pyrsistent._field_common import _make_pmap_field_type
    
    result = _make_pmap_field_type(str, int)
    
    assert "To" in result.__name__
    assert result.__name__.endswith("PMap")
    assert "Str" in result.__name__
    assert "Int" in result.__name__


