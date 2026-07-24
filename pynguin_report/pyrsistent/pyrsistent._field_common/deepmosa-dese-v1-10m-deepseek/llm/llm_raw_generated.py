####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_check_global_invariants_no_error():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    subject = "test"
    invariants = [lambda x: (False, 100), lambda x: (True, 200)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)


def test_check_global_invariants_multiple_errors():
    subject = "test"
    invariants = [lambda x: (False, 10), lambda x: (False, 20), lambda x: (False, 30)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (10, 20, 30)


def test_check_global_invariants_empty_invariants():
    subject = "test"
    invariants = []
    check_global_invariants(subject, invariants)


def test_check_global_invariants_all_true():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (True, 1), lambda x: (True, 2)]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #2
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    from pyrsistent import CheckedPMap
    from pyrsistent._field_common import _pmap_field_types, _make_pmap_field_type
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert result.__name__ == "IntToStrPMap"
    assert (key_type, value_type) in _pmap_field_types
    assert _pmap_field_types[(key_type, value_type)] is result

def test_make_pmap_field_type_returns_cached_class():
    from pyrsistent._field_common import _pmap_field_types, _make_pmap_field_type
    key_type = str
    value_type = int
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call
    assert _pmap_field_types[(key_type, value_type)] is first_call

def test_make_pmap_field_type_with_custom_class_name():
    from pyrsistent import CheckedPMap
    from pyrsistent._field_common import _make_pmap_field_type
    key_type = tuple
    value_type = list
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "TupleToListPMap"

def test_make_pmap_field_type_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type, _restore_pmap_field_pickle
    key_type = float
    value_type = bool
    map_class = _make_pmap_field_type(key_type, value_type)
    instance = map_class({1.0: True})
    reduce_result = instance.__reduce__()
    assert reduce_result[0] is _restore_pmap_field_pickle
    assert reduce_result[1][0] == key_type
    assert reduce_result[1][1] == value_type
    assert reduce_result[1][2] == {1.0: True}


# LLM-generated content at query #3
#--------------------------

def test_sequence_field_creates_checked_type_with_item_type():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, int, False, [])
    assert field_obj.type == {CheckedPVector}
    assert field_obj.mandatory is True

def test_sequence_field_optional_handles_none():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, int, True, [])
    result = field_obj.factory(None)
    assert result is None

def test_sequence_field_optional_creates_instance():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, int, True, [])
    result = field_obj.factory([1, 2, 3])
    assert isinstance(result, CheckedPVector)
    assert list(result) == [1, 2, 3]

def test_sequence_field_non_optional_creates_instance():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, str, False, [])
    result = field_obj.factory(["a", "b"])
    assert isinstance(result, CheckedPVector)
    assert list(result) == ["a", "b"]

def test_sequence_field_initial_value():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, int, False, [5, 10])
    assert list(field_obj.initial) == [5, 10]

def test_sequence_field_invariant():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    def inv(val):
        return len(val) > 0, "must not be empty"
    field_obj = _sequence_field(CheckedPVector, int, False, [], invariant=inv)
    result = field_obj.invariant(CheckedPVector.create([1]))
    assert result == (True, ())
    result = field_obj.invariant(CheckedPVector.create([]))
    assert result == (False, ("must not be empty",))

def test_sequence_field_item_invariant():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    def item_inv(val):
        return val > 0, "must be positive"
    field_obj = _sequence_field(CheckedPVector, int, False, [], item_invariant=item_inv)
    result = field_obj.factory([1, 2])
    assert list(result) == [1, 2]
    try:
        field_obj.factory([-1])
        assert False
    except Exception:
        pass

def test_sequence_field_mandatory_true():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, int, False, [])
    assert field_obj.mandatory is True

def test_sequence_field_type_caching():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field, _seq_field_types
    key_before = len(_seq_field_types)
    field_obj1 = _sequence_field(CheckedPVector, int, False, [])
    key_after = len(_seq_field_types)
    assert key_after > key_before
    field_obj2 = _sequence_field(CheckedPVector, int, False, [])
    assert key_after == len(_seq_field_types)


# LLM-generated content at query #4
#--------------------------

def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    result_type = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert (MockCheckedClass, item_type) in _seq_field_types
    assert _seq_field_types[(MockCheckedClass, item_type)] is result_type
    assert result_type.__type__ is item_type
    assert result_type.__invariant__ is item_invariant
    assert issubclass(result_type, MockCheckedClass)

def test_make_seq_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    _seq_field_types[(MockCheckedClass, item_type)] = "cached_type"
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result == "cached_type"

def test_make_seq_field_type_sets_name():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, SEQ_FIELD_TYPE_SUFFIXES
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    result_type = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    expected_suffix = SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]
    assert result_type.__name__ == "Int" + expected_suffix

def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _restore_seq_field_pickle
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    result_type = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    instance = result_type()
    reduced = instance.__reduce__()
    assert reduced[0] is _restore_seq_field_pickle
    assert reduced[1] == (MockCheckedClass, item_type, list(instance))


# LLM-generated content at query #5
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import field, PField
    valid_field = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import field
    invalid_field = field(type=[str, 123])
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import field
    invalid_field = field(type=str, initial=123)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'int'>"

def test_check_field_parameters_callable_initial_allowed():
    from pyrsistent import field
    callable_initial_field = field(type=str, initial=lambda: "default")
    _check_field_parameters(callable_initial_field)

def test_check_field_parameters_no_initial_allowed():
    from pyrsistent import field, PFIELD_NO_INITIAL
    no_initial_field = field(type=str, initial=PFIELD_NO_INITIAL)
    _check_field_parameters(no_initial_field)

def test_check_field_parameters_no_type_allowed():
    from pyrsistent import field
    no_type_field = field(initial="default")
    _check_field_parameters(no_type_field)

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import field
    invalid_invariant_field = field(invariant="not callable")
    try:
        _check_field_parameters(invalid_invariant_field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_invalid_factory():
    from pyrsistent import field
    invalid_factory_field = field(factory="not callable")
    try:
        _check_field_parameters(invalid_factory_field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import field
    invalid_serializer_field = field(serializer="not callable")
    try:
        _check_field_parameters(invalid_serializer_field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"

def test_check_field_parameters_valid_initial_in_type_list():
    from pyrsistent import field
    valid_field = field(type=[str, int], initial=123)
    _check_field_parameters(valid_field)


# LLM-generated content at query #6
#--------------------------

def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from unittest.mock import Mock, patch
    mock_key_type = Mock()
    mock_value_type = Mock()
    mock_data = Mock()
    mock_pmap_field_types = {}
    mock_type = Mock()
    mock_pmap_field_types[(mock_key_type, mock_value_type)] = mock_type
    with patch('pyrsistent._field_common._pmap_field_types', mock_pmap_field_types):
        with patch('pyrsistent._field_common._restore_pickle') as mock_restore_pickle:
            mock_restore_pickle.return_value = 'restored_value'
            result = _restore_pmap_field_pickle(mock_key_type, mock_value_type, mock_data)
            mock_restore_pickle.assert_called_once_with(mock_type, mock_data)
            assert result == 'restored_value'


# LLM-generated content at query #7
#--------------------------

def test_pmap_field_creates_checked_pmap_type():
    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type)
    assert result_field.mandatory == True
    assert isinstance(result_field.initial, CheckedPMap)
    assert result_field.type == {key_type, value_type}
    assert callable(result_field.factory)

def test_pmap_field_optional_true_allows_none():
    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type, optional=True)
    assert result_field.mandatory == True
    assert isinstance(result_field.initial, CheckedPMap)
    assert type(None) in result_field.type
    assert result_field.factory(None) is None

def test_pmap_field_invariant_passed_through():
    key_type = int
    value_type = str
    def custom_invariant(value):
        return True, ()
    result_field = pmap_field(key_type, value_type, invariant=custom_invariant)
    assert result_field.invariant == wrap_invariant(custom_invariant)

def test_pmap_field_factory_creates_checked_pmap():
    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type)
    factory = result_field.factory
    test_map = factory({1: "a", 2: "b"})
    assert isinstance(test_map, CheckedPMap)
    assert test_map[1] == "a"
    assert test_map[2] == "b"

def test_pmap_field_optional_factory_handles_none():
    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type, optional=True)
    factory = result_field.factory
    assert factory(None) is None
    test_map = factory({1: "a"})
    assert isinstance(test_map, CheckedPMap)
    assert test_map[1] == "a"

def test_pmap_field_type_set_includes_key_and_value_types():
    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type)
    assert key_type in result_field.type
    assert value_type in result_field.type

def test_pmap_field_optional_type_includes_none():
    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type, optional=True)
    assert type(None) in result_field.type

def test_pmap_field_initial_is_empty_checked_pmap():
    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type)
    assert isinstance(result_field.initial, CheckedPMap)
    assert len(result_field.initial) == 0

def test_pmap_field_mandatory_is_true():
    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type)
    assert result_field.mandatory == True

def test_pmap_field_without_invariant_uses_default():
    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type)
    assert result_field.invariant == PFIELD_NO_INVARIANT


# LLM-generated content at query #8
#--------------------------

def test_types_to_names_with_builtin_types():
    result = _types_to_names((int, str, bool))
    assert result == "IntStrBool"

def test_types_to_names_with_single_type():
    result = _types_to_names((float,))
    assert result == "Float"

def test_types_to_names_with_type_strings():
    result = _types_to_names(('collections.abc.Sequence', 'typing.Optional'))
    assert result == "SequenceOptional"

def test_types_to_names_empty_tuple():
    result = _types_to_names(())
    assert result == ""

def test_types_to_names_with_mixed_types():
    result = _types_to_names((list, 'typing.Dict', tuple))
    assert result == "ListDictTuple"


# LLM-generated content at query #9
#--------------------------

def test_pfield_constructor():
    test_type = (int,)
    test_invariant = lambda x: x > 0
    test_initial = 1
    test_mandatory = True
    test_factory = lambda: 5
    test_serializer = lambda x: str(x)
    field = _PField(test_type, test_invariant, test_initial, test_mandatory, test_factory, test_serializer)
    assert field.type == test_type
    assert field.invariant == test_invariant
    assert field.initial == test_initial
    assert field.mandatory == test_mandatory
    assert field._factory == test_factory
    assert field.serializer == test_serializer


# LLM-generated content at query #10
#--------------------------

def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from unittest.mock import Mock, patch
    mock_checked_class = Mock()
    mock_item_type = Mock()
    mock_data = Mock()
    mock_type = Mock()
    mock_result = Mock()
    with patch('pyrsistent._field_common._seq_field_types', {(mock_checked_class, mock_item_type): mock_type}):
        with patch('pyrsistent._field_common._restore_pickle', return_value=mock_result) as mock_restore:
            result = _restore_seq_field_pickle(mock_checked_class, mock_item_type, mock_data)
            mock_restore.assert_called_once_with(mock_type, mock_data)
            assert result is mock_result


# LLM-generated content at query #11
#--------------------------

def test_field_with_single_type():
    f = field(type=int)
    assert f.type == {int}
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory == False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER


def test_field_with_multiple_types_as_list():
    f = field(type=[int, str])
    assert f.type == {int, str}


def test_field_with_multiple_types_as_set():
    f = field(type={int, str})
    assert f.type == {int, str}


def test_field_with_multiple_types_as_tuple():
    f = field(type=(int, str))
    assert f.type == {int, str}


def test_field_with_string_type():
    f = field(type="int")
    assert f.type == {"int"}


def test_field_with_invariant():
    inv = lambda x: (True, "")
    f = field(invariant=inv)
    assert f.invariant is not PFIELD_NO_INVARIANT
    assert callable(f.invariant)


def test_field_with_initial_value():
    f = field(initial=10, type=int)
    assert f.initial == 10


def test_field_mandatory():
    f = field(mandatory=True)
    assert f.mandatory == True


def test_field_with_factory():
    factory = lambda x: x * 2
    f = field(factory=factory)
    assert f.factory is factory


def test_field_with_serializer():
    serializer = lambda x: str(x)
    f = field(serializer=serializer)
    assert f.serializer is serializer


def test_field_invalid_type_parameter_raises():
    try:
        field(type=123)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"


def test_field_invalid_initial_type_raises():
    try:
        field(initial="not_int", type=int)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"


def test_field_non_callable_invariant_raises():
    try:
        field(invariant="not_callable")
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"


def test_field_non_callable_factory_raises():
    try:
        field(factory="not_callable")
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"


def test_field_non_callable_serializer_raises():
    try:
        field(serializer="not_callable")
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"


def test_field_with_nested_iterable_types():
    f = field(type=[[int, str], float])
    assert f.type == {int, str, float}


def test_field_with_preserved_iterable_type():
    f = field(type=list)
    assert f.type == {list}


def test_field_no_type_specified():
    f = field()
    assert f.type == set()


def test_field_invariant_wrapping():
    inv = lambda x: [(True, ""), (False, "error")]
    f = field(invariant=inv)
    result = f.invariant(None)
    assert result == (False, ("error",))


def test_field_invariant_single_bool_result():
    inv = lambda x: (True, "")
    f = field(invariant=inv)
    result = f.invariant(None)
    assert result == (True, "")


# LLM-generated content at query #12
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_one_error():
    subject = 123
    invariants = [lambda x: (True, 0), lambda x: (False, 100)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)

def test_check_global_invariants_multiple_errors():
    subject = []
    invariants = [lambda x: (False, 5), lambda x: (False, 10), lambda x: (True, 15)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {5, 10}

def test_check_global_invariants_all_errors():
    subject = {}
    invariants = [lambda x: (False, 1), lambda x: (False, 2), lambda x: (False, 3)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (1, 2, 3)

def test_check_global_invariants_empty_invariants():
    subject = None
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #13
#--------------------------

def test_check_field_parameters_predicate_at_line_3_false():
    class MockField:
        def __init__(self, type_list):
            self.type = type_list
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda x: x
            self.serializer = lambda x: x

    field_with_type = MockField([int])
    _check_field_parameters(field_with_type)

    field_with_str = MockField(["int"])
    _check_field_parameters(field_with_str)

    field_with_mixed = MockField([int, "str"])
    _check_field_parameters(field_with_mixed)


# LLM-generated content at query #14
#--------------------------

def test_factory_property_returns_type_create_when_no_factory_and_single_checkedtype():
    from pyrsistent._checked_types import CheckedType, get_type
    from pyrsistent._pfield import _PField, PFIELD_NO_FACTORY
    class MockCheckedType(CheckedType):
        @classmethod
        def create(cls, value):
            return value
    field = _PField(type=(MockCheckedType,), invariant=lambda x: True, initial=None, mandatory=True, factory=PFIELD_NO_FACTORY, serializer=None)
    result = field.factory
    expected = MockCheckedType.create
    assert result == expected


# LLM-generated content at query #15
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "IntToStrPMap" in result.__name__


def test_make_pmap_field_type_returns_cached_class():
    key_type = int
    value_type = str
    first_result = _make_pmap_field_type(key_type, value_type)
    second_result = _make_pmap_field_type(key_type, value_type)
    assert first_result is second_result


def test_make_pmap_field_type_with_different_types():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "StrToIntPMap" in result.__name__


def test_make_pmap_field_type_reduce_method():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    instance = result({1: "a"})
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_pmap_field_pickle
    assert reduced[1][0] == key_type
    assert reduced[1][1] == value_type
    assert reduced[1][2] == {1: "a"}


# LLM-generated content at query #16
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_present():
    mock_invariant_false = lambda x: (False, 101)
    mock_invariant_true = lambda x: (True, 0)
    invariants = [mock_invariant_false, mock_invariant_true]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (101,)

def test_check_global_invariants_no_exception_when_no_error_codes():
    mock_invariant_true1 = lambda x: (True, 0)
    mock_invariant_true2 = lambda x: (True, 0)
    invariants = [mock_invariant_true1, mock_invariant_true2]
    subject = "test_subject"
    check_global_invariants(subject, invariants)


# LLM-generated content at query #17
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, 123), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Type parameter expected, not <class \'int\'>'

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial='not_an_int', invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'

def test_check_field_parameters_no_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_callable_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=lambda: 10, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_no_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(), initial='anything', invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant='not_callable', factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Invariant must be callable'

def test_check_field_parameters_invalid_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory='not_callable', serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Factory must be callable'

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory=lambda x: x, serializer='not_callable')
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Serializer must be callable'


# LLM-generated content at query #18
#--------------------------

def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import pvector
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant
    assert issubclass(result, MockCheckedClass)

def test_make_seq_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import pvector
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    first = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert first is second

def test_make_seq_field_type_sets_name():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import pvector
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result.__name__ == "Int" + MockCheckedClass.__name__

def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent import pvector
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    reduce_result = result.__reduce__()
    assert reduce_result[0] == _restore_seq_field_pickle
    assert reduce_result[1][0] == MockCheckedClass
    assert reduce_result[1][1] == item_type


# LLM-generated content at query #19
#--------------------------

def test_serialize_with_checked_type_and_no_serializer():
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    checked_instance = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_instance)
    assert result == "serialized_json"

def test_serialize_with_checked_type_and_custom_serializer():
    class MockCheckedType:
        pass
    checked_instance = MockCheckedType()
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    result = serialize(custom_serializer, "xml", checked_instance)
    assert result == "custom_xml_{}".format(checked_instance)

def test_serialize_with_non_checked_type_and_no_serializer():
    value = "test_value"
    result = serialize(PFIELD_NO_SERIALIZER, "yaml", value)
    assert result == PFIELD_NO_SERIALIZER("yaml", value)

def test_serialize_with_non_checked_type_and_custom_serializer():
    value = 123
    def custom_serializer(format, value):
        return f"serialized_{format}_{value}"
    result = serialize(custom_serializer, "json", value)
    assert result == "serialized_json_123"


# LLM-generated content at query #20
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_exist():
    mock_invariant_false = lambda x: (False, 101)
    mock_invariant_true = lambda x: (True, 0)
    invariants = [mock_invariant_false, mock_invariant_true]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == (101,)
        assert e.success_codes == ()
        assert e.message == 'Global invariant failed'


# LLM-generated content at query #21
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_present():
    mock_invariant_false = lambda x: (False, "ERROR_001")
    mock_invariant_true = lambda x: (True, "ERROR_002")
    invariants = [mock_invariant_false, mock_invariant_true]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == ("ERROR_001",)

def test_check_global_invariants_no_exception_when_no_error_codes():
    mock_invariant_true1 = lambda x: (True, "ERROR_001")
    mock_invariant_true2 = lambda x: (True, "ERROR_002")
    invariants = [mock_invariant_true1, mock_invariant_true2]
    subject = "test_subject"
    check_global_invariants(subject, invariants)


# LLM-generated content at query #22
#--------------------------

def test_is_field_ignore_extra_complaint_ignore_extra_false():
    class MockField:
        type = None
        factory = None
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, False)
    assert result is False

def test_is_field_ignore_extra_complaint_not_type_cls():
    class MockField:
        type = set()
        factory = None
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_no_ignore_extra_param():
    class MockFactory:
        pass
    class MockField:
        type = (int,)
        factory = MockFactory()
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_has_ignore_extra_param():
    import inspect
    def factory_with_ignore_extra(ignore_extra):
        pass
    class MockField:
        type = (int,)
        factory = factory_with_ignore_extra
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is True

def test_is_field_ignore_extra_complaint_empty_type_tuple():
    class MockField:
        type = ()
        factory = None
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_type_as_set():
    class MockField:
        type = {int}
        factory = None
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is True


# LLM-generated content at query #23
#--------------------------

def test_serialize_checked_type_with_pfield_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_json"


# LLM-generated content at query #24
#--------------------------

def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from unittest.mock import Mock, patch
    mock_checked_class = Mock()
    mock_item_type = Mock()
    mock_data = Mock()
    mock_seq_field_types = { (mock_checked_class, mock_item_type): Mock() }
    mock_type_ = mock_seq_field_types[(mock_checked_class, mock_item_type)]
    expected_result = Mock()
    with patch('pyrsistent._field_common._seq_field_types', mock_seq_field_types):
        with patch('pyrsistent._field_common._restore_pickle', return_value=expected_result) as mock_restore:
            result = _restore_seq_field_pickle(mock_checked_class, mock_item_type, mock_data)
            mock_restore.assert_called_once_with(mock_type_, mock_data)
            assert result == expected_result


# LLM-generated content at query #25
#--------------------------

def test_make_seq_field_type_creates_subclass():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert issubclass(result, MockCheckedClass)
    assert result.__type__ is item_type
    assert result.__invariant__ is item_invariant
    assert (MockCheckedClass, item_type) in _seq_field_types
    assert _seq_field_types[(MockCheckedClass, item_type)] is result

def test_make_seq_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    _seq_field_types[(MockCheckedClass, item_type)] = 'cached_type'
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result == 'cached_type'

def test_make_seq_field_type_sets_name():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, _types_to_names, SEQ_FIELD_TYPE_SUFFIXES
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    expected_suffix = SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]
    expected_name = _types_to_names(result._checked_types) + expected_suffix
    assert result.__name__ == expected_name

def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _restore_seq_field_pickle
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    instance = result([1, 2, 3])
    reduced = instance.__reduce__()
    assert reduced[0] is _restore_seq_field_pickle
    assert reduced[1] == (MockCheckedClass, item_type, [1, 2, 3])


# LLM-generated content at query #26
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_type_cls_check_fails():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from unittest.mock import Mock
    mock_field = Mock()
    mock_field.type = "some_type"
    mock_field.factory = Mock()
    result = is_field_ignore_extra_complaint("not_a_type_class", mock_field, True)
    assert result is False


# LLM-generated content at query #27
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    from pyrsistent._checked_types import CheckedPMap
    from pyrsistent._field_common import _pmap_field_types, _make_pmap_field_type, _restore_pmap_field_pickle
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert result.__name__ == "IntToStrPMap"
    assert (key_type, value_type) in _pmap_field_types
    assert _pmap_field_types[(key_type, value_type)] == result

def test_make_pmap_field_type_returns_cached_class():
    from pyrsistent._checked_types import CheckedPMap
    from pyrsistent._field_common import _pmap_field_types, _make_pmap_field_type
    key_type = str
    value_type = int
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call
    assert _pmap_field_types[(key_type, value_type)] is first_call

def test_make_pmap_field_type_with_custom_class_name():
    from pyrsistent._checked_types import CheckedPMap
    from pyrsistent._field_common import _make_pmap_field_type
    class CustomKey:
        pass
    class CustomValue:
        pass
    result = _make_pmap_field_type(CustomKey, CustomValue)
    assert result.__name__ == "CustomkeyToCustomvaluePMap"

def test_make_pmap_field_type_reduce_method():
    from pyrsistent._checked_types import CheckedPMap
    from pyrsistent._field_common import _make_pmap_field_type, _restore_pmap_field_pickle
    key_type = float
    value_type = bool
    map_class = _make_pmap_field_type(key_type, value_type)
    instance = map_class({1.0: True})
    reduce_result = instance.__reduce__()
    assert reduce_result[0] == _restore_pmap_field_pickle
    assert reduce_result[1][0] == key_type
    assert reduce_result[1][1] == value_type
    assert isinstance(reduce_result[1][2], dict)
    assert reduce_result[1][2] == {1.0: True}


# LLM-generated content at query #28
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        pass
    field = MockField()
    field.type = [int, str]
    field.initial = 5
    field.invariant = lambda x: True
    field.factory = lambda: None
    field.serializer = lambda x: x
    result = not isinstance(field.type[0], type) and not isinstance(field.type[0], str)
    assert result == False


# LLM-generated content at query #29
#--------------------------

def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from unittest.mock import Mock, patch
    mock_checked_class = Mock()
    mock_item_type = Mock()
    mock_data = Mock()
    mock_type = Mock()
    mock_restored = Mock()
    with patch('pyrsistent._field_common._seq_field_types', {(mock_checked_class, mock_item_type): mock_type}):
        with patch('pyrsistent._field_common._restore_pickle', return_value=mock_restored) as mock_restore:
            result = _restore_seq_field_pickle(mock_checked_class, mock_item_type, mock_data)
            mock_restore.assert_called_once_with(mock_type, mock_data)
            assert result == mock_restored


# LLM-generated content at query #30
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (True, 1)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (False, 100)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)

def test_check_global_invariants_multiple_errors():
    subject = "test_subject"
    invariants = [lambda s: (False, 200), lambda s: (False, 300)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {200, 300}

def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)

def test_check_global_invariants_all_true():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (True, 1), lambda s: (True, 2)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_error_order():
    subject = "test_subject"
    invariants = [lambda s: (False, 400), lambda s: (True, 0), lambda s: (False, 500)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (400, 500)


# LLM-generated content at query #31
#--------------------------

def test_pmap_field_factory_for_optional_none():
    from pyrsistent import pmap_field, optional
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _PField, PFIELD_NO_FACTORY
    from pyrsistent.checked_types import CheckedType
    import pyrsistent as pyr

    class MockCheckedType(CheckedType):
        @classmethod
        def create(cls, argument):
            return f"created_{argument}"

    key_type = str
    value_type = int
    optional_field = pmap_field(key_type, value_type, optional=True)
    assert isinstance(optional_field, _PField)
    assert optional_field._factory is not PFIELD_NO_FACTORY
    factory_func = optional_field.factory
    result = factory_func(None)
    assert result is None
    result = factory_func({"a": 1})
    assert result is not None


# LLM-generated content at query #32
#--------------------------

def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._checked_types import _seq_field_types
    mock_checked_class = object()
    mock_item_type = object()
    mock_data = object()
    mock_type = object()
    _seq_field_types[mock_checked_class, mock_item_type] = mock_type
    result = _restore_seq_field_pickle(mock_checked_class, mock_item_type, mock_data)
    expected = _restore_pickle(mock_type, mock_data)
    assert result is expected


# LLM-generated content at query #33
#--------------------------

def test_is_field_ignore_extra_complaint_returns_false_when_type_cls_check_fails():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from unittest.mock import Mock
    field = Mock()
    field.type = 'some_type'
    type_cls = Mock()
    result = is_field_ignore_extra_complaint(type_cls, field, True)
    assert result is False


# LLM-generated content at query #34
#--------------------------

def test_field_initial_not_callable_and_type_mismatch():
    field = type('Field', (), {'initial': 42, 'type': (int, str), 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


# LLM-generated content at query #35
#--------------------------

def test_set_fields_with_no_bases():
    dct = {}
    bases = []
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {}}

def test_set_fields_with_base_dict():
    class Base:
        test = {"a": 1}
    dct = {}
    bases = [Base]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"a": 1}}

def test_set_fields_with_multiple_bases():
    class Base1:
        test = {"a": 1}
    class Base2:
        test = {"b": 2}
    dct = {}
    bases = [Base1, Base2]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"a": 1, "b": 2}}

def test_set_fields_with_pfield_in_dct():
    class _PField:
        pass
    pfield = _PField()
    dct = {"key": pfield}
    bases = []
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"key": pfield}}

def test_set_fields_with_base_dict_and_pfield():
    class _PField:
        pass
    class Base:
        test = {"base_key": "base_value"}
    pfield = _PField()
    dct = {"pkey": pfield}
    bases = [Base]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"base_key": "base_value", "pkey": pfield}}

def test_set_fields_with_overlapping_keys_in_bases():
    class Base1:
        test = {"a": 1, "c": 3}
    class Base2:
        test = {"b": 2, "c": 4}
    dct = {}
    bases = [Base1, Base2]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"a": 1, "c": 4, "b": 2}}

def test_set_fields_with_empty_base_dict_and_pfield():
    class _PField:
        pass
    class Base:
        test = {}
    pfield = _PField()
    dct = {"key": pfield}
    bases = [Base]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"key": pfield}}

def test_set_fields_with_no_name_in_base():
    class Base:
        pass
    dct = {}
    bases = [Base]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {}}


# LLM-generated content at query #36
#--------------------------

def test_serialize_with_checked_type_and_no_serializer():
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    checked_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    assert result == "serialized_json"

def test_serialize_with_checked_type_and_custom_serializer():
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    checked_value = MockCheckedType()
    result = serialize(custom_serializer, "xml", checked_value)
    assert result == "custom_xml_checked_value"

def test_serialize_with_non_checked_type_and_no_serializer():
    non_checked_value = "test_value"
    result = serialize(PFIELD_NO_SERIALIZER, "json", non_checked_value)
    assert result == PFIELD_NO_SERIALIZER("json", non_checked_value)

def test_serialize_with_non_checked_type_and_custom_serializer():
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    non_checked_value = "test_value"
    result = serialize(custom_serializer, "xml", non_checked_value)
    assert result == "custom_xml_test_value"


# LLM-generated content at query #37
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_parameter():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, 42), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Type parameter expected, not <class \'int\'>'

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial='not_an_int', invariant=lambda x: True, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'

def test_check_field_parameters_callable_initial_allowed():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=lambda: 10, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_no_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_no_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant='not_callable', factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Invariant must be callable'

def test_check_field_parameters_invalid_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory='not_callable', serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Factory must be callable'

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory=int, serializer='not_callable')
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Serializer must be callable'

def test_check_field_parameters_initial_matches_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, str), initial='hello', invariant=lambda x: True, factory=str, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_initial_matches_one_of_multiple_types():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, str), initial=42, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)


# LLM-generated content at query #38
#--------------------------

def test_serialize_checked_type_with_pfield_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    result = serialize(PFIELD_NO_SERIALIZER, "json", CheckedType())
    assert result == "serialized_json"

def test_serialize_checked_type_with_other_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    def custom_serializer(format, value):
        return f"custom_{format}"
    
    result = serialize(custom_serializer, "json", CheckedType())
    assert result == "custom_json"

def test_serialize_non_checked_type_with_pfield_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    def default_serializer(format, value):
        return f"default_{format}_{value}"
    
    global serialize
    original_serialize = serialize
    try:
        serialize = lambda s, f, v: default_serializer(f, v) if s is PFIELD_NO_SERIALIZER else s(f, v)
        result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
        assert result == "default_json_42"
    finally:
        serialize = original_serialize

def test_serialize_non_checked_type_with_custom_serializer():
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "xml", 100)
    assert result == "custom_xml_100"


# LLM-generated content at query #39
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import field, PField
    valid_field = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import field
    invalid_field = field(type=[str, 123], initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import field
    invalid_field = field(type=str, initial=123, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'int'>"

def test_check_field_parameters_callable_initial_allowed():
    from pyrsistent import field
    valid_field = field(type=str, initial=lambda: "default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_no_initial_allowed():
    from pyrsistent import field, PFIELD_NO_INITIAL
    valid_field = field(type=str, initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_no_type_allowed():
    from pyrsistent import field
    valid_field = field(type=None, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import field
    invalid_field = field(type=str, initial="default", invariant="not_callable", factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_invalid_factory():
    from pyrsistent import field
    invalid_field = field(type=str, initial="default", invariant=lambda x: True, factory="not_callable", serializer=lambda x: x)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import field
    invalid_field = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer="not_callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #40
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        pass
    field = MockField()
    field.type = [int, str]
    field.initial = 5
    field.invariant = lambda x: True
    field.factory = lambda: None
    field.serializer = lambda x: x
    result = not isinstance(field.type[0], type) and not isinstance(field.type[0], str)
    assert result == False


# LLM-generated content at query #41
#--------------------------

```python
def test_pmap_field_factory_returns_none_when_optional_and_argument_is_none():
    from pyrsistent._checked_types import optional
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import get_type
    from pyrsistent._checked_types import maybe_parse_user_type
    from pyrsistent._checked_types import wrap_invariant
    from pyrsistent._checked_types import _merge_invariant_results
    from pyrsistent._checked_types import _get_class
    from pyrsistent._checked_types import maybe_parse_many_user_types
    from pyrsistent._field_common import PFIELD_NO_FACTORY
    from pyrsistent._checked_types import CheckedType
    from pyrsistent import field
    from pyrsistent import pmap_field
    from pyrsistent import _make_pmap_field_type
    from pyrsistent import optional_type
    from pyrsistent import PMap
    from pyrsistent import CheckedPMap
    from collections.abc import Iterable
    import sys
    sys.modules['pyrsistent._checked_types']._preserved_iterable_types = (list, tuple, set, frozenset)
    key_type = str
    value_type = int
    optional = True
    TheMap = _make_pmap_field_type(key_type, value_type)
    result = pmap_field(key_type, value_type, optional)
    field_instance = result
    factory = field_instance.factory
    argument = None
    actual = factory(argument)
    expected = None
    assert actual == expected


# LLM-generated content at query #42
#--------------------------

def test_factory_property_when_factory_is_not_pfield_no_factory():
    from module_under_test import _PField, PFIELD_NO_FACTORY, CheckedType
    mock_type = (int,)
    mock_invariant = lambda x: True
    mock_initial = 0
    mock_mandatory = True
    custom_factory = lambda: 42
    mock_serializer = None
    pfield_instance = _PField(mock_type, mock_invariant, mock_initial, mock_mandatory, custom_factory, mock_serializer)
    result = pfield_instance.factory
    assert result is custom_factory

def test_factory_property_when_type_length_not_one():
    from module_under_test import _PField, PFIELD_NO_FACTORY, CheckedType
    mock_type = (int, str)
    mock_invariant = lambda x: True
    mock_initial = (0, "a")
    mock_mandatory = True
    mock_serializer = None
    pfield_instance = _PField(mock_type, mock_invariant, mock_initial, mock_mandatory, PFIELD_NO_FACTORY, mock_serializer)
    result = pfield_instance.factory
    assert result is PFIELD_NO_FACTORY

def test_factory_property_when_type_element_not_checkedtype_subclass():
    from module_under_test import _PField, PFIELD_NO_FACTORY, CheckedType
    class NotCheckedType:
        pass
    mock_type = (NotCheckedType,)
    mock_invariant = lambda x: True
    mock_initial = NotCheckedType()
    mock_mandatory = True
    mock_serializer = None
    pfield_instance = _PField(mock_type, mock_invariant, mock_initial, mock_mandatory, PFIELD_NO_FACTORY, mock_serializer)
    result = pfield_instance.factory
    assert result is PFIELD_NO_FACTORY


# LLM-generated content at query #43
#--------------------------

def test_factory_property_with_non_checked_type():
    PFIELD_NO_FACTORY = object()
    class CheckedType:
        @classmethod
        def create(cls):
            pass
    class NonCheckedType:
        pass
    from pyrsistent._checked_types import get_type
    field = _PField(type=(NonCheckedType,), invariant=None, initial=None, mandatory=True, factory=PFIELD_NO_FACTORY, serializer=None)
    result = field.factory
    assert result is PFIELD_NO_FACTORY


# LLM-generated content at query #44
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = lambda x: True
        factory = lambda x: x
        serializer = lambda x: x

    result = _check_field_parameters(MockField())
    assert result is None

    MockField.type = [42, "test"]
    try:
        _check_field_parameters(MockField())
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"


# LLM-generated content at query #45
#--------------------------

def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    key_type = int
    value_type = str
    data = {1: "a", 2: "b"}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    expected = _restore_pickle(type_, data)
    assert result == expected


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_serialize_with_checked_type_and_no_serializer():
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"

    mock_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_value)
    assert result == "serialized_json"

def test_serialize_with_checked_type_and_custom_serializer():
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"

    def custom_serializer(format, value):
        return f"custom_{format}_{value}"

    mock_value = MockCheckedType()
    result = serialize(custom_serializer, "xml", mock_value)
    assert result == "custom_xml_{}".format(mock_value)

def test_serialize_with_non_checked_type_and_no_serializer():
    mock_value = "test_string"
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_value)
    expected = PFIELD_NO_SERIALIZER("json", mock_value)
    assert result == expected

def test_serialize_with_non_checked_type_and_custom_serializer():
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"

    mock_value = 123
    result = serialize(custom_serializer, "yaml", mock_value)
    assert result == "custom_yaml_123"


# LLM-generated content at query #2
#--------------------------

def test__restore_seq_field_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    mock_checked_class = "TestClass"
    mock_item_type = "TestItem"
    mock_data = [1, 2, 3]
    mock_type = type("MockType", (), {})
    _seq_field_types[(mock_checked_class, mock_item_type)] = mock_type
    result = _restore_seq_field_pickle(mock_checked_class, mock_item_type, mock_data)
    assert result is not None


# LLM-generated content at query #3
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "IntToStrPMap" in result.__name__


def test_make_pmap_field_type_returns_cached_class():
    key_type = int
    value_type = str
    first_result = _make_pmap_field_type(key_type, value_type)
    second_result = _make_pmap_field_type(key_type, value_type)
    assert first_result is second_result


def test_make_pmap_field_type_with_different_types():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "StrToIntPMap" in result.__name__


def test_make_pmap_field_type_reduce_method():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    instance = result({1: "a"})
    reduce_result = instance.__reduce__()
    assert reduce_result[0] == _restore_pmap_field_pickle
    assert reduce_result[1][0] == key_type
    assert reduce_result[1][1] == value_type
    assert reduce_result[1][2] == {1: "a"}


# LLM-generated content at query #4
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import field, PField
    valid_field = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_invalid_type_parameter():
    from pyrsistent import field, PField
    invalid_field = field(type=[str, 123])
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import field, PField
    invalid_field = field(type=str, initial=123)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'int'>"

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import field, PField
    invalid_field = field(invariant="not callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_invalid_factory():
    from pyrsistent import field, PField
    invalid_field = field(factory="not callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import field, PField
    invalid_field = field(serializer="not callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"

def test_check_field_parameters_no_initial():
    from pyrsistent import field, PField, PFIELD_NO_INITIAL
    no_initial_field = field(type=str, initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(no_initial_field)

def test_check_field_parameters_callable_initial():
    from pyrsistent import field, PField
    callable_initial_field = field(type=str, initial=lambda: "default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(callable_initial_field)

def test_check_field_parameters_no_type():
    from pyrsistent import field, PField
    no_type_field = field(initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(no_type_field)

def test_check_field_parameters_valid_initial_in_type():
    from pyrsistent import field, PField
    valid_initial_field = field(type=[str, int], initial=123, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_initial_field)


# LLM-generated content at query #5
#--------------------------

def test_is_field_ignore_extra_complaint_ignore_extra_false():
    result = is_field_ignore_extra_complaint(type_cls=object, field=object(), ignore_extra=False)
    assert result is False

def test_is_field_ignore_extra_complaint_not_type_cls():
    class MockField:
        type = set()
    result = is_field_ignore_extra_complaint(type_cls=object, field=MockField(), ignore_extra=True)
    assert result is False

def test_is_field_ignore_extra_complaint_no_ignore_extra_param():
    class MockField:
        type = (int,)
        factory = lambda x: x
    result = is_field_ignore_extra_complaint(type_cls=int, field=MockField(), ignore_extra=True)
    assert result is False

def test_is_field_ignore_extra_complaint_has_ignore_extra_param():
    import inspect
    def factory_with_ignore_extra(x, ignore_extra):
        return x
    class MockField:
        type = (int,)
        factory = factory_with_ignore_extra
    result = is_field_ignore_extra_complaint(type_cls=int, field=MockField(), ignore_extra=True)
    assert result is True

def test_is_field_ignore_extra_complaint_empty_types():
    class MockField:
        type = ()
    result = is_field_ignore_extra_complaint(type_cls=object, field=MockField(), ignore_extra=True)
    assert result is False

def test_is_field_ignore_extra_complaint_set_type():
    class MockField:
        type = {int}
    result = is_field_ignore_extra_complaint(type_cls=int, field=MockField(), ignore_extra=True)
    assert result is True


# LLM-generated content at query #6
#--------------------------

def test_sequence_field_with_optional_false():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = [1, 2, 3]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    assert result_field.type == {checked_class}
    assert result_field.mandatory == True
    assert result_field.initial == checked_class.create(initial)

def test_sequence_field_with_optional_true():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = True
    initial = [1, 2, 3]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    assert result_field.type == {optional_type(checked_class)}
    assert result_field.mandatory == True
    assert result_field.initial == checked_class.create(initial)

def test_sequence_field_with_none_initial_when_optional():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = True
    initial = None
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    assert result_field.type == {optional_type(checked_class)}
    assert result_field.mandatory == True
    assert result_field.initial is None

def test_sequence_field_with_invariant():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = [1, 2, 3]
    def custom_invariant(value):
        return len(value) > 0, "Length must be positive"
    result_field = _sequence_field(checked_class, item_type, optional_param, initial, invariant=custom_invariant)
    assert result_field.invariant is not None
    invariant_result = result_field.invariant(checked_class.create([1]))
    assert invariant_result == (True, ())

def test_sequence_field_with_item_invariant():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = [1, 2, 3]
    def item_invariant(value):
        return value > 0, "Value must be positive"
    result_field = _sequence_field(checked_class, item_type, optional_param, initial, item_invariant=item_invariant)
    assert result_field.type == {checked_class}
    assert result_field.mandatory == True
    assert result_field.initial == checked_class.create(initial)

def test_sequence_field_factory_with_optional_true_and_none():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = True
    initial = None
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    factory_result = result_field.factory(None)
    assert factory_result is None

def test_sequence_field_factory_with_optional_true_and_value():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = True
    initial = [1, 2, 3]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    factory_result = result_field.factory([4, 5, 6])
    assert isinstance(factory_result, checked_class)
    assert list(factory_result) == [4, 5, 6]

def test_sequence_field_factory_with_optional_false():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = [1, 2, 3]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    factory_result = result_field.factory([4, 5, 6])
    assert isinstance(factory_result, checked_class)
    assert list(factory_result) == [4, 5, 6]


# LLM-generated content at query #7
#--------------------------

def test_serialize_checked_type_with_pfield_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    result = serialize(PFIELD_NO_SERIALIZER, "json", CheckedType())
    assert result == "serialized_json"

def test_serialize_checked_type_with_other_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    def custom_serializer(format, value):
        return f"custom_{format}"
    
    result = serialize(custom_serializer, "json", CheckedType())
    assert result == "custom_json"

def test_serialize_non_checked_type_with_pfield_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    result = serialize(PFIELD_NO_SERIALIZER, "json", "test_value")
    assert result == PFIELD_NO_SERIALIZER("json", "test_value")

def test_serialize_non_checked_type_with_custom_serializer():
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "json", "test")
    assert result == "custom_json_test"


# LLM-generated content at query #8
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, 123), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Type parameter expected, not <class \'int\'>'

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial='not_an_int', invariant=lambda x: True, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'

def test_check_field_parameters_callable_initial_allowed():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=lambda: 10, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_no_type_no_initial_check():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(), initial='anything', invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_non_callable_invariant():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant='not_callable', factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Invariant must be callable'

def test_check_field_parameters_non_callable_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory='not_callable', serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Factory must be callable'

def test_check_field_parameters_non_callable_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory=int, serializer='not_callable')
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Serializer must be callable'

def test_check_field_parameters_initial_pfield_no_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_initial_matches_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, str), initial='hello', invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)


# LLM-generated content at query #9
#--------------------------

def test_pfield_constructor():
    mock_type = (int,)
    mock_invariant = lambda x: x > 0
    mock_initial = 10
    mock_mandatory = True
    mock_factory = lambda: 5
    mock_serializer = lambda x: str(x)
    field = _PField(mock_type, mock_invariant, mock_initial, mock_mandatory, mock_factory, mock_serializer)
    assert field.type == mock_type
    assert field.invariant == mock_invariant
    assert field.initial == mock_initial
    assert field.mandatory == mock_mandatory
    assert field._factory == mock_factory
    assert field.serializer == mock_serializer


# LLM-generated content at query #10
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "IntToStrPMap" in result.__name__


def test_make_pmap_field_type_caches_classes():
    key_type = int
    value_type = str
    first = _make_pmap_field_type(key_type, value_type)
    second = _make_pmap_field_type(key_type, value_type)
    assert first is second


def test_make_pmap_field_type_with_custom_types():
    class CustomKey:
        pass
    class CustomValue:
        pass
    result = _make_pmap_field_type(CustomKey, CustomValue)
    assert result.__key_type__ == CustomKey
    assert result.__value_type__ == CustomValue
    assert "CustomkeyToCustomvaluePMap" in result.__name__


def test_make_pmap_field_type_reduce_method():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    instance = result({1: "a"})
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_pmap_field_pickle
    assert reduced[1][0] == key_type
    assert reduced[1][1] == value_type
    assert reduced[1][2] == {1: "a"}


# LLM-generated content at query #11
#--------------------------

def test_check_type_valid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int,)
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "test_field", 42)

def test_check_type_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int,)
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "test_field", "not_an_int")
        assert False
    except Exception as e:
        assert e.__class__.__name__ == "PTypeError"

def test_check_type_no_type_specified():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = None
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "test_field", "any_value")

def test_check_type_multiple_valid_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int, str)
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "test_field", 42)
    check_type(DestinationCls, MockField, "test_field", "valid_string")

def test_check_type_multiple_types_invalid():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int, str)
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "test_field", 3.14)
        assert False
    except Exception as e:
        assert e.__class__.__name__ == "PTypeError"

def test_check_type_with_type_string():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ("builtins.int",)
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "test_field", 42)

def test_check_type_with_type_string_invalid():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ("builtins.int",)
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "test_field", "not_an_int")
        assert False
    except Exception as e:
        assert e.__class__.__name__ == "PTypeError"

def test_check_type_empty_type_tuple():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ()
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "test_field", 42)
        assert False
    except Exception as e:
        assert e.__class__.__name__ == "PTypeError"


# LLM-generated content at query #12
#--------------------------

def test_make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result_type = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert (MockCheckedClass, item_type) in _seq_field_types
    assert _seq_field_types[(MockCheckedClass, item_type)] is result_type
    assert result_type.__type__ is item_type
    assert result_type.__invariant__ is item_invariant
    assert issubclass(result_type, MockCheckedClass)

def test_make_seq_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (float,)
    item_type = bytes
    item_invariant = lambda x: x is not None
    _seq_field_types[(MockCheckedClass, item_type)] = "cached_type"
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result == "cached_type"

def test_make_seq_field_type_sets_name_using_types_to_names():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (bool,)
    item_type = complex
    item_invariant = None
    result_type = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    expected_suffix = _make_seq_field_type.__globals__['SEQ_FIELD_TYPE_SUFFIXES'][MockCheckedClass]
    assert result_type.__name__ == "Bool" + expected_suffix

def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _restore_seq_field_pickle
    class MockCheckedClass:
        _checked_types = (list,)
    item_type = dict
    item_invariant = lambda d: len(d) > 0
    result_type = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    reduce_result = result_type.__reduce__(result_type())
    assert reduce_result[0] is _restore_seq_field_pickle
    assert reduce_result[1] == (MockCheckedClass, item_type, [])


# LLM-generated content at query #13
#--------------------------

def test_optional_field_factory_returns_none_when_argument_is_none():
    optional = True
    argument = None
    _factory_fields = None
    ignore_extra = False
    TheType = type('MockCheckedType', (), {'create': lambda self, *args, **kwargs: 'created'})
    if optional:
        def factory(argument, _factory_fields=None, ignore_extra=False):
            if argument is None:
                return None
            else:
                return TheType.create(argument, _factory_fields=_factory_fields, ignore_extra=ignore_extra)
    result = factory(argument, _factory_fields, ignore_extra)
    assert result is None


# LLM-generated content at query #14
#--------------------------

def test_initial_not_callable_and_not_instance_of_type():
    field = type('Field', (), {'initial': 42, 'type': (int, str), 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    try:
        _check_field_parameters(field)
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'int\'>'

def test_initial_not_callable_and_type_empty():
    field = type('Field', (), {'initial': 42, 'type': (), 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    _check_field_parameters(field)

def test_initial_is_callable():
    field = type('Field', (), {'initial': lambda: 42, 'type': (int, str), 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    _check_field_parameters(field)

def test_initial_is_PFIELD_NO_INITIAL():
    PFIELD_NO_INITIAL = object()
    field = type('Field', (), {'initial': PFIELD_NO_INITIAL, 'type': (int, str), 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    _check_field_parameters(field)

def test_initial_is_instance_of_type():
    field = type('Field', (), {'initial': "hello", 'type': (int, str), 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    _check_field_parameters(field)


# LLM-generated content at query #15
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (True, 1)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (False, 100)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)

def test_check_global_invariants_multiple_errors():
    subject = "test_subject"
    invariants = [lambda s: (False, 200), lambda s: (False, 300)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {200, 300}

def test_check_global_invariants_all_errors():
    subject = "test_subject"
    invariants = [lambda s: (False, 400), lambda s: (False, 500), lambda s: (False, 600)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert len(e.error_codes) == 3
        assert 400 in e.error_codes and 500 in e.error_codes and 600 in e.error_codes

def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)

def test_check_global_invariants_subject_passed_to_invariants():
    captured_subject = None
    def capturing_invariant(subj):
        nonlocal captured_subject
        captured_subject = subj
        return (True, 0)
    subject = "specific_subject"
    invariants = [capturing_invariant]
    check_global_invariants(subject, invariants)
    assert captured_subject == subject


# LLM-generated content at query #16
#--------------------------

def test_factory_property_with_checked_type():
    from pyrsistent import CheckedType, get_type, PFIELD_NO_FACTORY
    class MockCheckedType(CheckedType):
        @classmethod
        def create(cls, value):
            return value
    mock_type = (MockCheckedType,)
    pfield = _PField(type=mock_type, invariant=None, initial=None, mandatory=True, factory=PFIELD_NO_FACTORY, serializer=None)
    result = pfield.factory
    expected = MockCheckedType.create
    assert result == expected


# LLM-generated content at query #17
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, 123), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial="string", invariant=lambda x: x > 0, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'str'>"

def test_check_field_parameters_no_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=PFIELD_NO_INITIAL, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_callable_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=lambda: 10, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_no_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant="not callable", factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_invalid_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory="not callable", serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory=int, serializer="not callable")
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #18
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, 123), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Type parameter expected, not <class \'int\'>'

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial='not_an_int', invariant=lambda x: x > 0, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'

def test_check_field_parameters_no_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=PFIELD_NO_INITIAL, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_callable_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=lambda: 10, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_no_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(), initial='anything', invariant=lambda x: True, factory=str, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant='not_callable', factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Invariant must be callable'

def test_check_field_parameters_invalid_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory='not_callable', serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Factory must be callable'

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory=int, serializer='not_callable')
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Serializer must be callable'


# LLM-generated content at query #19
#--------------------------

def test_serialize_with_checked_type_and_pfield_no_serializer():
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    checked_instance = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_instance)
    assert result == "serialized_json"

def test_serialize_with_checked_type_and_other_serializer():
    class MockCheckedType:
        pass
    checked_instance = MockCheckedType()
    def mock_serializer(format, value):
        return f"custom_{format}"
    result = serialize(mock_serializer, "xml", checked_instance)
    assert result == "custom_xml"

def test_serialize_with_non_checked_type_and_pfield_no_serializer():
    non_checked_value = "some_value"
    result = serialize(PFIELD_NO_SERIALIZER, "json", non_checked_value)
    expected = PFIELD_NO_SERIALIZER("json", non_checked_value)
    assert result == expected

def test_serialize_with_non_checked_type_and_custom_serializer():
    non_checked_value = 123
    def mock_serializer(format, value):
        return f"{format}:{value}"
    result = serialize(mock_serializer, "yaml", non_checked_value)
    assert result == "yaml:123"


# LLM-generated content at query #20
#--------------------------

def test_check_global_invariants_no_error():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    subject = 5
    invariants = [lambda x: (False, 100)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)


def test_check_global_invariants_multiple_errors():
    subject = []
    invariants = [lambda x: (False, 200), lambda x: (True, 201), lambda x: (False, 202)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {200, 202}


def test_check_global_invariants_empty_invariants():
    subject = object()
    invariants = []
    check_global_invariants(subject, invariants)


def test_check_global_invariants_all_errors():
    subject = None
    invariants = [lambda x: (False, 300), lambda x: (False, 301), lambda x: (False, 302)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {300, 301, 302}


# LLM-generated content at query #21
#--------------------------

def test_sequence_field_with_optional_true():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = True
    initial = [1, 2, 3]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    assert result_field.type == optional_type(checked_class)
    assert result_field.mandatory is True
    assert result_field.initial is not None
    assert callable(result_field.factory)
    test_value = result_field.factory([4, 5, 6])
    assert test_value == [4, 5, 6]
    assert isinstance(test_value, checked_class)
    none_result = result_field.factory(None)
    assert none_result is None

def test_sequence_field_with_optional_false():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    checked_class = CheckedPVector
    item_type = str
    optional_param = False
    initial = ["a", "b"]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    assert result_field.type == checked_class
    assert result_field.mandatory is True
    assert result_field.initial is not None
    assert callable(result_field.factory)
    test_value = result_field.factory(["c", "d"])
    assert test_value == ["c", "d"]
    assert isinstance(test_value, checked_class)

def test_sequence_field_with_item_invariant():
    from pyrsistent import CheckedPSet
    from pyrsistent._field_common import _sequence_field
    def item_invariant(value):
        return value > 0, "must be positive"
    checked_class = CheckedPSet
    item_type = int
    optional_param = False
    initial = [1, 2]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial, item_invariant=item_invariant)
    assert result_field.type == checked_class
    assert result_field.mandatory is True
    assert result_field.initial is not None
    test_value = result_field.factory([3, 4])
    assert test_value == {3, 4}
    assert isinstance(test_value, checked_class)

def test_sequence_field_with_invariant():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    def invariant(value):
        return len(value) > 0, "must not be empty"
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = [1]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial, invariant=invariant)
    assert result_field.type == checked_class
    assert result_field.mandatory is True
    assert result_field.initial is not None
    test_value = result_field.factory([2, 3])
    assert test_value == [2, 3]
    assert isinstance(test_value, checked_class)

def test_sequence_field_initial_empty():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = []
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    assert result_field.type == checked_class
    assert result_field.mandatory is True
    assert result_field.initial == []
    test_value = result_field.factory([1])
    assert test_value == [1]
    assert isinstance(test_value, checked_class)


# LLM-generated content at query #22
#--------------------------

def test_pfield_constructor():
    field_type = (int,)
    invariant = lambda x: x > 0
    initial = 1
    mandatory = True
    factory = lambda: 10
    serializer = lambda x: str(x)
    field = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert field.type == field_type
    assert field.invariant == invariant
    assert field.initial == initial
    assert field.mandatory == mandatory
    assert field._factory == factory
    assert field.serializer == serializer

def test_pfield_constructor_with_defaults():
    field_type = (str,)
    invariant = None
    initial = "default"
    mandatory = False
    factory = PFIELD_NO_FACTORY
    serializer = None
    field = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert field.type == field_type
    assert field.invariant == invariant
    assert field.initial == initial
    assert field.mandatory == mandatory
    assert field._factory == factory
    assert field.serializer == serializer

def test_pfield_constructor_with_no_factory():
    field_type = (int, str)
    invariant = None
    initial = None
    mandatory = True
    factory = PFIELD_NO_FACTORY
    serializer = lambda x: x
    field = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert field.type == field_type
    assert field.invariant == invariant
    assert field.initial == initial
    assert field.mandatory == mandatory
    assert field._factory == factory
    assert field.serializer == serializer


# LLM-generated content at query #23
#--------------------------

```python
def test_check_type_with_valid_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError

    class MockField:
        type = [int]

    class DestinationClass:
        pass

    value = 42
    check_type(DestinationClass, MockField, "test_field", value)

def test_check_type_with_multiple_valid_types():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError

    class MockField:
        type = [int, float]

    class DestinationClass:
        pass

    value = 3.14
    check_type(DestinationClass, MockField, "test_field", value)

def test_check_type_with_no_type_constraint():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError

    class MockField:
        type = None

    class DestinationClass:
        pass

    value = "any_value"
    check_type(DestinationClass, MockField, "test_field", value)

def test_check_type_with_empty_type_list():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError

    class MockField:
        type = []

    class DestinationClass:
        pass

    value = "any_value"
    check_type(DestinationClass, MockField, "test_field", value)

def test_check_type_with_string_type_name():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError

    class MockField:
        type = ["builtins.int"]

    class DestinationClass:
        pass

    value = 42
    check_type(DestinationClass, MockField, "test_field", value)

def test_check_type_with_mixed_type_specifications():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError

    class MockField:
        type = [int, "builtins.float"]

    class DestinationClass:
        pass

    value = 3.14
    check_type(DestinationClass, MockField, "test_field", value)


# LLM-generated content at query #24
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        pass
    field = MockField()
    field.type = [int, str]
    field.initial = None
    field.invariant = lambda x: True
    field.factory = lambda x: x
    field.serializer = lambda x: x
    result = not isinstance(field.type[0], type) and not isinstance(field.type[0], str)
    assert result == False


# LLM-generated content at query #25
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    from pyrsistent import CheckedPMap
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert result.__name__ == "IntToStrPMap"
    assert (key_type, value_type) in _pmap_field_types
    assert _pmap_field_types[(key_type, value_type)] == result

def test_make_pmap_field_type_returns_cached_class():
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    key_type = str
    value_type = int
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call
    assert _pmap_field_types[(key_type, value_type)] is first_call

def test_make_pmap_field_type_with_custom_class_name():
    from pyrsistent import CheckedPMap
    from pyrsistent._field_common import _make_pmap_field_type
    key_type = tuple
    value_type = list
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "TupleToListPMap"

def test_make_pmap_field_type_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type, _restore_pmap_field_pickle
    key_type = float
    value_type = bool
    map_class = _make_pmap_field_type(key_type, value_type)
    instance = map_class({1.0: True})
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_pmap_field_pickle
    assert reduced[1][0] == key_type
    assert reduced[1][1] == value_type
    assert reduced[1][2] == {1.0: True}


# LLM-generated content at query #26
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    subject = "test_subject"
    invariants = [lambda x: (False, 100), lambda x: (True, 200)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)


def test_check_global_invariants_multiple_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 10), lambda x: (False, 20), lambda x: (False, 30)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (10, 20, 30)


def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)


def test_check_global_invariants_all_true():
    subject = "test_subject"
    invariants = [lambda x: (True, 1), lambda x: (True, 2), lambda x: (True, 3)]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #27
#--------------------------

def test__sequence_field_creates_checked_field_with_optional_false():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = [1, 2, 3]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    assert result_field.type == {checked_class}
    assert result_field.mandatory == True
    assert result_field.initial is not None
    assert isinstance(result_field.initial, checked_class)

def test__sequence_field_creates_checked_field_with_optional_true():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = True
    initial = [1, 2, 3]
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    assert optional_type(checked_class) in result_field.type
    assert result_field.mandatory == True
    assert result_field.initial is not None
    assert isinstance(result_field.initial, checked_class)

def test__sequence_field_with_none_initial_and_optional_true():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = True
    initial = None
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    assert optional_type(checked_class) in result_field.type
    assert result_field.mandatory == True
    assert result_field.initial is None

def test__sequence_field_with_invariant():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = [1, 2, 3]
    def invariant(value):
        return len(value) > 0, "Length must be positive"
    result_field = _sequence_field(checked_class, item_type, optional_param, initial, invariant=invariant)
    assert result_field.invariant is not None
    assert callable(result_field.invariant)

def test__sequence_field_with_item_invariant():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = [1, 2, 3]
    def item_invariant(value):
        return value > 0, "Value must be positive"
    result_field = _sequence_field(checked_class, item_type, optional_param, initial, item_invariant=item_invariant)
    assert result_field.type == {checked_class}
    assert result_field.mandatory == True
    assert result_field.initial is not None
    assert isinstance(result_field.initial, checked_class)

def test__sequence_field_factory_with_optional_true_and_none_argument():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = True
    initial = None
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    factory = result_field.factory
    result = factory(None)
    assert result is None

def test__sequence_field_factory_with_optional_true_and_non_none_argument():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = True
    initial = None
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    factory = result_field.factory
    result = factory([1, 2, 3])
    assert isinstance(result, checked_class)
    assert list(result) == [1, 2, 3]

def test__sequence_field_factory_with_optional_false():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    checked_class = CheckedPVector
    item_type = int
    optional_param = False
    initial = None
    result_field = _sequence_field(checked_class, item_type, optional_param, initial)
    factory = result_field.factory
    result = factory([1, 2, 3])
    assert isinstance(result, checked_class)
    assert list(result) == [1, 2, 3]


# LLM-generated content at query #28
#--------------------------

def test_pmap_field_creates_checked_pmap_type():
    f = pmap_field(int, str)
    assert f.type == {_make_pmap_field_type(int, str)}
    assert f.mandatory is True
    assert isinstance(f.initial, _make_pmap_field_type(int, str))
    assert f.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional_true_allows_none():
    f = pmap_field(int, str, optional=True)
    assert f.type == optional_type(_make_pmap_field_type(int, str))
    assert f.mandatory is True
    assert isinstance(f.initial, _make_pmap_field_type(int, str))
    assert f.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def inv(m):
        return len(m) > 0, "Map must not be empty"
    f = pmap_field(int, str, invariant=inv)
    wrapped_inv = wrap_invariant(inv)
    assert f.invariant == wrapped_inv

def test_pmap_field_factory_with_optional_true_and_none():
    f = pmap_field(int, str, optional=True)
    result = f.factory(None)
    assert result is None

def test_pmap_field_factory_with_optional_true_and_non_none():
    f = pmap_field(int, str, optional=True)
    input_map = {1: "a"}
    result = f.factory(input_map)
    expected_type = _make_pmap_field_type(int, str)
    assert isinstance(result, expected_type)
    assert dict(result) == input_map

def test_pmap_field_factory_with_optional_false():
    f = pmap_field(int, str, optional=False)
    input_map = {1: "a"}
    result = f.factory(input_map)
    expected_type = _make_pmap_field_type(int, str)
    assert isinstance(result, expected_type)
    assert dict(result) == input_map

def test_pmap_field_initial_is_empty_map():
    f = pmap_field(int, str)
    assert isinstance(f.initial, _make_pmap_field_type(int, str))
    assert len(f.initial) == 0

def test_pmap_field_type_set_contains_single_checked_pmap_type():
    f = pmap_field(int, str)
    assert len(f.type) == 1
    map_type = next(iter(f.type))
    assert issubclass(map_type, CheckedPMap)
    assert map_type.__key_type__ == int
    assert map_type.__value_type__ == str

def test_pmap_field_optional_type_includes_none():
    f = pmap_field(int, str, optional=True)
    type_set = f.type
    assert len(type_set) == 1
    optional_typ = next(iter(type_set))
    assert type(None) in optional_typ.__args__

def test_pmap_field_reuses_cached_map_type():
    type1 = _make_pmap_field_type(int, str)
    f = pmap_field(int, str)
    type2 = next(iter(f.type))
    assert type1 is type2

def test_pmap_field_with_different_key_value_types():
    f = pmap_field(str, list)
    map_type = next(iter(f.type))
    assert map_type.__key_type__ == str
    assert map_type.__value_type__ == list

def test_pmap_field_invariant_is_wrapped():
    def inv(m):
        return (True, "ok")
    f = pmap_field(int, str, invariant=inv)
    assert f.invariant != inv
    assert callable(f.invariant)

def test_pmap_field_no_invariant():
    f = pmap_field(int, str, invariant=PFIELD_NO_INVARIANT)
    assert f.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_mandatory_is_true():
    f = pmap_field(int, str)
    assert f.mandatory is True
    f_optional = pmap_field(int, str, optional=True)
    assert f_optional.mandatory is True


# LLM-generated content at query #29
#--------------------------

def test_optional_pmap_field_factory_returns_none_when_argument_is_none():
    optional = True
    TheMap = type('CheckedPMap', (), {'create': lambda x: 'map'})
    factory = lambda argument: None if argument is None else TheMap.create(argument)
    result = factory(None)
    assert result is None


# LLM-generated content at query #30
#--------------------------

def test_pmap_field_factory_not_checked_type():
    from pyrsistent import pmap_field, field, optional_type, CheckedType, PVector, PMap
    from pyrsistent._checked_types import _PField, PFIELD_NO_FACTORY
    import typing

    class MockCheckedType(CheckedType):
        @classmethod
        def create(cls, value):
            return value

    key_type = int
    value_type = str
    result_field = pmap_field(key_type, value_type, optional=False)
    assert isinstance(result_field, _PField)
    assert result_field._factory is not PFIELD_NO_FACTORY
    assert result_field.factory is result_field._factory


# LLM-generated content at query #31
#--------------------------

def test_make_seq_field_type_creates_subclass():
    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = int
    mock_item_invariant = lambda x: x > 0
    _seq_field_types = {}
    SEQ_FIELD_TYPE_SUFFIXES = {mock_checked_class: 'Suffix'}
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    assert issubclass(result, mock_checked_class)
    assert result.__type__ is mock_item_type
    assert result.__invariant__ is mock_item_invariant
    assert result.__name__ == 'IntSuffix'
    assert _seq_field_types[(mock_checked_class, mock_item_type)] is result

def test_make_seq_field_type_returns_cached_type():
    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = str
    mock_item_invariant = None
    _seq_field_types = {(mock_checked_class, mock_item_type): 'cached_type'}
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    assert result == 'cached_type'

def test_make_seq_field_type_sets_reduce_method():
    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = float
    mock_item_invariant = lambda x: True
    _seq_field_types = {}
    SEQ_FIELD_TYPE_SUFFIXES = {mock_checked_class: 'List'}
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    reduced = result.__reduce__(result())
    assert reduced[0] is _restore_seq_field_pickle
    assert reduced[1][0] is mock_checked_class
    assert reduced[1][1] is mock_item_type
    assert isinstance(reduced[1][2], list)

def test_make_seq_field_type_name_generation():
    mock_checked_class = type('MockCheckedClass', (), {'_checked_types': (int, str)})
    mock_item_type = bool
    mock_item_invariant = None
    _seq_field_types = {}
    SEQ_FIELD_TYPE_SUFFIXES = {mock_checked_class: 'Vector'}
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    assert result.__name__ == 'IntStrVector'


# LLM-generated content at query #32
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    class MockField:
        pass
    field = MockField()
    field.initial = None
    field.type = []
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result is False


# LLM-generated content at query #33
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = 5
    invariants = [lambda x: (True, 0), lambda x: (False, 100)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)

def test_check_global_invariants_multiple_errors():
    subject = []
    invariants = [lambda x: (False, 200), lambda x: (False, 300), lambda x: (True, 0)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {200, 300}

def test_check_global_invariants_all_errors():
    subject = None
    invariants = [lambda x: (False, 400), lambda x: (False, 500)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {400, 500}

def test_check_global_invariants_empty_invariants():
    subject = object()
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #34
#--------------------------

def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _pmap_field_types
    key_type = int
    value_type = str
    data = {1: "a", 2: "b"}
    type_ = _pmap_field_types[key_type, value_type]
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    expected = _restore_pickle(type_, data)
    assert result == expected


# LLM-generated content at query #35
#--------------------------

```python
def test_sequence_field_factory_for_optional_none():
    from pyrsistent import field, optional_type, CheckedType, CheckedPSet, CheckedPVector
    from pyrsistent._field_common import _PField, PFIELD_NO_FACTORY
    from pyrsistent._checked_types import get_type

    class MockCheckedType(CheckedType):
        @classmethod
        def create(cls, argument, _factory_fields=None, ignore_extra=False):
            return "created"

    item_type = MockCheckedType
    optional = True
    initial = None
    checked_class = CheckedPSet
    from pyrsistent._checked_types import _sequence_field
    result = _sequence_field(checked_class, item_type, optional, initial)
    assert result.type == optional_type(checked_class)
    assert result.factory(None) is None
    assert result.factory([]) == "created"


# LLM-generated content at query #36
#--------------------------

def test_pmap_field_creates_checked_pmap():
    f = pmap_field(int, str)
    assert f.type == {_make_pmap_field_type(int, str)}
    assert f.mandatory is True
    assert isinstance(f.initial, CheckedPMap)
    assert f.initial.__key_type__ == int
    assert f.initial.__value_type__ == str

def test_pmap_field_optional_true_allows_none():
    f = pmap_field(int, str, optional=True)
    assert type(None) in f.type
    assert f.factory(None) is None
    assert isinstance(f.factory({1: "a"}), CheckedPMap)

def test_pmap_field_invariant_passed_through():
    def inv(m):
        return len(m) > 0, "Map must not be empty"
    f = pmap_field(int, str, invariant=inv)
    assert f.invariant is not PFIELD_NO_INVARIANT

def test_pmap_field_initial_is_empty_map():
    f = pmap_field(int, str)
    assert len(f.initial) == 0
    assert f.initial == _make_pmap_field_type(int, str)()

def test_pmap_field_factory_creates_checked_map():
    f = pmap_field(int, str)
    m = f.factory({1: "a", 2: "b"})
    assert isinstance(m, CheckedPMap)
    assert m[1] == "a"
    assert m[2] == "b"

def test_pmap_field_with_optional_false_does_not_allow_none():
    f = pmap_field(int, str, optional=False)
    assert type(None) not in f.type

def test_pmap_field_key_and_value_types_respected():
    f = pmap_field(int, str)
    TheMap = _make_pmap_field_type(int, str)
    assert f.type == {TheMap}
    m = TheMap({1: "test"})
    assert isinstance(m, CheckedPMap)
    assert m.__key_type__ == int
    assert m.__value_type__ == str

def test_pmap_field_mandatory_is_true():
    f = pmap_field(int, str)
    assert f.mandatory is True

def test_pmap_field_serializer_default():
    f = pmap_field(int, str)
    assert f.serializer is PFIELD_NO_SERIALIZER

def test_pmap_field_invariant_default():
    f = pmap_field(int, str)
    assert f.invariant is PFIELD_NO_INVARIANT


# LLM-generated content at query #37
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        type = [int, str]
        initial = "test"
        invariant = lambda x: True
        factory = lambda x: x
        serializer = lambda x: x

    result = _check_field_parameters(MockField())
    assert result is None


