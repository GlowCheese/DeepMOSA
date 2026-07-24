####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_check_global_invariants_no_errors():
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

def test_check_global_invariants_error_message():
    subject = "test"
    invariants = [lambda x: (False, 500)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.message == 'Global invariant failed'


# LLM-generated content at query #2
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import field, PField
    valid_field = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import field, PField
    invalid_field = field(type=[str, 123], initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import field, PField
    invalid_field = field(type=str, initial=123, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'int'>"

def test_check_field_parameters_non_callable_invariant():
    from pyrsistent import field, PField
    invalid_field = field(type=str, initial="default", invariant="not_callable", factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_non_callable_factory():
    from pyrsistent import field, PField
    invalid_field = field(type=str, initial="default", invariant=lambda x: True, factory="not_callable", serializer=lambda x: x)
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_non_callable_serializer():
    from pyrsistent import field, PField
    invalid_field = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer="not_callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"

def test_check_field_parameters_callable_initial():
    from pyrsistent import field, PField
    valid_field = field(type=str, initial=lambda: "default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_no_initial():
    from pyrsistent import field, PField, PFIELD_NO_INITIAL
    valid_field = field(type=str, initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_no_type():
    from pyrsistent import field, PField
    valid_field = field(type=None, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)


# LLM-generated content at query #3
#--------------------------

def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from unittest.mock import patch
    mock_key_type = "int"
    mock_value_type = "str"
    mock_data = {"key": "value"}
    mock_type = type("MockPMapField", (), {})
    with patch("pyrsistent._field_common._pmap_field_types", {(mock_key_type, mock_value_type): mock_type}):
        with patch("pyrsistent._field_common._restore_pickle") as mock_restore:
            mock_restore.return_value = "restored_object"
            result = _restore_pmap_field_pickle(mock_key_type, mock_value_type, mock_data)
            mock_restore.assert_called_once_with(mock_type, mock_data)
            assert result == "restored_object"


# LLM-generated content at query #4
#--------------------------

def test_pfield_constructor():
    field_type = (int,)
    invariant = lambda x: x > 0
    initial = 1
    mandatory = True
    factory = lambda: 10
    serializer = lambda x: str(x)
    pfield = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert pfield.type == field_type
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer

def test_pfield_constructor_with_defaults():
    field_type = (str,)
    invariant = None
    initial = None
    mandatory = False
    factory = None
    serializer = None
    pfield = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert pfield.type == field_type
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer

def test_pfield_constructor_with_empty_type():
    field_type = ()
    invariant = lambda x: True
    initial = None
    mandatory = False
    factory = lambda: None
    serializer = lambda x: x
    pfield = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert pfield.type == field_type
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #5
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
    map_class = _make_pmap_field_type(key_type, value_type)
    instance = map_class({1: "a"})
    reduce_result = instance.__reduce__()
    assert reduce_result[0] == _restore_pmap_field_pickle
    assert reduce_result[1][0] == key_type
    assert reduce_result[1][1] == value_type
    assert dict(reduce_result[1][2]) == {1: "a"}


# LLM-generated content at query #6
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
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial="string", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'str'>"

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
    field = PField(type=(), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=PFIELD_NO_INITIAL, invariant="not callable", factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_invalid_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory="not callable", serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer="not callable")
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #7
#--------------------------

def test_is_field_ignore_extra_complaint_false_when_ignore_extra_false():
    mock_field = type('MockField', (), {'type': set(), 'factory': lambda: None})()
    result = is_field_ignore_extra_complaint(type, mock_field, False)
    assert result is False

def test_is_field_ignore_extra_complaint_false_when_not_type_cls():
    mock_field = type('MockField', (), {'type': {int}, 'factory': lambda: None})()
    result = is_field_ignore_extra_complaint(str, mock_field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_false_when_no_ignore_extra_param():
    import inspect
    def factory_without(x): pass
    mock_field = type('MockField', (), {'type': {int}, 'factory': factory_without})()
    result = is_field_ignore_extra_complaint(int, mock_field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_true_when_all_conditions_met():
    import inspect
    def factory_with_ignore_extra(x, ignore_extra=False): pass
    mock_field = type('MockField', (), {'type': {int}, 'factory': factory_with_ignore_extra})()
    result = is_field_ignore_extra_complaint(int, mock_field, True)
    assert result is True

def test_is_field_ignore_extra_complaint_false_with_empty_type_set():
    mock_field = type('MockField', (), {'type': set(), 'factory': lambda: None})()
    result = is_field_ignore_extra_complaint(int, mock_field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_true_with_string_type():
    import inspect
    def factory_with_ignore_extra(x, ignore_extra=False): pass
    mock_field = type('MockField', (), {'type': {'builtins.int'}, 'factory': factory_with_ignore_extra})()
    result = is_field_ignore_extra_complaint(int, mock_field, True)
    assert result is True


# LLM-generated content at query #8
#--------------------------

def test_factory_property_with_non_checked_type():
    from pyrsistent._checked_types import CheckedType, _PField, PFIELD_NO_FACTORY
    class NonCheckedType:
        pass
    field = _PField(type=(NonCheckedType,), invariant=lambda x: True, initial=None, mandatory=True, factory=PFIELD_NO_FACTORY, serializer=None)
    result = field.factory
    assert result is PFIELD_NO_FACTORY


# LLM-generated content at query #9
#--------------------------

def test_check_field_parameters_field_initial_not_callable_and_type_mismatch():
    from collections import namedtuple
    Field = namedtuple('Field', ['type', 'initial', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(type=[int], initial="not_an_int", invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'


# LLM-generated content at query #10
#--------------------------

def test_check_type_valid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int]
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "test_field", 42)

def test_check_type_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int]
    class DestinationClass:
        pass
    try:
        check_type(DestinationClass, MockField, "test_field", "string")
        assert False
    except Exception as e:
        assert e.__class__.__name__ == "PTypeError"
        assert "Invalid type for field DestinationClass.test_field, was str" in str(e)

def test_check_type_multiple_valid_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int, str]
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "test_field", 42)
    check_type(DestinationClass, MockField, "test_field", "string")

def test_check_type_no_type_restriction():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = None
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "test_field", 42)
    check_type(DestinationClass, MockField, "test_field", "string")
    check_type(DestinationClass, MockField, "test_field", [1, 2, 3])

def test_check_type_with_custom_class():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class CustomClass:
        pass
    class MockField:
        type = [CustomClass]
    class DestinationClass:
        pass
    custom_instance = CustomClass()
    check_type(DestinationClass, MockField, "test_field", custom_instance)

def test_check_type_with_type_string():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ["builtins.int"]
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "test_field", 42)


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from collections import namedtuple
    Field = namedtuple('Field', ['initial', 'type'])
    PFIELD_NO_INITIAL = object()
    field = Field(initial=PFIELD_NO_INITIAL, type=(int,))
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result == False
    field = Field(initial=lambda: 5, type=(int,))
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result == False
    field = Field(initial=5, type=())
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result == False
    field = Field(initial=5, type=(int,))
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result == False


# LLM-generated content at query #12
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type

def test_make_pmap_field_type_returns_cached_class():
    from pyrsistent._field_common import _make_pmap_field_type
    key_type = int
    value_type = str
    first = _make_pmap_field_type(key_type, value_type)
    second = _make_pmap_field_type(key_type, value_type)
    assert first is second

def test_make_pmap_field_type_sets_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"

def test_make_pmap_field_type_with_tuple_types():
    from pyrsistent._field_common import _make_pmap_field_type
    key_type = (int, str)
    value_type = (bool,)
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntStrToBoolPMap"

def test_make_pmap_field_type_has_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    instance = result()
    reduced = instance.__reduce__()
    assert reduced[0].__name__ == "_restore_pmap_field_pickle"
    assert reduced[1][0] == key_type
    assert reduced[1][1] == value_type
    assert reduced[1][2] == {}


# LLM-generated content at query #13
#--------------------------

def test_factory_property_with_non_checked_type():
    PFIELD_NO_FACTORY = object()
    class CheckedType:
        @classmethod
        def create(cls):
            return "checked_factory"
    class NonCheckedType:
        pass
    field = _PField(type=(NonCheckedType,), invariant=None, initial=None, mandatory=True, factory=PFIELD_NO_FACTORY, serializer=None)
    result = field.factory
    assert result is PFIELD_NO_FACTORY


# LLM-generated content at query #14
#--------------------------

def test_is_field_ignore_extra_complaint_false_when_ignore_extra_false():
    class MockField:
        type = None
        factory = None
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, False)
    assert result is False

def test_is_field_ignore_extra_complaint_false_when_not_type_cls():
    class MockField:
        type = int
        factory = None
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_true_when_ignore_extra_in_factory():
    import inspect
    def factory_with_ignore_extra(ignore_extra):
        pass
    class MockField:
        type = (type,)
        factory = factory_with_ignore_extra
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is True

def test_is_field_ignore_extra_complaint_false_when_ignore_extra_not_in_factory():
    import inspect
    def factory_without_ignore_extra():
        pass
    class MockField:
        type = (type,)
        factory = factory_without_ignore_extra
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_with_set_type():
    import inspect
    def factory_with_ignore_extra(ignore_extra):
        pass
    class MockField:
        type = {type}
        factory = factory_with_ignore_extra
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is True

def test_is_field_ignore_extra_complaint_with_empty_tuple_type():
    class MockField:
        type = ()
        factory = None
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is False


# LLM-generated content at query #15
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
    assert _pmap_field_types[(key_type, value_type)] is result

def test_make_pmap_field_type_returns_cached_class():
    from pyrsistent._field_common import _make_pmap_field_type, _pmap_field_types
    key_type = float
    value_type = bool
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call
    assert _pmap_field_types[(key_type, value_type)] is first_call

def test_make_pmap_field_type_with_string_types():
    from pyrsistent import CheckedPMap
    from pyrsistent._field_common import _make_pmap_field_type
    key_type = "builtins.int"
    value_type = "builtins.str"
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == int
    assert result.__value_type__ == str
    assert result.__name__ == "IntToStrPMap"

def test_make_pmap_field_type_name_formatting():
    from pyrsistent._field_common import _make_pmap_field_type
    key_type = (int, str)
    value_type = (bool, float)
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntStrToBoolFloatPMap"

def test_make_pmap_field_type_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type, _restore_pmap_field_pickle
    key_type = int
    value_type = str
    map_class = _make_pmap_field_type(key_type, value_type)
    instance = map_class({1: "a", 2: "b"})
    reduce_result = instance.__reduce__()
    assert reduce_result[0] is _restore_pmap_field_pickle
    assert reduce_result[1][0] == key_type
    assert reduce_result[1][1] == value_type
    assert reduce_result[1][2] == {1: "a", 2: "b"}


# LLM-generated content at query #16
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_present():
    mock_invariant_false = lambda x: (False, 1001)
    mock_invariant_true = lambda x: (True, 0)
    invariants = [mock_invariant_false, mock_invariant_true]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (1001,)
        assert e.success_codes == ()
        assert e.message == 'Global invariant failed'

def test_check_global_invariants_does_not_raise_exception_when_no_error_codes():
    mock_invariant_true1 = lambda x: (True, 0)
    mock_invariant_true2 = lambda x: (True, 0)
    invariants = [mock_invariant_true1, mock_invariant_true2]
    subject = "test_subject"
    check_global_invariants(subject, invariants)


# LLM-generated content at query #17
#--------------------------

def test_make_seq_field_type_creates_new_type():
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant
    assert result.__bases__[0] == MockCheckedClass

def test_make_seq_field_type_caches_type():
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    first = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert first is second

def test_make_seq_field_type_sets_name():
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result.__name__ == "Int" + SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]

def test_make_seq_field_type_reduce_returns_restore_function():
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    reduce_result = result.__reduce__()
    assert reduce_result[0] == _restore_seq_field_pickle
    assert reduce_result[1][0] == MockCheckedClass
    assert reduce_result[1][1] == item_type


# LLM-generated content at query #18
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_parameter():
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
    field = PField(type=(), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
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

```python
def test_is_field_ignore_extra_complaint_returns_false_when_type_check_fails():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from unittest.mock import Mock
    field = Mock()
    field.type = "some_type"
    field.factory = Mock()
    type_cls = Mock()
    result = is_field_ignore_extra_complaint(type_cls, field, True)
    assert result is False


# LLM-generated content at query #20
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
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    checked_instance = MockCheckedType()
    result = serialize(custom_serializer, "xml", checked_instance)
    assert result == "custom_xml_{}".format(checked_instance)

def test_serialize_with_non_checked_type_and_no_serializer():
    value = "test_value"
    result = serialize(PFIELD_NO_SERIALIZER, "yaml", value)
    assert result == PFIELD_NO_SERIALIZER("yaml", value)

def test_serialize_with_non_checked_type_and_custom_serializer():
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    value = 123
    result = serialize(custom_serializer, "json", value)
    assert result == "custom_json_123"


# LLM-generated content at query #21
#--------------------------

def test_field_with_single_type():
    f = field(type=int)
    assert f.type == {int}
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory == False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

def test_field_with_multiple_types_list():
    f = field(type=[int, str])
    assert f.type == {int, str}

def test_field_with_multiple_types_tuple():
    f = field(type=(int, str))
    assert f.type == {int, str}

def test_field_with_multiple_types_set():
    f = field(type={int, str})
    assert f.type == {int, str}

def test_field_with_string_type():
    f = field(type="int")
    assert f.type == {"int"}

def test_field_with_invariant():
    inv = lambda x: (True, "")
    f = field(invariant=inv)
    assert f.invariant != PFIELD_NO_INVARIANT
    assert callable(f.invariant)

def test_field_with_initial():
    f = field(initial=10, type=int)
    assert f.initial == 10

def test_field_mandatory():
    f = field(mandatory=True)
    assert f.mandatory == True

def test_field_with_factory():
    fac = lambda x: x * 2
    f = field(factory=fac)
    assert f.factory == fac

def test_field_with_serializer():
    ser = lambda x: str(x)
    f = field(serializer=ser)
    assert f.serializer == ser

def test_field_type_check_fails_on_non_type():
    try:
        field(type=123)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_field_initial_type_mismatch():
    try:
        field(initial="not_an_int", type=int)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_field_invariant_not_callable():
    try:
        field(invariant="not_callable")
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_field_factory_not_callable():
    try:
        field(factory="not_callable")
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_field_serializer_not_callable():
    try:
        field(serializer="not_callable")
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_field_with_nested_iterable_types():
    f = field(type=[[int, str], float])
    assert int in f.type
    assert str in f.type
    assert float in f.type

def test_field_no_type_spec():
    f = field()
    assert f.type == set()

def test_field_with_preserved_iterable_type():
    f = field(type=list)
    assert f.type == {list}

def test_field_factory_default_for_checkedtype():
    class MockCheckedType(CheckedType):
        @classmethod
        def create(cls, x):
            return x
    f = field(type=MockCheckedType)
    assert f.factory == MockCheckedType.create

def test_field_invariant_wrapped():
    def inv(x):
        return (True, "ok"), (False, "bad")
    f = field(invariant=inv)
    result = f.invariant(None)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == False
    assert isinstance(result[1], tuple)
    assert "bad" in result[1]


# LLM-generated content at query #22
#--------------------------

def test_serialize_checked_type_with_pfield_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    serializer = PFIELD_NO_SERIALIZER
    format = "json"
    value = CheckedType()
    result = serialize(serializer, format, value)
    assert result == "serialized_json"


# LLM-generated content at query #23
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 100), lambda x: (True, 1), lambda x: (False, 200)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100, 200)

def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)

def test_check_global_invariants_all_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 500), lambda x: (False, 600)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (500, 600)


# LLM-generated content at query #24
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    class MockField:
        pass
    field = MockField()
    field.initial = None
    field.type = []
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result == False


# LLM-generated content at query #25
#--------------------------

def test_predicate_at_line_6_false():
    class MockField:
        pass
    field = MockField()
    field.initial = None
    field.type = []
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result == False


# LLM-generated content at query #26
#--------------------------

def test_check_field_parameters_initial_invalid_type():
    field = type('Field', (), {'type': (int,), 'initial': 'not_an_int', 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    PFIELD_NO_INITIAL = object()
    field.PFIELD_NO_INITIAL = PFIELD_NO_INITIAL
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'


# LLM-generated content at query #27
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_present():
    mock_invariant_false = lambda x: (False, 1001)
    mock_invariant_true = lambda x: (True, 0)
    invariants = [mock_invariant_false, mock_invariant_true]
    subject = "test_subject"
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (1001,)

def test_check_global_invariants_no_exception_when_no_error_codes():
    mock_invariant_true1 = lambda x: (True, 0)
    mock_invariant_true2 = lambda x: (True, 0)
    invariants = [mock_invariant_true1, mock_invariant_true2]
    subject = "test_subject"
    check_global_invariants(subject, invariants)


# LLM-generated content at query #28
#--------------------------

def test_check_global_invariants_no_error():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    subject = "test"
    invariants = [lambda x: (False, 100), lambda x: (True, 1)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)


def test_check_global_invariants_multiple_errors():
    subject = "test"
    invariants = [lambda x: (False, 100), lambda x: (False, 200), lambda x: (False, 300)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {100, 200, 300}


def test_check_global_invariants_empty_invariants():
    subject = "test"
    invariants = []
    check_global_invariants(subject, invariants)


def test_check_global_invariants_all_true():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (True, 1), lambda x: (True, 2)]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #29
#--------------------------

def test_check_field_parameters_initial_invalid_type():
    from collections import namedtuple
    Field = namedtuple('Field', ['type', 'initial', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(type=[int], initial="not_an_int", invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'


# LLM-generated content at query #30
#--------------------------

def test_set_fields_with_no_bases():
    dct = {}
    bases = []
    set_fields(dct, bases, 'test')
    assert dct == {'test': {}}


def test_set_fields_with_base_containing_name():
    class Base:
        test = {'a': 1}
    dct = {}
    bases = [Base]
    set_fields(dct, bases, 'test')
    assert dct == {'test': {'a': 1}}


def test_set_fields_with_multiple_bases():
    class Base1:
        test = {'a': 1}
    class Base2:
        test = {'b': 2}
    dct = {}
    bases = [Base1, Base2]
    set_fields(dct, bases, 'test')
    assert dct == {'test': {'a': 1, 'b': 2}}


def test_set_fields_with_overlapping_keys_in_bases():
    class Base1:
        test = {'a': 1, 'c': 3}
    class Base2:
        test = {'b': 2, 'c': 4}
    dct = {}
    bases = [Base1, Base2]
    set_fields(dct, bases, 'test')
    assert dct == {'test': {'a': 1, 'c': 4, 'b': 2}}


def test_set_fields_with_pfield_in_dct():
    class _PField:
        pass
    pfield = _PField()
    dct = {'x': pfield}
    bases = []
    set_fields(dct, bases, 'test')
    assert dct == {'test': {'x': pfield}}


def test_set_fields_with_pfield_and_regular_key():
    class _PField:
        pass
    pfield = _PField()
    dct = {'x': pfield, 'y': 5}
    bases = []
    set_fields(dct, bases, 'test')
    assert dct == {'test': {'x': pfield}, 'y': 5}


def test_set_fields_with_base_and_pfield():
    class _PField:
        pass
    class Base:
        test = {'a': 1}
    pfield = _PField()
    dct = {'b': pfield}
    bases = [Base]
    set_fields(dct, bases, 'test')
    assert dct == {'test': {'a': 1, 'b': pfield}}


# LLM-generated content at query #31
#--------------------------

def test_is_field_ignore_extra_complaint_returns_false_when_type_cls_check_fails():
    class MockField:
        type = "some_type"
        factory = lambda: None

    result = is_field_ignore_extra_complaint(type_cls="not_a_type_class", field=MockField(), ignore_extra=True)
    assert result is False


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
            return f"checked_serialized_{format}"
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
        return f"serialized_{format}_{value}"
    non_checked_value = 123
    result = serialize(custom_serializer, "yaml", non_checked_value)
    assert result == "serialized_yaml_123"


# LLM-generated content at query #2
#--------------------------

def test_set_fields_adds_name_dict():
    class Base1:
        field1 = "value1"
    class Base2:
        field2 = "value2"
    dct = {}
    bases = (Base1, Base2)
    name = "test_fields"
    set_fields(dct, bases, name)
    assert dct == {"test_fields": {"field1": "value1", "field2": "value2"}}

def test_set_fields_merges_base_dicts():
    class Base1:
        test_fields = {"a": 1}
    class Base2:
        test_fields = {"b": 2}
    dct = {}
    bases = (Base1, Base2)
    name = "test_fields"
    set_fields(dct, bases, name)
    assert dct == {"test_fields": {"a": 1, "b": 2}}

def test_set_fields_moves_pfield():
    class _PField:
        pass
    pfield = _PField()
    dct = {"custom": pfield}
    bases = ()
    name = "fields"
    set_fields(dct, bases, name)
    assert dct == {"fields": {"custom": pfield}}

def test_set_fields_empty_bases():
    dct = {}
    bases = ()
    name = "meta"
    set_fields(dct, bases, name)
    assert dct == {"meta": {}}

def test_set_fields_overwrites_existing_name():
    class Base:
        existing = "old"
    dct = {"meta": {"key": "original"}}
    bases = (Base,)
    name = "meta"
    set_fields(dct, bases, name)
    assert dct == {"meta": {"existing": "old"}}

def test_set_fields_handles_multiple_pfields():
    class _PField:
        pass
    pfield1 = _PField()
    pfield2 = _PField()
    dct = {"fieldA": pfield1, "fieldB": pfield2}
    bases = ()
    name = "attrs"
    set_fields(dct, bases, name)
    assert dct == {"attrs": {"fieldA": pfield1, "fieldB": pfield2}}

def test_set_fields_ignores_non_pfield_entries():
    dct = {"regular": 42, "function": lambda x: x}
    bases = ()
    name = "special"
    set_fields(dct, bases, name)
    assert dct == {"special": {}}

def test_set_fields_mixed_base_and_pfield():
    class _PField:
        pass
    class Base:
        base_field = "base_value"
    pfield = _PField()
    dct = {"pfield": pfield}
    bases = (Base,)
    name = "fields"
    set_fields(dct, bases, name)
    assert dct == {"fields": {"base_field": "base_value", "pfield": pfield}}


# LLM-generated content at query #3
#--------------------------

def test_set_fields_pfield_condition_true():
    class _PField:
        pass
    pfield_instance = _PField()
    dct = {'key1': pfield_instance, 'key2': 'not_pfield'}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct['test_name']['key1'] is pfield_instance
    assert 'key1' not in dct


# LLM-generated content at query #4
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = "test_subject"
    invariants = [lambda x: (True, 0), lambda x: (False, 2)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (2,)

def test_check_global_invariants_multiple_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 5), lambda x: (False, 3)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {5, 3}

def test_check_global_invariants_all_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 10), lambda x: (False, 20), lambda x: (False, 30)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {10, 20, 30}

def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)

def test_check_global_invariants_subject_passed():
    captured_subject = None
    def invariant(subj):
        nonlocal captured_subject
        captured_subject = subj
        return (True, 0)
    subject = "specific_subject"
    invariants = [invariant]
    check_global_invariants(subject, invariants)
    assert captured_subject == subject


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

def test_sequence_field_creates_checked_type():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    assert result.type == {CheckedPVector}
    assert result.mandatory == True
    assert callable(result.factory)
    assert result.initial is not None

def test_sequence_field_with_optional_true():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, True, [1, 2, 3])
    assert type(None) in result.type
    assert result.mandatory == True
    factory = result.factory
    assert factory(None) is None
    assert factory([1, 2, 3]) is not None

def test_sequence_field_with_item_invariant():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    def inv(x):
        return x > 0, "Must be positive"
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3], item_invariant=inv)
    assert result.type == {CheckedPVector}
    assert result.mandatory == True

def test_sequence_field_initial_factory_called():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    initial = result.initial
    assert isinstance(initial, CheckedPVector)
    assert list(initial) == [1, 2, 3]

def test_sequence_field_with_custom_invariant():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    def inv(obj):
        return len(obj) > 0, "Must not be empty"
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3], invariant=inv)
    assert result.invariant is not None
    assert callable(result.invariant)

def test_sequence_field_type_set_correctly():
    from pyrsistent import CheckedPSet
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPSet, str, False, {"a", "b"})
    assert result.type == {CheckedPSet}
    assert result.mandatory == True

def test_sequence_field_optional_type_includes_none():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, True, [1, 2, 3])
    assert type(None) in result.type
    assert len(result.type) > 1

def test_sequence_field_factory_creates_instance():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, False, [])
    factory = result.factory
    instance = factory([4, 5, 6])
    assert isinstance(instance, CheckedPVector)
    assert list(instance) == [4, 5, 6]

def test_sequence_field_initial_none_when_optional():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, True, None)
    assert result.initial is None

def test_sequence_field_mandatory_always_true():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, False, [])
    assert result.mandatory == True
    result2 = _sequence_field(CheckedPVector, int, True, [])
    assert result2.mandatory == True


# LLM-generated content at query #7
#--------------------------

def test_sequence_field_factory_for_optional_none():
    from pyrsistent import field, optional_type
    from pyrsistent._checked_types import _make_seq_field_type
    from pyrsistent._field_common import _PField
    from pyrsistent._field_common import PFIELD_NO_FACTORY, PFIELD_NO_INVARIANT
    from pyrsistent import CheckedType
    class MockCheckedType(CheckedType):
        pass
    class MockCheckedClass:
        create = lambda self, *args, **kwargs: "created"
    checked_class = MockCheckedClass
    item_type = int
    optional = True
    initial = None
    TheType = type('TheType', (), {'create': lambda *args, **kwargs: "TheType_created"})
    _make_seq_field_type = lambda *args: TheType
    result = field(type=optional_type(TheType) if optional else TheType, factory=lambda argument, **kwargs: None if argument is None else TheType.create(argument, **kwargs), mandatory=True, invariant=PFIELD_NO_INVARIANT, initial=None)
    field_instance = _PField(type=(optional_type(TheType) if optional else TheType,), invariant=PFIELD_NO_INVARIANT, initial=None, mandatory=True, factory=PFIELD_NO_FACTORY, serializer=None)
    assert field_instance.factory is not PFIELD_NO_FACTORY
    assert callable(field_instance.factory)
    argument = None
    factory_result = field_instance.factory(argument, _factory_fields=None, ignore_extra=False)
    assert factory_result is None


# LLM-generated content at query #8
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_exist():
    error_codes = (1, 2, 3)
    result = (False, 1)
    mock_invariant = lambda x: result
    subject = None
    invariants = [mock_invariant, mock_invariant, mock_invariant]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == error_codes

def test_check_global_invariants_does_not_raise_exception_when_no_error_codes():
    result = (True, None)
    mock_invariant = lambda x: result
    subject = None
    invariants = [mock_invariant, mock_invariant]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #9
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_parameter():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, 123), initial=5, invariant=lambda x: x > 0, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial="not_an_int", invariant=lambda x: x > 0, factory=int, serializer=str)
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
    field = PField(type=(int,), initial=5, invariant="not_callable", factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_invalid_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory="not_callable", serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory=int, serializer="not_callable")
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #10
#--------------------------

def test_serialize_checked_type_with_pfield_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    serializer = PFIELD_NO_SERIALIZER
    format = "json"
    value = CheckedType()
    result = serialize(serializer, format, value)
    assert result == "serialized_json"

def test_serialize_non_checked_type_with_pfield_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    serializer = PFIELD_NO_SERIALIZER
    format = "json"
    value = "some_value"
    result = serialize(serializer, format, value)
    assert result == serializer(format, value)

def test_serialize_checked_type_with_other_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    serializer = custom_serializer
    format = "json"
    value = CheckedType()
    result = serialize(serializer, format, value)
    assert result == custom_serializer(format, value)


# LLM-generated content at query #11
#--------------------------

def test_make_seq_field_type_creates_subclass():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: x == "test"
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
    item_invariant = lambda x: x == "test"
    _seq_field_types.clear()
    first = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert first is second

def test_make_seq_field_type_sets_name():
    from pyrsistent._field_common import _make_seq_field_type, SEQ_FIELD_TYPE_SUFFIXES
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    expected_suffix = SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]
    assert result.__name__ == "Int" + expected_suffix

def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _restore_seq_field_pickle
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    reduce_result = result.__reduce__()
    assert reduce_result[0] is _restore_seq_field_pickle
    assert reduce_result[1][0] is MockCheckedClass
    assert reduce_result[1][1] is item_type


# LLM-generated content at query #12
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import field, PField
    valid_field = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_invalid_type_parameter():
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

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import field
    invalid_field = field(invariant="not callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_invalid_factory():
    from pyrsistent import field
    invalid_field = field(factory="not callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import field
    invalid_field = field(serializer="not callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"

def test_check_field_parameters_no_initial():
    from pyrsistent import field, PFIELD_NO_INITIAL
    no_initial_field = field(type=str, initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(no_initial_field)

def test_check_field_parameters_callable_initial():
    from pyrsistent import field
    callable_initial_field = field(type=str, initial=lambda: "default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(callable_initial_field)

def test_check_field_parameters_no_type():
    from pyrsistent import field
    no_type_field = field(initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(no_type_field)

def test_check_field_parameters_multiple_types_valid_initial():
    from pyrsistent import field
    multi_type_field = field(type=[str, int], initial=123, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(multi_type_field)

def test_check_field_parameters_multiple_types_invalid_initial():
    from pyrsistent import field
    multi_type_field = field(type=[str, int], initial=12.34)
    try:
        _check_field_parameters(multi_type_field)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'float'>"


# LLM-generated content at query #13
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_with_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 100), lambda x: (True, 1), lambda x: (False, 200)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100, 200)


def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)


def test_check_global_invariants_all_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 500), lambda x: (False, 600)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (500, 600)


# LLM-generated content at query #14
#--------------------------

def test_serialize_checked_type_with_pfield_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    result = serialize(PFIELD_NO_SERIALIZER, "json", CheckedType())
    assert result == "serialized_json"


# LLM-generated content at query #15
#--------------------------

def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from unittest.mock import Mock, patch
    mock_key_type = Mock()
    mock_value_type = Mock()
    mock_data = Mock()
    mock_pmap_field_types = { (mock_key_type, mock_value_type): Mock() }
    mock_type_ = mock_pmap_field_types[(mock_key_type, mock_value_type)]
    expected_result = Mock()
    with patch('pyrsistent._field_common._pmap_field_types', mock_pmap_field_types):
        with patch('pyrsistent._field_common._restore_pickle', return_value=expected_result) as mock_restore:
            result = _restore_pmap_field_pickle(mock_key_type, mock_value_type, mock_data)
    mock_restore.assert_called_once_with(mock_type_, mock_data)
    assert result is expected_result


# LLM-generated content at query #16
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import field, PField
    valid_field = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(valid_field)

def test_check_field_parameters_invalid_type_parameter():
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

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import field
    invalid_field = field(invariant="not callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_invalid_factory():
    from pyrsistent import field
    invalid_field = field(factory="not callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import field
    invalid_field = field(serializer="not callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"

def test_check_field_parameters_no_initial():
    from pyrsistent import field, PFIELD_NO_INITIAL
    no_initial_field = field(type=str, initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(no_initial_field)

def test_check_field_parameters_callable_initial():
    from pyrsistent import field
    callable_initial_field = field(type=str, initial=lambda: "default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(callable_initial_field)

def test_check_field_parameters_no_type():
    from pyrsistent import field
    no_type_field = field(initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(no_type_field)

def test_check_field_parameters_multiple_types_valid_initial():
    from pyrsistent import field
    multi_type_field = field(type=[str, int], initial=123, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(multi_type_field)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_serialize_with_checked_type_and_no_serializer():
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    checked_instance = MockCheckedType()
    result = serialize(None, "json", checked_instance)
    assert result == "serialized_json"

def test_serialize_with_checked_type_and_custom_serializer():
    class MockCheckedType:
        pass
    checked_instance = MockCheckedType()
    def mock_serializer(format, value):
        return f"custom_{format}"
    result = serialize(mock_serializer, "xml", checked_instance)
    assert result == "custom_xml"

def test_serialize_with_non_checked_type_and_no_serializer():
    non_checked_value = "test_string"
    result = serialize(None, "json", non_checked_value)
    assert result is None

def test_serialize_with_non_checked_type_and_custom_serializer():
    non_checked_value = "test_string"
    def mock_serializer(format, value):
        return f"serialized_{value}_{format}"
    result = serialize(mock_serializer, "yaml", non_checked_value)
    assert result == "serialized_test_string_yaml"

def test_serialize_with_checked_type_and_no_serializer_returns_serialized():
    class MockCheckedType:
        def serialize(self, format):
            return {"format": format, "value": "data"}
    checked_instance = MockCheckedType()
    result = serialize(None, "binary", checked_instance)
    assert result == {"format": "binary", "value": "data"}

def test_serialize_with_checked_type_and_no_serializer_calls_serialize_with_format():
    class MockCheckedType:
        def serialize(self, format):
            return f"checked_serialize_called_with_{format}"
    checked_instance = MockCheckedType()
    result = serialize(None, "csv", checked_instance)
    assert result == "checked_serialize_called_with_csv"


# LLM-generated content at query #2
#--------------------------

def test_check_type_valid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int]
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "field_name", 42)

def test_check_type_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int]
    class DestinationClass:
        pass
    try:
        check_type(DestinationClass, MockField, "field_name", "not_an_int")
        assert False
    except Exception as e:
        assert "Invalid type for field DestinationClass.field_name, was str" in str(e)

def test_check_type_no_type_specified():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = None
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "field_name", "any_value")

def test_check_type_multiple_valid_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int, str]
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "field_name", 42)
    check_type(DestinationClass, MockField, "field_name", "valid_string")

def test_check_type_multiple_invalid_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int, str]
    class DestinationClass:
        pass
    try:
        check_type(DestinationClass, MockField, "field_name", 3.14)
        assert False
    except Exception as e:
        assert "Invalid type for field DestinationClass.field_name, was float" in str(e)

def test_check_type_with_type_string():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ["builtins.int"]
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "field_name", 42)

def test_check_type_with_type_string_invalid():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ["builtins.int"]
    class DestinationClass:
        pass
    try:
        check_type(DestinationClass, MockField, "field_name", "not_an_int")
        assert False
    except Exception as e:
        assert "Invalid type for field DestinationClass.field_name, was str" in str(e)


# LLM-generated content at query #3
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
    field = PField(type=(int,), initial="not_an_int", invariant=lambda x: x > 0, factory=int, serializer=str)
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

def test_check_field_parameters_non_callable_invariant():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant="not_callable", factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_non_callable_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory="not_callable", serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_non_callable_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: x > 0, factory=int, serializer="not_callable")
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from collections import namedtuple
    PFIELD_NO_INITIAL = object()
    Field = namedtuple('Field', ['type', 'initial', 'invariant', 'factory', 'serializer'])
    field = Field(type=[], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result == False


# LLM-generated content at query #5
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, 123), initial=5, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial="string", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'str'>"

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
    field = PField(type=(), initial=5, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_invariant():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant="not callable", factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_invalid_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory="not callable", serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_invalid_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory=lambda x: x, serializer="not callable")
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #6
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import field, PField
    f = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(f)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import field, PField
    f = field(type=[str, 123], initial="default", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(f)
        assert False
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import field, PField
    f = field(type=str, initial=123, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(f)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'int'>"

def test_check_field_parameters_non_callable_invariant():
    from pyrsistent import field, PField
    f = field(type=str, initial="default", invariant="not callable", factory=lambda x: x, serializer=lambda x: x)
    try:
        _check_field_parameters(f)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_non_callable_factory():
    from pyrsistent import field, PField
    f = field(type=str, initial="default", invariant=lambda x: True, factory="not callable", serializer=lambda x: x)
    try:
        _check_field_parameters(f)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_non_callable_serializer():
    from pyrsistent import field, PField
    f = field(type=str, initial="default", invariant=lambda x: True, factory=lambda x: x, serializer="not callable")
    try:
        _check_field_parameters(f)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"

def test_check_field_parameters_callable_initial_with_type():
    from pyrsistent import field, PField
    f = field(type=str, initial=lambda: "callable", invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(f)

def test_check_field_parameters_no_type_with_invalid_initial():
    from pyrsistent import field, PField
    f = field(initial=123, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(f)

def test_check_field_parameters_pfield_no_initial():
    from pyrsistent import field, PField, PFIELD_NO_INITIAL
    f = field(type=str, initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(f)

def test_check_field_parameters_multiple_types_valid_initial():
    from pyrsistent import field, PField
    f = field(type=[str, int], initial=10, invariant=lambda x: True, factory=lambda x: x, serializer=lambda x: x)
    _check_field_parameters(f)


# LLM-generated content at query #7
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


def test_make_pmap_field_type_with_custom_types():
    class CustomKey:
        pass
    class CustomValue:
        pass
    key_type = CustomKey
    value_type = CustomValue
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "CustomkeyToCustomvaluePMap" in result.__name__


def test_make_pmap_field_type_reduce_method():
    key_type = int
    value_type = str
    map_class = _make_pmap_field_type(key_type, value_type)
    instance = map_class({1: "a"})
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_pmap_field_pickle
    assert reduced[1][0] == key_type
    assert reduced[1][1] == value_type
    assert reduced[1][2] == {1: "a"}


# LLM-generated content at query #8
#--------------------------

def test_pmap_field_creates_checked_pmap_with_key_and_value_types():
    from pyrsistent import pmap_field, CheckedPMap
    field_spec = pmap_field(key_type=int, value_type=str)
    assert field_spec.type == {CheckedPMap}
    assert field_spec.mandatory is True
    assert isinstance(field_spec.initial, CheckedPMap)
    assert field_spec.initial.__key_type__ == int
    assert field_spec.initial.__value_type__ == str

def test_pmap_field_optional_allows_none():
    from pyrsistent import pmap_field, optional_type, CheckedPMap
    field_spec = pmap_field(key_type=int, value_type=str, optional=True)
    assert field_spec.type == {optional_type(CheckedPMap)}
    assert field_spec.factory(None) is None
    assert isinstance(field_spec.factory({}), CheckedPMap)

def test_pmap_field_invariant_is_wrapped():
    from pyrsistent import pmap_field, CheckedPMap
    def custom_invariant(value):
        return len(value) > 0, "Map must not be empty"
    field_spec = pmap_field(key_type=int, value_type=str, invariant=custom_invariant)
    result = field_spec.invariant(CheckedPMap.create({1: "a"}))
    assert result == (True, ())

def test_pmap_field_initial_is_empty_checked_pmap():
    from pyrsistent import pmap_field, CheckedPMap
    field_spec = pmap_field(key_type=str, value_type=int)
    initial_map = field_spec.initial
    assert isinstance(initial_map, CheckedPMap)
    assert len(initial_map) == 0
    assert initial_map.__key_type__ == str
    assert initial_map.__value_type__ == int

def test_pmap_field_factory_creates_checked_pmap_from_dict():
    from pyrsistent import pmap_field, CheckedPMap
    field_spec = pmap_field(key_type=str, value_type=int)
    created_map = field_spec.factory({"x": 1, "y": 2})
    assert isinstance(created_map, CheckedPMap)
    assert created_map["x"] == 1
    assert created_map["y"] == 2
    assert created_map.__key_type__ == str
    assert created_map.__value_type__ == int

def test_pmap_field_with_custom_invariant_enforces_constraint():
    from pyrsistent import pmap_field, CheckedPMap
    def size_invariant(value):
        return len(value) <= 2, "Map too large"
    field_spec = pmap_field(key_type=int, value_type=str, invariant=size_invariant)
    small_map = CheckedPMap.create({1: "a"})
    large_map = CheckedPMap.create({1: "a", 2: "b", 3: "c"})
    assert field_spec.invariant(small_map) == (True, ())
    assert field_spec.invariant(large_map) == (False, ("Map too large",))

def test_pmap_field_mandatory_is_true():
    from pyrsistent import pmap_field
    field_spec = pmap_field(key_type=float, value_type=bool)
    assert field_spec.mandatory is True

def test_pmap_field_optional_factory_handles_none_and_dict():
    from pyrsistent import pmap_field, CheckedPMap
    field_spec = pmap_field(key_type=str, value_type=list, optional=True)
    assert field_spec.factory(None) is None
    created_map = field_spec.factory({"key": [1, 2]})
    assert isinstance(created_map, CheckedPMap)
    assert created_map["key"] == [1, 2]

def test_pmap_field_type_set_contains_checked_pmap_class():
    from pyrsistent import pmap_field, CheckedPMap
    field_spec = pmap_field(key_type=bytes, value_type=float)
    assert len(field_spec.type) == 1
    map_class = next(iter(field_spec.type))
    assert issubclass(map_class, CheckedPMap)

def test_pmap_field_serializer_is_default():
    from pyrsistent import pmap_field, PFIELD_NO_SERIALIZER
    field_spec = pmap_field(key_type=int, value_type=str)
    assert field_spec.serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #9
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
    
    result = serialize(custom_serializer, "xml", CheckedType())
    assert result == "custom_xml"

def test_serialize_non_checked_type_with_pfield_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    def default_serializer(format, value):
        return f"default_{format}_{value}"
    
    result = serialize(PFIELD_NO_SERIALIZER, "yaml", 42)
    assert result == "default_yaml_42"

def test_serialize_non_checked_type_with_custom_serializer():
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "csv", "data")
    assert result == "custom_csv_data"


# LLM-generated content at query #10
#--------------------------

def test_check_type_valid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int]
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)

def test_check_type_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int]
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "field_name", "string")
        assert False
    except Exception as e:
        assert "Invalid type for field DestinationCls.field_name, was str" in str(e)

def test_check_type_no_type_specified():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = None
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", "any_value")

def test_check_type_multiple_valid_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int, str]
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)
    check_type(DestinationCls, MockField, "field_name", "string")

def test_check_type_multiple_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int, str]
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "field_name", 3.14)
        assert False
    except Exception as e:
        assert "Invalid type for field DestinationCls.field_name, was float" in str(e)

def test_check_type_with_type_string():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ["builtins.int"]
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)

def test_check_type_with_type_string_invalid():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ["builtins.int"]
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "field_name", "string")
        assert False
    except Exception as e:
        assert "Invalid type for field DestinationCls.field_name, was str" in str(e)


# LLM-generated content at query #11
#--------------------------

def test_is_field_ignore_extra_complaint_ignore_extra_false():
    result = is_field_ignore_extra_complaint(type, None, False)
    assert result == False

def test_is_field_ignore_extra_complaint_not_type_cls():
    class MockField:
        type = int
    result = is_field_ignore_extra_complaint(dict, MockField, True)
    assert result == False

def test_is_field_ignore_extra_complaint_no_ignore_extra_param():
    class MockFactory:
        pass
    class MockField:
        type = {dict}
        factory = MockFactory
    result = is_field_ignore_extra_complaint(dict, MockField, True)
    assert result == False

def test_is_field_ignore_extra_complaint_valid():
    class MockFactory:
        def __init__(self, ignore_extra):
            pass
    class MockField:
        type = {dict}
        factory = MockFactory
    result = is_field_ignore_extra_complaint(dict, MockField, True)
    assert result == True

def test_is_field_ignore_extra_complaint_empty_type_set():
    class MockFactory:
        def __init__(self, ignore_extra):
            pass
    class MockField:
        type = set()
        factory = MockFactory
    result = is_field_ignore_extra_complaint(dict, MockField, True)
    assert result == False

def test_is_field_ignore_extra_complaint_type_tuple():
    class MockFactory:
        def __init__(self, ignore_extra):
            pass
    class MockField:
        type = (dict,)
        factory = MockFactory
    result = is_field_ignore_extra_complaint(dict, MockField, True)
    assert result == True


# LLM-generated content at query #12
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

def test_make_seq_field_type_name_generation():
    from pyrsistent._field_common import _make_seq_field_type, SEQ_FIELD_TYPE_SUFFIXES
    class MockCheckedClass:
        _checked_types = (int, str)
    item_type = bool
    item_invariant = None
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    expected_suffix = SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]
    assert result.__name__ == "IntStr" + expected_suffix

def test_make_seq_field_type_reuse_cached_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (float,)
    item_type = list
    item_invariant = lambda x: len(x) > 0
    first_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert first_call is second_call
    assert _seq_field_types[(MockCheckedClass, item_type)] is first_call

def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _restore_seq_field_pickle
    class MockCheckedClass:
        _checked_types = (bytes,)
    item_type = dict
    item_invariant = None
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    reduce_output = result.__reduce__()
    assert reduce_output[0] is _restore_seq_field_pickle
    assert reduce_output[1] == (MockCheckedClass, item_type, [])


# LLM-generated content at query #13
#--------------------------

def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from unittest.mock import patch, MagicMock
    mock_key_type = MagicMock()
    mock_value_type = MagicMock()
    mock_data = MagicMock()
    mock_type = MagicMock()
    mock_restored = MagicMock()
    with patch('pyrsistent._field_common._pmap_field_types', {(mock_key_type, mock_value_type): mock_type}):
        with patch('pyrsistent._field_common._restore_pickle', return_value=mock_restored) as mock_restore:
            result = _restore_pmap_field_pickle(mock_key_type, mock_value_type, mock_data)
            mock_restore.assert_called_once_with(mock_type, mock_data)
            assert result == mock_restored


# LLM-generated content at query #14
#--------------------------

def test_set_fields_with_no_bases():
    dct = {}
    bases = []
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {}}

def test_set_fields_with_base_containing_name():
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
    dct = {"x": pfield}
    bases = []
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"x": pfield}}

def test_set_fields_with_bases_and_pfield():
    class _PField:
        pass
    class Base:
        test = {"a": 1}
    pfield = _PField()
    dct = {"x": pfield}
    bases = [Base]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"a": 1, "x": pfield}}

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

def test_set_fields_with_empty_dict_in_base():
    class Base:
        test = {}
    dct = {}
    bases = [Base]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {}}

def test_set_fields_with_non_dict_in_base():
    class Base:
        test = "not a dict"
    dct = {}
    bases = [Base]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {}}


# LLM-generated content at query #15
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
    
    PFIELD_NO_SERIALIZER = object()
    result = serialize(custom_serializer, "json", CheckedType())
    assert result == "custom_json"

def test_serialize_non_checked_type_with_pfield_no_serializer():
    class NonCheckedType:
        pass
    
    PFIELD_NO_SERIALIZER = object()
    result = serialize(PFIELD_NO_SERIALIZER, "json", NonCheckedType())
    assert result is None

def test_serialize_non_checked_type_with_custom_serializer():
    class NonCheckedType:
        pass
    
    def custom_serializer(format, value):
        return f"custom_{format}"
    
    result = serialize(custom_serializer, "json", NonCheckedType())
    assert result == "custom_json"


# LLM-generated content at query #16
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
        pass
    checked_value = MockCheckedType()
    def custom_serializer(format, value):
        return f"custom_{format}"
    result = serialize(custom_serializer, "xml", checked_value)
    assert result == "custom_xml"

def test_serialize_with_non_checked_type_and_no_serializer():
    non_checked_value = "some_value"
    result = serialize(PFIELD_NO_SERIALIZER, "json", non_checked_value)
    assert result == PFIELD_NO_SERIALIZER("json", non_checked_value)

def test_serialize_with_non_checked_type_and_custom_serializer():
    non_checked_value = "some_value"
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    result = serialize(custom_serializer, "yaml", non_checked_value)
    assert result == "custom_yaml_some_value"


# LLM-generated content at query #17
#--------------------------

def test_make_seq_field_type_creates_new_type():
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant
    assert issubclass(result, MockCheckedClass)

def test_make_seq_field_type_caches_type():
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    first_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert first_call is second_call

def test_make_seq_field_type_sets_name():
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result.__name__ == "Int" + SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]

def test_make_seq_field_type_reduce_method():
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    reduce_result = result.__reduce__()
    assert reduce_result[0] == _restore_seq_field_pickle
    assert reduce_result[1] == (MockCheckedClass, item_type, [])

def test_make_seq_field_type_with_multiple_checked_types():
    class MockCheckedClass:
        _checked_types = (int, float)
    item_type = str
    item_invariant = None
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result.__name__ == "IntFloat" + SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]


# LLM-generated content at query #18
#--------------------------

def test_pfield_constructor():
    field_type = (int,)
    invariant = lambda x: x > 0
    initial = 1
    mandatory = True
    factory = lambda x: x
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
    initial = None
    mandatory = False
    factory = None
    serializer = None
    field = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert field.type == field_type
    assert field.invariant == invariant
    assert field.initial == initial
    assert field.mandatory == mandatory
    assert field._factory == factory
    assert field.serializer == serializer

def test_pfield_constructor_with_empty_type():
    field_type = ()
    invariant = None
    initial = None
    mandatory = False
    factory = None
    serializer = None
    field = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert field.type == field_type
    assert field.invariant == invariant
    assert field.initial == initial
    assert field.mandatory == mandatory
    assert field._factory == factory
    assert field.serializer == serializer

def test_pfield_constructor_with_multiple_types():
    field_type = (int, float)
    invariant = None
    initial = 0
    mandatory = True
    factory = None
    serializer = None
    field = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert field.type == field_type
    assert field.invariant == invariant
    assert field.initial == initial
    assert field.mandatory == mandatory
    assert field._factory == factory
    assert field.serializer == serializer

def test_pfield_constructor_with_complex_invariant():
    field_type = (list,)
    invariant = lambda lst: len(lst) > 0
    initial = []
    mandatory = False
    factory = list
    serializer = lambda lst: ','.join(map(str, lst))
    field = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert field.type == field_type
    assert field.invariant == invariant
    assert field.initial == initial
    assert field.mandatory == mandatory
    assert field._factory == factory
    assert field.serializer == serializer


# LLM-generated content at query #19
#--------------------------

def test_check_field_parameters_initial_invalid_type():
    from collections import namedtuple
    Field = namedtuple('Field', ['type', 'initial', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(type=[int], initial='not_an_int', invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


# LLM-generated content at query #20
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    class MockField:
        pass
    field = MockField()
    field.initial = None
    field.type = []
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result == False


# LLM-generated content at query #21
#--------------------------

```python
def test_pmap_field_factory_for_optional_true():
    from pyrsistent import pmap_field, optional_type, field
    from pyrsistent._checked_types import optional
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import get_type
    from pyrsistent._checked_types import maybe_parse_user_type
    from pyrsistent._checked_types import wrap_invariant
    from pyrsistent._checked_types import _merge_invariant_results
    from pyrsistent._checked_types import _get_class
    from pyrsistent._checked_types import maybe_parse_many_user_types
    from pyrsistent._checked_types import _preserved_iterable_types
    from pyrsistent import CheckedType
    from pyrsistent import PFIELD_NO_INVARIANT
    from pyrsistent import PFIELD_NO_FACTORY
    from pyrsistent._checked_types import Iterable
    import typing
    result = pmap_field(int, str, optional=True)
    assert isinstance(result, _PField)
    assert result.mandatory is True
    assert result.type == optional_type(result.type) if result.type is not None else None
    assert callable(result.factory)
    assert result.invariant == PFIELD_NO_INVARIANT
    test_value = None
    factory_result = result.factory(test_value)
    assert factory_result is None
    test_value = {1: "a"}
    factory_result = result.factory(test_value)
    assert factory_result is not None
    assert isinstance(factory_result, CheckedType)
    assert 1 in factory_result
    assert factory_result[1] == "a"


# LLM-generated content at query #22
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
    subject = 42
    invariants = [lambda x: (True, 0), lambda x: (True, 1), lambda x: (True, 2)]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #23
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 100), lambda x: (True, 1), lambda x: (False, 200)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100, 200)

def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)

def test_check_global_invariants_all_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 500), lambda x: (False, 600)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (500, 600)

def test_check_global_invariants_single_error():
    subject = "test_subject"
    invariants = [lambda x: (False, 999)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (999,)

def test_check_global_invariants_error_order():
    subject = "test_subject"
    invariants = [lambda x: (False, 300), lambda x: (False, 200), lambda x: (False, 100)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (300, 200, 100)


# LLM-generated content at query #24
#--------------------------

def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from unittest.mock import Mock, patch
    mock_checked_class = Mock()
    mock_item_type = Mock()
    test_data = [1, 2, 3]
    mock_type_ = Mock()
    with patch('pyrsistent._field_common._seq_field_types', {(mock_checked_class, mock_item_type): mock_type_}):
        with patch('pyrsistent._field_common._restore_pickle') as mock_restore_pickle:
            mock_restore_pickle.return_value = 'restored_value'
            result = _restore_seq_field_pickle(mock_checked_class, mock_item_type, test_data)
            mock_restore_pickle.assert_called_once_with(mock_type_, test_data)
            assert result == 'restored_value'


# LLM-generated content at query #25
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_exist():
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


# LLM-generated content at query #26
#--------------------------

def test_check_field_parameters_initial_invalid_type():
    from collections import namedtuple
    Field = namedtuple('Field', ['type', 'initial', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(type=[int], initial="not_an_int", invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


# LLM-generated content at query #27
#--------------------------

def test_set_fields_pfield_condition_true():
    class _PField:
        pass
    pfield_instance = _PField()
    dct = {'field1': pfield_instance, 'field2': 'normal_value'}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert 'field1' not in dct
    assert dct['test_name']['field1'] is pfield_instance


# LLM-generated content at query #28
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_exist():
    error_codes = ('error1', 'error2')
    mock_invariants = [lambda x: (False, 'error1'), lambda x: (False, 'error2')]
    subject = 'test_subject'
    try:
        check_global_invariants(subject, mock_invariants)
        assert False
    except InvariantException as e:
        assert e.args[0] == error_codes
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #29
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type

def test_make_pmap_field_type_returns_cached_class():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    key_type = int
    value_type = str
    first = _make_pmap_field_type(key_type, value_type)
    second = _make_pmap_field_type(key_type, value_type)
    assert first is second

def test_make_pmap_field_type_sets_correct_name():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"

def test_make_pmap_field_type_with_tuple_types():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    key_type = (int, str)
    value_type = (bool,)
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntStrToBoolPMap"

def test_make_pmap_field_type_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type
    from pyrsistent import CheckedPMap
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    instance = result({1: "a"})
    reduced = instance.__reduce__()
    assert reduced[0].__name__ == "_restore_pmap_field_pickle"
    assert reduced[1][0] == key_type
    assert reduced[1][1] == value_type
    assert reduced[1][2] == {1: "a"}


# LLM-generated content at query #30
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        pass
    field = MockField()
    field.type = [int, str]
    field.initial = "test"
    field.invariant = lambda x: True
    field.factory = lambda x: x
    field.serializer = lambda x: x
    result = not isinstance(field.type[0], type) and not isinstance(field.type[0], str)
    assert result == False


# LLM-generated content at query #31
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_type_check_fails():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from unittest.mock import Mock
    field = Mock()
    field.type = "some_type"
    field.factory = Mock()
    type_cls = Mock()
    result = is_field_ignore_extra_complaint(type_cls, field, True)
    assert result is False


# LLM-generated content at query #32
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
    subject = 42
    invariants = [lambda x: (True, 0), lambda x: (True, 1), lambda x: (True, 2)]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #33
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
    first = _make_pmap_field_type(key_type, value_type)
    second = _make_pmap_field_type(key_type, value_type)
    assert first is second
    assert _pmap_field_types[(key_type, value_type)] is first

def test_make_pmap_field_type_with_custom_class_name():
    from pyrsistent import CheckedPMap
    from pyrsistent._field_common import _make_pmap_field_type
    class CustomKey:
        pass
    class CustomValue:
        pass
    result = _make_pmap_field_type(CustomKey, CustomValue)
    assert result.__name__ == "CustomkeyToCustomvaluePMap"

def test_make_pmap_field_type_reduce_method():
    from pyrsistent._field_common import _make_pmap_field_type, _restore_pmap_field_pickle
    key_type = bool
    value_type = float
    cls = _make_pmap_field_type(key_type, value_type)
    instance = cls({True: 1.5, False: 2.5})
    reduce_result = instance.__reduce__()
    assert reduce_result[0] == _restore_pmap_field_pickle
    assert reduce_result[1][0] == key_type
    assert reduce_result[1][1] == value_type
    assert dict(reduce_result[1][2]) == {True: 1.5, False: 2.5}


# LLM-generated content at query #34
#--------------------------

def test_set_fields_pfield_assignment():
    class _PField:
        pass

    class Base1:
        pass

    class Base2:
        pass

    pfield_instance = _PField()
    Base1.__dict__['test_name'] = {'key1': 'value1'}
    Base2.__dict__['test_name'] = {'key2': 'value2'}
    dct = {'extra_key': pfield_instance}
    bases = (Base1, Base2)
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct['test_name']['extra_key'] is pfield_instance


# LLM-generated content at query #35
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from unittest.mock import Mock
    PFIELD_NO_INITIAL = object()
    field = Mock()
    field.initial = PFIELD_NO_INITIAL
    field.type = []
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result is False
    field.initial = lambda: None
    field.type = [int]
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result is False
    field.initial = 5
    field.type = []
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result is False
    field.initial = 5
    field.type = [int]
    result = field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type)
    assert result is False


# LLM-generated content at query #36
#--------------------------

def test_pfield_constructor():
    test_type = (int,)
    test_invariant = lambda x: x > 0
    test_initial = 1
    test_mandatory = True
    test_factory = int
    test_serializer = lambda x: str(x)
    field = _PField(test_type, test_invariant, test_initial, test_mandatory, test_factory, test_serializer)
    assert field.type == test_type
    assert field.invariant == test_invariant
    assert field.initial == test_initial
    assert field.mandatory == test_mandatory
    assert field._factory == test_factory
    assert field.serializer == test_serializer


# LLM-generated content at query #37
#--------------------------

def test_factory_property_with_checked_type():
    PFIELD_NO_FACTORY = object()
    class CheckedType:
        @classmethod
        def create(cls):
            return "checked_type_factory"
    class MockCheckedType(CheckedType):
        pass
    from pyrsistent._checked_types import get_type
    original_get_type = get_type
    get_type = lambda x: MockCheckedType
    field = _PField(type=(MockCheckedType,), invariant=None, initial=None, mandatory=True, factory=PFIELD_NO_FACTORY, serializer=None)
    result = field.factory
    get_type = original_get_type
    assert result == MockCheckedType.create


# LLM-generated content at query #38
#--------------------------

def test_set_fields_pfield_condition_true():
    class _PField:
        pass

    pfield_instance = _PField()
    dct = {'key1': pfield_instance, 'key2': 'not_pfield'}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert 'test_name' in dct
    assert isinstance(dct['test_name'], dict)
    assert 'key1' in dct['test_name']
    assert dct['test_name']['key1'] is pfield_instance
    assert 'key1' not in dct
    assert 'key2' in dct


# LLM-generated content at query #39
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_present():
    error_codes = ('ERROR_1', 'ERROR_2')
    mock_invariant = lambda s: (False, error_codes[0])
    mock_invariant2 = lambda s: (False, error_codes[1])
    invariants = [mock_invariant, mock_invariant2]
    subject = None
    try:
        check_global_invariants(subject, invariants)
        raised = False
    except InvariantException as e:
        raised = True
        assert e.error_codes == error_codes
    assert raised

def test_check_global_invariants_no_exception_when_no_error_codes():
    mock_invariant = lambda s: (True, None)
    invariants = [mock_invariant]
    subject = None
    check_global_invariants(subject, invariants)


