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
    subject = 123
    invariants = [lambda x: (True, 0), lambda x: (False, 100)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)

def test_check_global_invariants_multiple_errors():
    subject = []
    invariants = [lambda x: (False, 5), lambda x: (True, 10), lambda x: (False, 15)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {5, 15}

def test_check_global_invariants_all_errors():
    subject = {}
    invariants = [lambda x: (False, 1), lambda x: (False, 2), lambda x: (False, 3)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert len(e.error_codes) == 3
        assert sorted(e.error_codes) == [1, 2, 3]

def test_check_global_invariants_empty_invariants():
    subject = None
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #2
#--------------------------

def test_set_fields_adds_field_to_dct():
    class MockBase1:
        field1 = {"a": 1}
    class MockBase2:
        field2 = {"b": 2}
    dct = {}
    bases = (MockBase1, MockBase2)
    name = "test_field"
    set_fields(dct, bases, name)
    expected = {"a": 1, "b": 2}
    assert dct[name] == expected

def test_set_fields_handles_empty_bases():
    dct = {}
    bases = ()
    name = "test_field"
    set_fields(dct, bases, name)
    assert dct[name] == {}

def test_set_fields_handles_missing_field_in_bases():
    class MockBase:
        pass
    dct = {}
    bases = (MockBase,)
    name = "test_field"
    set_fields(dct, bases, name)
    assert dct[name] == {}

def test_set_fields_moves_pfield_instances():
    class _PField:
        pass
    pfield_instance = _PField()
    dct = {"key1": pfield_instance, "key2": "normal_value"}
    bases = ()
    name = "test_field"
    set_fields(dct, bases, name)
    assert dct[name]["key1"] is pfield_instance
    assert "key1" not in dct
    assert dct["key2"] == "normal_value"
    assert "key2" not in dct[name]

def test_set_fields_preserves_non_pfield_entries():
    dct = {"key1": "value1", "key2": 42}
    bases = ()
    name = "test_field"
    set_fields(dct, bases, name)
    assert dct["key1"] == "value1"
    assert dct["key2"] == 42
    assert dct[name] == {}

def test_set_fields_merges_duplicate_keys_from_bases():
    class MockBase1:
        field = {"a": 1, "b": 2}
    class MockBase2:
        field = {"b": 99, "c": 3}
    dct = {}
    bases = (MockBase1, MockBase2)
    name = "field"
    set_fields(dct, bases, name)
    expected = {"a": 1, "b": 99, "c": 3}
    assert dct[name] == expected


# LLM-generated content at query #3
#--------------------------

def test_make_seq_field_type_creates_new_type():
    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = int
    mock_item_invariant = lambda x: x > 0
    _seq_field_types = {}
    SEQ_FIELD_TYPE_SUFFIXES = {mock_checked_class: 'Suffix'}
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    assert result.__type__ == mock_item_type
    assert result.__invariant__ == mock_item_invariant
    assert result.__bases__ == (mock_checked_class,)
    assert (mock_checked_class, mock_item_type) in _seq_field_types
    assert _seq_field_types[(mock_checked_class, mock_item_type)] is result

def test_make_seq_field_type_returns_cached_type():
    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = str
    mock_item_invariant = None
    _seq_field_types = {(mock_checked_class, mock_item_type): 'cached_type'}
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    assert result == 'cached_type'

def test_make_seq_field_type_sets_name_using_types_to_names():
    mock_checked_class = type('MockCheckedClass', (), {'_checked_types': (int, str)})
    mock_item_type = float
    mock_item_invariant = None
    SEQ_FIELD_TYPE_SUFFIXES = {mock_checked_class: 'Seq'}
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    assert result.__name__ == 'IntStrSeq'

def test_make_seq_field_type_reduce_method():
    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = list
    mock_item_invariant = lambda x: len(x) > 0
    SEQ_FIELD_TYPE_SUFFIXES = {mock_checked_class: 'Suffix'}
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    reduced = result.__reduce__()
    assert reduced[0] == _restore_seq_field_pickle
    assert reduced[1] == (mock_checked_class, mock_item_type, [])


# LLM-generated content at query #4
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


# LLM-generated content at query #5
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
        check_type(DestinationClass, MockField, "test_field", "not_an_int")
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
    check_type(DestinationClass, MockField, "test_field", "valid_string")

def test_check_type_no_type_restriction():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = None
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "test_field", 42)
    check_type(DestinationClass, MockField, "test_field", "any_value")
    check_type(DestinationClass, MockField, "test_field", [1, 2, 3])

def test_check_type_with_type_string():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ["builtins.int", "builtins.str"]
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "test_field", 42)
    check_type(DestinationClass, MockField, "test_field", "valid_string")

def test_check_type_with_mixed_type_and_string():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int, "builtins.str"]
    class DestinationClass:
        pass
    check_type(DestinationClass, MockField, "test_field", 42)
    check_type(DestinationClass, MockField, "test_field", "valid_string")


# LLM-generated content at query #6
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert result.__name__ == "IntToStrPMap"


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
    assert result.__name__ == "StrToIntPMap"


def test_make_pmap_field_type_with_tuple_types():
    key_type = (int, str)
    value_type = (bool,)
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert result.__name__ == "IntStrToBoolPMap"


def test_make_pmap_field_type_reduce_method():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    reduce_result = result.__reduce__()
    assert reduce_result[0] == _restore_pmap_field_pickle
    assert reduce_result[1][0] == key_type
    assert reduce_result[1][1] == value_type
    assert isinstance(reduce_result[1][2], dict)


# LLM-generated content at query #7
#--------------------------

def test_check_type_valid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int,)
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)

def test_check_type_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int,)
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "field_name", "not_an_int")
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
        type = (int, str)
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)
    check_type(DestinationCls, MockField, "field_name", "valid_string")

def test_check_type_multiple_types_invalid():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int, str)
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
        type = ("builtins.int",)
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)

def test_check_type_with_type_string_invalid():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ("builtins.int",)
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "field_name", "not_an_int")
        assert False
    except Exception as e:
        assert "Invalid type for field DestinationCls.field_name, was str" in str(e)


# LLM-generated content at query #8
#--------------------------

def test_check_field_parameters_initial_not_pfield_no_initial_not_callable_and_type_mismatch():
    from collections import namedtuple
    Field = namedtuple('Field', ['initial', 'type', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(initial=123, type=[str], invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'int\'>'

def test_check_field_parameters_initial_not_pfield_no_initial_not_callable_and_type_matches():
    from collections import namedtuple
    Field = namedtuple('Field', ['initial', 'type', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(initial="hello", type=[str], invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_initial_is_pfield_no_initial():
    from collections import namedtuple
    Field = namedtuple('Field', ['initial', 'type', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(initial=PFIELD_NO_INITIAL, type=[str], invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_initial_not_pfield_no_initial_and_callable():
    from collections import namedtuple
    Field = namedtuple('Field', ['initial', 'type', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(initial=lambda: 123, type=[int], invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_initial_not_pfield_no_initial_not_callable_and_empty_type():
    from collections import namedtuple
    Field = namedtuple('Field', ['initial', 'type', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(initial=123, type=[], invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_initial_not_pfield_no_initial_not_callable_and_multiple_types_one_matches():
    from collections import namedtuple
    Field = namedtuple('Field', ['initial', 'type', 'invariant', 'factory', 'serializer'])
    PFIELD_NO_INITIAL = object()
    field = Field(initial=123, type=[str, int], invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)


# LLM-generated content at query #9
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

def test_is_field_ignore_extra_complaint_with_ignore_extra_param():
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

def test_is_field_ignore_extra_complaint_type_set():
    class MockField:
        type = {int}
        factory = None
    field = MockField()
    result = is_field_ignore_extra_complaint(type, field, True)
    assert result is True


# LLM-generated content at query #10
#--------------------------

def test_sequence_field_creates_checked_vector_field():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, False, [])
    assert result.type == {CheckedPVector}
    assert result.mandatory is True
    assert result.initial == CheckedPVector.create([])

def test_sequence_field_creates_checked_set_field():
    from pyrsistent import CheckedPSet, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPSet, str, False, set())
    assert result.type == {CheckedPSet}
    assert result.mandatory is True
    assert result.initial == CheckedPSet.create(set())

def test_sequence_field_with_optional_true():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, True, None)
    assert result.type == {CheckedPVector, type(None)}
    assert result.mandatory is True
    assert result.initial is None

def test_sequence_field_with_initial_list():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    assert result.initial == CheckedPVector.create([1, 2, 3])

def test_sequence_field_with_item_invariant():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    def positive(x):
        return x > 0
    result = _sequence_field(CheckedPVector, int, False, [], item_invariant=positive)
    assert result.type == {CheckedPVector}
    assert result.mandatory is True

def test_sequence_field_with_invariant():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    def non_empty(seq):
        return len(seq) > 0
    result = _sequence_field(CheckedPVector, int, False, [], invariant=non_empty)
    assert result.type == {CheckedPVector}
    assert result.mandatory is True

def test_sequence_field_optional_factory_handles_none():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, True, None)
    assert result.factory(None) is None

def test_sequence_field_optional_factory_creates_instance():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, True, None)
    instance = result.factory([1, 2])
    assert isinstance(instance, CheckedPVector)
    assert list(instance) == [1, 2]

def test_sequence_field_non_optional_factory_creates_instance():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result = _sequence_field(CheckedPVector, int, False, [])
    instance = result.factory([1, 2])
    assert isinstance(instance, CheckedPVector)
    assert list(instance) == [1, 2]

def test_sequence_field_caches_field_type():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    result1 = _sequence_field(CheckedPVector, int, False, [])
    result2 = _sequence_field(CheckedPVector, int, False, [])
    assert result1.type == result2.type


# LLM-generated content at query #11
#--------------------------

```python
def test_check_global_invariants_all_pass():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (True, 1), lambda s: (True, 2)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_one_fails():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (False, 1), lambda s: (True, 2)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (1,)

def test_check_global_invariants_multiple_fail():
    subject = "test_subject"
    invariants = [lambda s: (False, 0), lambda s: (True, 1), lambda s: (False, 2)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (0, 2)

def test_check_global_invariants_all_fail():
    subject = "test_subject"
    invariants = [lambda s: (False, 0), lambda s: (False, 1), lambda s: (False, 2)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (0, 1, 2)

def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)

def test_check_global_invariants_subject_passed_to_invariants():
    captured_subject = None
    def capturing_invariant(s):
        nonlocal captured_subject
        captured_subject = s
        return (True, 0)
    subject = "test_subject"
    invariants = [capturing_invariant]
    check_global_invariants(subject, invariants)
    assert captured_subject == subject


# LLM-generated content at query #12
#--------------------------

def test_is_type_cls_with_type_set():
    result = is_type_cls(type, {int, str})
    assert result is True

def test_is_type_cls_with_empty_set():
    result = is_type_cls(type, set())
    assert result is False

def test_is_type_cls_with_single_type():
    result = is_type_cls(type, (int,))
    assert result is True

def test_is_type_cls_with_multiple_types():
    result = is_type_cls(type, (int, str))
    assert result is True

def test_is_type_cls_with_non_type_first_element():
    result = is_type_cls(type, ("not_a_type",))
    assert result is False

def test_is_type_cls_with_subclass():
    class BaseClass:
        pass
    class DerivedClass(BaseClass):
        pass
    result = is_type_cls(BaseClass, (DerivedClass,))
    assert result is True

def test_is_type_cls_with_non_subclass():
    class ClassA:
        pass
    class ClassB:
        pass
    result = is_type_cls(ClassA, (ClassB,))
    assert result is False

def test_is_type_cls_with_string_type_name():
    result = is_type_cls(type, ("builtins.int",))
    assert result is True

def test_is_type_cls_with_empty_tuple():
    result = is_type_cls(type, ())
    assert result is False

def test_is_type_cls_with_type_cls_not_type():
    class CustomTypeClass:
        pass
    result = is_type_cls(CustomTypeClass, (int,))
    assert result is False


# LLM-generated content at query #13
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


# LLM-generated content at query #14
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


# LLM-generated content at query #15
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


# LLM-generated content at query #16
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
    value = "test_string"
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == PFIELD_NO_SERIALIZER("json", value)

def test_serialize_with_non_checked_type_and_custom_serializer():
    value = 123
    def custom_serializer(format, value):
        return f"serialized_{format}_{value}"
    result = serialize(custom_serializer, "yaml", value)
    assert result == "serialized_yaml_123"


# LLM-generated content at query #17
#--------------------------

def test_check_field_parameters_initial_invalid_type():
    field = type('Field', (), {'type': (int,), 'initial': 'not_an_int', 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    PFIELD_NO_INITIAL = object()
    field.initial is not PFIELD_NO_INITIAL
    not callable(field.initial)
    field.type
    any(isinstance(field.initial, t) for t in field.type)


# LLM-generated content at query #18
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (True, 1)]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    subject = "test_subject"
    invariants = [lambda s: (False, 100), lambda s: (True, 1)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)


def test_check_global_invariants_multiple_errors():
    subject = "test_subject"
    invariants = [lambda s: (False, 100), lambda s: (False, 200), lambda s: (False, 300)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100, 200, 300)


def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)


def test_check_global_invariants_mixed_errors_and_ok():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (False, 500), lambda s: (True, 2), lambda s: (False, 600)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (500, 600)


# LLM-generated content at query #19
#--------------------------

def test_make_seq_field_type_creates_new_type():
    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = int
    mock_item_invariant = lambda x: x > 0
    _seq_field_types.clear()
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    assert result.__type__ == mock_item_type
    assert result.__invariant__ == mock_item_invariant
    assert result.__bases__ == (mock_checked_class,)
    assert _seq_field_types[(mock_checked_class, mock_item_type)] == result

def test_make_seq_field_type_returns_cached_type():
    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = str
    mock_item_invariant = None
    _seq_field_types.clear()
    first_result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    second_result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    assert first_result is second_result

def test_make_seq_field_type_sets_correct_name():
    mock_checked_class = type('MockCheckedClass', (), {'_checked_types': (int, str)})
    mock_item_type = float
    mock_item_invariant = None
    _seq_field_types.clear()
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    expected_suffix = SEQ_FIELD_TYPE_SUFFIXES[mock_checked_class]
    assert result.__name__ == 'IntStr' + expected_suffix

def test_make_seq_field_type_reduce_method():
    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = list
    mock_item_invariant = None
    _seq_field_types.clear()
    result = _make_seq_field_type(mock_checked_class, mock_item_type, mock_item_invariant)
    reduce_result = result.__reduce__()
    assert reduce_result[0] == _restore_seq_field_pickle
    assert reduce_result[1] == (mock_checked_class, mock_item_type, [])


# LLM-generated content at query #20
#--------------------------

def test_make_seq_field_type_creates_subclass():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = None
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
    item_type = bytes
    item_invariant = lambda x: True
    first_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert first_call is second_call
    assert _seq_field_types[(MockCheckedClass, item_type)] is first_call

def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _restore_seq_field_pickle
    class MockCheckedClass:
        _checked_types = (list,)
    item_type = dict
    item_invariant = None
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    reduce_output = result.__reduce__()
    assert reduce_output[0] is _restore_seq_field_pickle
    assert reduce_output[1] == (MockCheckedClass, item_type, [])


# LLM-generated content at query #21
#--------------------------

def test_pmap_field_creates_checked_pmap_field():
    result = pmap_field(int, str)
    assert isinstance(result, _PField)
    assert result.mandatory is True
    assert isinstance(result.initial, CheckedPMap)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert callable(result.factory)
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional_true_allows_none():
    result = pmap_field(int, str, optional=True)
    assert isinstance(result, _PField)
    assert result.mandatory is True
    assert isinstance(result.initial, CheckedPMap)
    assert result.type == {optional_type(_make_pmap_field_type(int, str))}
    assert callable(result.factory)
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def custom_invariant(value):
        return True, ()
    result = pmap_field(int, str, invariant=custom_invariant)
    assert isinstance(result, _PField)
    assert result.invariant is not PFIELD_NO_INVARIANT
    assert callable(result.invariant)

def test_pmap_field_factory_with_optional_none():
    result = pmap_field(int, str, optional=True)
    assert result.factory(None) is None

def test_pmap_field_factory_without_optional():
    TheMap = _make_pmap_field_type(int, str)
    result = pmap_field(int, str)
    test_input = {1: "a"}
    assert isinstance(result.factory(test_input), TheMap)

def test_pmap_field_initial_is_checked_pmap():
    result = pmap_field(int, str)
    assert isinstance(result.initial, CheckedPMap)
    assert len(result.initial) == 0

def test_pmap_field_type_set_contains_one_element():
    result = pmap_field(int, str)
    assert len(result.type) == 1
    assert isinstance(next(iter(result.type)), type)

def test_pmap_field_optional_type_includes_none():
    result = pmap_field(int, str, optional=True)
    type_set = result.type
    type_obj = next(iter(type_set))
    assert type(None) in type_obj

def test_pmap_field_mandatory_is_true():
    result = pmap_field(int, str)
    assert result.mandatory is True
    result_optional = pmap_field(int, str, optional=True)
    assert result_optional.mandatory is True


# LLM-generated content at query #22
#--------------------------

def test_pfield_constructor():
    field_type = (int,)
    invariant = lambda x: x > 0
    initial = 1
    mandatory = True
    factory = lambda: 5
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

def test_pfield_constructor_with_no_factory():
    from pyrsistent import PFIELD_NO_FACTORY
    field_type = (int,)
    invariant = None
    initial = None
    mandatory = False
    factory = PFIELD_NO_FACTORY
    serializer = None
    field = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert field.type == field_type
    assert field.invariant == invariant
    assert field.initial == initial
    assert field.mandatory == mandatory
    assert field._factory == PFIELD_NO_FACTORY
    assert field.serializer == serializer


# LLM-generated content at query #23
#--------------------------

```python
def test_check_type_raises_ptype_error_when_value_type_not_in_field_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError

    class MockField:
        type = [int, str]

    class DestinationClass:
        pass

    value = 3.14
    field = MockField()
    name = "test_field"

    try:
        check_type(DestinationClass, field, name, value)
        assert False, "Expected PTypeError to be raised"
    except PTypeError as e:
        assert e.destination_cls == DestinationClass
        assert e.field_name == name
        assert e.expected_types == field.type
        assert e.actual_type == type(value)
        assert "Invalid type for field DestinationClass.test_field, was float" in str(e)


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

def test_sequence_field_creates_checked_type():
    from pyrsistent import CheckedPVector, optional
    result = _sequence_field(CheckedPVector, int, False, [])
    assert result.type == {CheckedPVector}
    assert result.mandatory is True
    assert result.initial == CheckedPVector.create([])

def test_sequence_field_with_optional():
    from pyrsistent import CheckedPVector, optional
    result = _sequence_field(CheckedPVector, str, True, None)
    assert result.type == {optional(CheckedPVector)}
    assert result.mandatory is True
    assert result.initial is None

def test_sequence_field_with_initial():
    from pyrsistent import CheckedPVector
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    assert result.initial == CheckedPVector.create([1, 2, 3])

def test_sequence_field_with_invariant():
    from pyrsistent import CheckedPVector
    def inv(value):
        return len(value) > 0, "Must not be empty"
    result = _sequence_field(CheckedPVector, int, False, [], invariant=inv)
    assert result.invariant is not None

def test_sequence_field_with_item_invariant():
    from pyrsistent import CheckedPVector
    def item_inv(value):
        return value > 0, "Must be positive"
    result = _sequence_field(CheckedPVector, int, False, [], item_invariant=item_inv)
    assert result.type == {CheckedPVector}

def test_sequence_field_factory_with_optional_none():
    from pyrsistent import CheckedPVector, optional
    result = _sequence_field(CheckedPVector, int, True, None)
    factory = result.factory
    assert factory(None) is None

def test_sequence_field_factory_with_optional_value():
    from pyrsistent import CheckedPVector, optional
    result = _sequence_field(CheckedPVector, int, True, None)
    factory = result.factory
    from pyrsistent import pvector
    value = factory([1, 2])
    assert isinstance(value, CheckedPVector)
    assert list(value) == [1, 2]

def test_sequence_field_factory_without_optional():
    from pyrsistent import CheckedPVector
    result = _sequence_field(CheckedPVector, int, False, [])
    factory = result.factory
    from pyrsistent import pvector
    value = factory([1, 2])
    assert isinstance(value, CheckedPVector)
    assert list(value) == [1, 2]


# LLM-generated content at query #26
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
    _seq_field_types[(MockCheckedClass, item_type)] = 'cached'
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result == 'cached'

def test_make_seq_field_type_sets_name():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, SEQ_FIELD_TYPE_SUFFIXES
    class MockCheckedClass:
        _checked_types = (int,)
        __name__ = 'MockCheckedClass'
    item_type = str
    item_invariant = lambda x: True
    suffix = SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    expected_name = 'Int' + suffix
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
    assert reduced[1][0] is MockCheckedClass
    assert reduced[1][1] is item_type
    assert reduced[1][2] == [1, 2, 3]


# LLM-generated content at query #27
#--------------------------

def test_check_type_valid():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int]
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 5)

def test_check_type_invalid():
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
        assert "Invalid type for field" in str(e)

def test_check_type_multiple_valid():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = [int, str]
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", "string")
    check_type(DestinationCls, MockField, "field_name", 5)

def test_check_type_no_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = None
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", object())

def test_check_type_type_string():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ["builtins.int"]
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 5)


# LLM-generated content at query #28
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = lambda x: True
        factory = lambda x: x
        serializer = lambda x: x
    result = _check_field_parameters(MockField)
    assert result is None


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        def __init__(self, type_list, initial, invariant, factory, serializer):
            self.type = type_list
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    PFIELD_NO_INITIAL = object()
    field_instance = MockField(type_list=[int, str], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field_instance)


# LLM-generated content at query #31
#--------------------------

def test_set_fields_adds_name_dict_with_base_items():
    class Base1:
        field1 = "value1"
    class Base2:
        field2 = "value2"
    dct = {}
    bases = (Base1, Base2)
    name = "test_fields"
    set_fields(dct, bases, name)
    assert dct[name] == {"field1": "value1", "field2": "value2"}

def test_set_fields_moves_pfield_instances_to_name_dict():
    class _PField:
        pass
    pfield_instance = _PField()
    dct = {"custom_field": pfield_instance}
    bases = ()
    name = "fields"
    set_fields(dct, bases, name)
    assert dct["fields"] == {"custom_field": pfield_instance}
    assert "custom_field" not in dct

def test_set_fields_handles_empty_bases():
    dct = {}
    bases = ()
    name = "fields"
    set_fields(dct, bases, name)
    assert dct[name] == {}

def test_set_fields_merges_duplicate_keys_from_bases():
    class Base1:
        key = "value1"
    class Base2:
        key = "value2"
    dct = {}
    bases = (Base1, Base2)
    name = "merged"
    set_fields(dct, bases, name)
    assert dct[name] == {"key": "value1", "key": "value2"}

def test_set_fields_ignores_non_pfield_items_in_dct():
    dct = {"regular_field": "regular_value", "pfield": _PField()}
    bases = ()
    name = "fields"
    set_fields(dct, bases, name)
    assert dct["fields"] == {"pfield": dct["fields"]["pfield"]}
    assert "pfield" not in dct
    assert dct["regular_field"] == "regular_value"

def test_set_fields_with_mixed_base_dict_and_dct_pfields():
    class Base:
        base_field = "base_value"
    pfield_instance = _PField()
    dct = {"dct_field": pfield_instance}
    bases = (Base,)
    name = "collected"
    set_fields(dct, bases, name)
    assert dct["collected"] == {"base_field": "base_value", "dct_field": pfield_instance}
    assert "dct_field" not in dct


# LLM-generated content at query #32
#--------------------------

def test_set_fields_pfield_condition_true():
    class _PField:
        pass
    pfield_instance = _PField()
    dct = {'field1': pfield_instance, 'field2': 'normal'}
    bases = []
    name = 'meta'
    set_fields(dct, bases, name)
    assert dct == {'field2': 'normal', 'meta': {'field1': pfield_instance}}


# LLM-generated content at query #33
#--------------------------

def test_set_fields_pfield_condition_true():
    class _PField:
        pass
    pfield_instance = _PField()
    dct = {'key1': pfield_instance, 'key2': 'not_pfield'}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'key2': 'not_pfield', 'test_name': {'key1': pfield_instance}}


# LLM-generated content at query #34
#--------------------------

```python
def test_check_type_raises_ptype_error_when_value_type_not_in_field_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import PTypeError

    class MockField:
        type = [int, str]

    class DestinationClass:
        pass

    value = 3.14
    field = MockField()
    name = "test_field"
    destination_cls = DestinationClass

    try:
        check_type(destination_cls, field, name, value)
        assert False
    except PTypeError as e:
        assert e.destination_cls == destination_cls
        assert e.field_name == name
        assert e.expected_types == field.type
        assert e.actual_type == type(value)
        assert str(e) == "Invalid type for field DestinationClass.test_field, was float"


# LLM-generated content at query #35
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


# LLM-generated content at query #36
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
    invariants = [lambda s: (False, 200), lambda s: (True, 0), lambda s: (False, 300)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (200, 300)


def test_check_global_invariants_all_errors():
    subject = "test_subject"
    invariants = [lambda s: (False, 400), lambda s: (False, 500)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (400, 500)


def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #37
#--------------------------

def test_restore_seq_field_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types
    mock_checked_class = "TestClass"
    mock_item_type = "TestItem"
    mock_data = [1, 2, 3]
    mock_type = type("MockType", (), {"create": lambda self, data, _factory_fields: data})
    _seq_field_types[(mock_checked_class, mock_item_type)] = mock_type
    result = _restore_seq_field_pickle(mock_checked_class, mock_item_type, mock_data)
    assert result == mock_data


# LLM-generated content at query #38
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "IntToStrPMap" in result.__name__


def test_make_pmap_field_type_caches_and_returns_cached():
    key_type = str
    value_type = int
    first = _make_pmap_field_type(key_type, value_type)
    second = _make_pmap_field_type(key_type, value_type)
    assert first is second


def test_make_pmap_field_type_with_tuple_types():
    key_type = (int, str)
    value_type = (bool,)
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "IntStrToBoolPMap" in result.__name__


# LLM-generated content at query #39
#--------------------------

def test_sequence_field_creates_checked_vector_field():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    f = _sequence_field(CheckedPVector, int, False, [1, 2, 3])
    assert f.type == {CheckedPVector}
    assert f.mandatory == True
    assert f.initial == CheckedPVector.create([1, 2, 3])

def test_sequence_field_creates_checked_set_field():
    from pyrsistent import CheckedPSet, optional
    from pyrsistent._field_common import _sequence_field
    f = _sequence_field(CheckedPSet, str, False, ["a", "b"])
    assert f.type == {CheckedPSet}
    assert f.mandatory == True
    assert f.initial == CheckedPSet.create(["a", "b"])

def test_sequence_field_with_optional_true():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    f = _sequence_field(CheckedPVector, int, True, None)
    assert f.type == {CheckedPVector, type(None)}
    assert f.mandatory == True
    assert f.initial is None

def test_sequence_field_with_custom_invariant():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    def inv(val):
        return len(val) > 0, "must not be empty"
    f = _sequence_field(CheckedPVector, int, False, [1], invariant=inv)
    assert f.invariant is not None

def test_sequence_field_with_item_invariant():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    def item_inv(val):
        return val > 0, "must be positive"
    f = _sequence_field(CheckedPVector, int, False, [1], item_invariant=item_inv)
    assert f.type == {CheckedPVector}
    assert f.initial == CheckedPVector.create([1])

def test_sequence_field_factory_handles_none_for_optional():
    from pyrsistent import CheckedPVector, optional
    from pyrsistent._field_common import _sequence_field
    f = _sequence_field(CheckedPVector, int, True, None)
    result = f.factory(None)
    assert result is None

def test_sequence_field_factory_creates_instance_for_non_optional():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    f = _sequence_field(CheckedPVector, int, False, [])
    result = f.factory([1, 2, 3])
    assert isinstance(result, CheckedPVector)
    assert list(result) == [1, 2, 3]


# LLM-generated content at query #40
#--------------------------

def test_pfield_constructor():
    field_type = (int,)
    invariant = lambda x: x > 0
    initial = 10
    mandatory = True
    factory = lambda: 5
    serializer = lambda x: str(x)
    field = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert field.type == field_type
    assert field.invariant == invariant
    assert field.initial == initial
    assert field.mandatory == mandatory
    assert field._factory == factory
    assert field.serializer == serializer


# LLM-generated content at query #41
#--------------------------

def test_check_field_parameters_initial_invalid_type():
    field = type('Field', (), {'type': (int,), 'initial': 'not_an_int', 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'


# LLM-generated content at query #42
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


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

def test_pfield_constructor_with_pfield_no_factory():
    from pyrsistent import PFIELD_NO_FACTORY
    field_type = (int,)
    invariant = None
    initial = 0
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


# LLM-generated content at query #2
#--------------------------

def test_check_type_valid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int,)
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)

def test_check_type_invalid_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int,)
    class DestinationCls:
        pass
    try:
        check_type(DestinationCls, MockField, "field_name", "not_an_int")
        assert False
    except Exception as e:
        assert "Invalid type for field" in str(e)

def test_check_type_multiple_valid_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int, str)
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)
    check_type(DestinationCls, MockField, "field_name", "string")

def test_check_type_no_type_restriction():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = None
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)
    check_type(DestinationCls, MockField, "field_name", "string")
    check_type(DestinationCls, MockField, "field_name", [1, 2, 3])

def test_check_type_with_type_string():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = ("builtins.int", "builtins.str")
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)
    check_type(DestinationCls, MockField, "field_name", "string")

def test_check_type_with_mixed_type_and_string():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    class MockField:
        type = (int, "builtins.str")
    class DestinationCls:
        pass
    check_type(DestinationCls, MockField, "field_name", 42)
    check_type(DestinationCls, MockField, "field_name", "string")


# LLM-generated content at query #3
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

def test_check_field_parameters_non_callable_invariant():
    from pyrsistent import field
    invalid_field = field(invariant="not_callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_non_callable_factory():
    from pyrsistent import field
    invalid_field = field(factory="not_callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_non_callable_serializer():
    from pyrsistent import field
    invalid_field = field(serializer="not_callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"

def test_check_field_parameters_callable_initial_valid():
    from pyrsistent import field
    valid_field = field(type=int, initial=lambda: 10)
    _check_field_parameters(valid_field)

def test_check_field_parameters_no_initial_and_no_type():
    from pyrsistent import field
    valid_field = field()
    _check_field_parameters(valid_field)

def test_check_field_parameters_initial_matches_type():
    from pyrsistent import field
    valid_field = field(type=int, initial=5)
    _check_field_parameters(valid_field)

def test_check_field_parameters_initial_matches_one_of_multiple_types():
    from pyrsistent import field
    valid_field = field(type=[int, str], initial=5)
    _check_field_parameters(valid_field)

def test_check_field_parameters_initial_callable_with_type():
    from pyrsistent import field
    valid_field = field(type=int, initial=lambda: 42)
    _check_field_parameters(valid_field)


# LLM-generated content at query #4
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

def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    key = (MockCheckedClass, item_type)
    _seq_field_types.clear()
    result1 = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    result2 = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result1 is result2
    assert _seq_field_types[key] is result1

def test_make_seq_field_type_sets_name():
    from pyrsistent._field_common import _make_seq_field_type, SEQ_FIELD_TYPE_SUFFIXES
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    suffix = SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    expected_name = "Int" + suffix
    assert result.__name__ == expected_name

def test_make_seq_field_type_reduce_method():
    from pyrsistent._field_common import _make_seq_field_type, _restore_seq_field_pickle
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    reduce_result = result.__reduce__()
    assert reduce_result[0] is _restore_seq_field_pickle
    assert reduce_result[1][0] is MockCheckedClass
    assert reduce_result[1][1] is item_type


# LLM-generated content at query #5
#--------------------------

def test_sequence_field_creates_checked_type_with_item_type():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import get_type
    item_type = int
    field_obj = _sequence_field(CheckedPVector, item_type, optional=False, initial=[])
    assert field_obj.type == {get_type(CheckedPVector)}
    assert field_obj.mandatory is True
    assert field_obj.initial == CheckedPVector.create([])

def test_sequence_field_optional_none_allowed():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, str, optional=True, initial=None)
    result = field_obj.factory(None)
    assert result is None

def test_sequence_field_optional_factory_creates_instance():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, str, optional=True, initial=[])
    result = field_obj.factory(["a", "b"])
    assert isinstance(result, CheckedPVector)
    assert list(result) == ["a", "b"]

def test_sequence_field_non_optional_factory_creates_instance():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, int, optional=False, initial=[1, 2])
    result = field_obj.factory([3, 4])
    assert isinstance(result, CheckedPVector)
    assert list(result) == [3, 4]

def test_sequence_field_invariant_applied():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    def custom_invariant(value):
        return len(value) > 0, "must not be empty"
    field_obj = _sequence_field(CheckedPVector, int, optional=False, initial=[1], invariant=custom_invariant)
    verdict, data = field_obj.invariant(CheckedPVector.create([]))
    assert verdict is False
    verdict, data = field_obj.invariant(CheckedPVector.create([1]))
    assert verdict is True

def test_sequence_field_item_invariant_applied():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    def item_invariant(value):
        return value > 0, "must be positive"
    field_obj = _sequence_field(CheckedPVector, int, optional=False, initial=[1], item_invariant=item_invariant)
    TheType = field_obj.factory([1])
    assert TheType.__invariant__ is item_invariant

def test_sequence_field_initial_empty():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, str, optional=False, initial=[])
    assert field_obj.initial == CheckedPVector.create([])

def test_sequence_field_initial_with_values():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, float, optional=False, initial=[1.0, 2.0])
    assert list(field_obj.initial) == [1.0, 2.0]

def test_sequence_field_mandatory_true():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    field_obj = _sequence_field(CheckedPVector, bool, optional=False, initial=[])
    assert field_obj.mandatory is True

def test_sequence_field_type_set_correctly():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import get_type
    field_obj = _sequence_field(CheckedPVector, complex, optional=True, initial=None)
    expected_type = optional_type(get_type(CheckedPVector))
    assert field_obj.type == {expected_type}


# LLM-generated content at query #6
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (False, 100)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)

def test_check_global_invariants_multiple_errors():
    subject = "test"
    invariants = [lambda x: (False, 200), lambda x: (False, 300)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {200, 300}

def test_check_global_invariants_empty_invariants():
    subject = "test"
    invariants = []
    check_global_invariants(subject, invariants)

def test_check_global_invariants_all_true():
    subject = 42
    invariants = [lambda x: (True, 0), lambda x: (True, 1), lambda x: (True, 2)]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #7
#--------------------------

def test_check_field_parameters_valid_field():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_element():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int, 123), initial=5, invariant=lambda x: True, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial="string", invariant=lambda x: True, factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'str'>"

def test_check_field_parameters_no_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_callable_initial():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=lambda: 10, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_no_type():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(), initial=5, invariant=lambda x: True, factory=int, serializer=str)
    _check_field_parameters(field)

def test_check_field_parameters_non_callable_invariant():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant="not callable", factory=int, serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_non_callable_factory():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory="not callable", serializer=str)
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_non_callable_serializer():
    from pyrsistent import PField, PFIELD_NO_INITIAL
    field = PField(type=(int,), initial=5, invariant=lambda x: True, factory=int, serializer="not callable")
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #8
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda s: (True, 0), lambda s: (True, 1)]
    check_global_invariants(subject, invariants)


def test_check_global_invariants_single_error():
    subject = "test_subject"
    invariants = [lambda s: (False, 100)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (100,)


def test_check_global_invariants_multiple_errors():
    subject = "test_subject"
    invariants = [lambda s: (False, 200), lambda s: (True, 0), lambda s: (False, 300)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (200, 300)


def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #9
#--------------------------

def test_check_field_parameters_initial_invalid_type():
    field = type('Field', (), {'type': (int,), 'initial': 'not_an_int', 'invariant': lambda x: True, 'factory': lambda x: x, 'serializer': lambda x: x})()
    try:
        _check_field_parameters(field)
        assert False
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'


# LLM-generated content at query #10
#--------------------------

def test_pmap_field_creates_checked_pmap_with_key_and_value_types():
    f = pmap_field(int, str)
    assert f.type == {_make_pmap_field_type(int, str)}
    assert f.mandatory is True
    assert isinstance(f.initial, _make_pmap_field_type(int, str))
    assert f.factory == _make_pmap_field_type(int, str).create

def test_pmap_field_optional_true_allows_none():
    f = pmap_field(int, str, optional=True)
    assert f.type == {optional_type(_make_pmap_field_type(int, str))}
    assert f.mandatory is True
    assert isinstance(f.initial, _make_pmap_field_type(int, str))
    assert f.factory is not None
    result = f.factory(None)
    assert result is None

def test_pmap_field_invariant_is_wrapped():
    def custom_invariant(x):
        return True, ()
    f = pmap_field(int, str, invariant=custom_invariant)
    assert f.invariant != PFIELD_NO_INVARIANT
    assert callable(f.invariant)

def test_pmap_field_without_optional_does_not_allow_none():
    f = pmap_field(int, str, optional=False)
    assert f.type == {_make_pmap_field_type(int, str)}
    assert f.factory == _make_pmap_field_type(int, str).create

def test_pmap_field_initial_is_empty_pmap():
    f = pmap_field(int, str)
    assert isinstance(f.initial, CheckedPMap)
    assert len(f.initial) == 0

def test_pmap_field_mandatory_is_true():
    f = pmap_field(int, str)
    assert f.mandatory is True

def test_pmap_field_optional_factory_handles_none():
    f = pmap_field(int, str, optional=True)
    assert f.factory(None) is None

def test_pmap_field_optional_factory_creates_pmap():
    f = pmap_field(int, str, optional=True)
    result = f.factory({1: "a"})
    assert isinstance(result, CheckedPMap)
    assert result[1] == "a"

def test_pmap_field_non_optional_factory_creates_pmap():
    f = pmap_field(int, str, optional=False)
    result = f.factory({1: "a"})
    assert isinstance(result, CheckedPMap)
    assert result[1] == "a"

def test_pmap_field_serializer_default():
    f = pmap_field(int, str)
    assert f.serializer == PFIELD_NO_SERIALIZER

def test_pmap_field_check_field_parameters():
    f = pmap_field(int, str)
    _check_field_parameters(f)

def test_pmap_field_with_custom_invariant():
    def inv(x):
        return len(x) > 0, ()
    f = pmap_field(int, str, invariant=inv)
    assert f.invariant != PFIELD_NO_INVARIANT
    assert callable(f.invariant)


# LLM-generated content at query #11
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


# LLM-generated content at query #12
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


# LLM-generated content at query #13
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        def __init__(self, type_list, initial, invariant, factory, serializer):
            self.type = type_list
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    PFIELD_NO_INITIAL = object()
    field_instance = MockField(type_list=[int, str], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field_instance)


# LLM-generated content at query #14
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

def test_check_field_parameters_non_callable_invariant():
    from pyrsistent import field
    invalid_field = field(invariant="not_callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_non_callable_factory():
    from pyrsistent import field
    invalid_field = field(factory="not_callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_non_callable_serializer():
    from pyrsistent import field
    invalid_field = field(serializer="not_callable")
    try:
        _check_field_parameters(invalid_field)
        assert False
    except TypeError as e:
        assert str(e) == "Serializer must be callable"

def test_check_field_parameters_callable_initial():
    from pyrsistent import field
    valid_field = field(type=int, initial=lambda: 42)
    _check_field_parameters(valid_field)

def test_check_field_parameters_no_initial():
    from pyrsistent import field, PFIELD_NO_INITIAL
    valid_field = field(type=str, initial=PFIELD_NO_INITIAL)
    _check_field_parameters(valid_field)

def test_check_field_parameters_no_type():
    from pyrsistent import field
    valid_field = field(initial="test")
    _check_field_parameters(valid_field)

def test_check_field_parameters_initial_matches_type():
    from pyrsistent import field
    valid_field = field(type=[str, int], initial=100)
    _check_field_parameters(valid_field)


# LLM-generated content at query #15
#--------------------------

def test_check_field_parameters_initial_invalid_type():
    from unittest.mock import Mock
    field = Mock()
    field.initial = 42
    field.type = (int, str)
    field.invariant = lambda x: True
    field.factory = lambda x: x
    field.serializer = lambda x: x
    try:
        _check_field_parameters(field)
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'int\'>'


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
        return f"{format}:{value}"
    result = serialize(custom_serializer, "csv", value)
    assert result == "csv:123"

def test_serialize_with_none_value_and_no_serializer():
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result == PFIELD_NO_SERIALIZER("json", None)

def test_serialize_with_none_value_and_custom_serializer():
    def custom_serializer(format, value):
        return f"serialized_{value}"
    result = serialize(custom_serializer, "xml", None)
    assert result == "serialized_None"


# LLM-generated content at query #18
#--------------------------

def test_restore_pmap_field_pickle():
    from pyrsistent._field_common import _restore_pmap_field_pickle
    from pyrsistent._checked_types import _restore_pickle
    from unittest.mock import Mock, patch
    key_type = Mock()
    value_type = Mock()
    data = Mock()
    mock_pmap_field_types = {(key_type, value_type): Mock()}
    with patch('pyrsistent._field_common._pmap_field_types', mock_pmap_field_types):
        expected_result = Mock()
        with patch('pyrsistent._field_common._restore_pickle', return_value=expected_result) as mock_restore:
            result = _restore_pmap_field_pickle(key_type, value_type, data)
    mock_restore.assert_called_once_with(mock_pmap_field_types[(key_type, value_type)], data)
    assert result is expected_result


# LLM-generated content at query #19
#--------------------------

def test_pmap_field_factory_optional_none():
    from pyrsistent import pmap_field, optional
    from pyrsistent._checked_types import optional_type
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._checked_types import PFIELD_NO_FACTORY
    import pyrsistent

    key_type = str
    value_type = int
    optional = True
    field = pmap_field(key_type, value_type, optional)
    assert isinstance(field, _PField)
    assert field._factory is not PFIELD_NO_FACTORY
    assert field.factory is not PFIELD_NO_FACTORY
    assert field.factory(None) is None


# LLM-generated content at query #20
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
    invariants = [lambda x: (False, 5), lambda x: (True, 1), lambda x: (False, 3)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {5, 3}

def test_check_global_invariants_all_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, 10), lambda x: (False, 20)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert set(e.error_codes) == {10, 20}

def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)

def test_check_global_invariants_subject_passed_correctly():
    captured_subject = None
    def capturing_invariant(subj):
        nonlocal captured_subject
        captured_subject = subj
        return (True, 0)
    subject = "specific_subject"
    invariants = [capturing_invariant]
    check_global_invariants(subject, invariants)
    assert captured_subject == subject


# LLM-generated content at query #21
#--------------------------

def test_check_field_parameters_predicate_false():
    class MockField:
        def __init__(self, type_list, initial, invariant, factory, serializer):
            self.type = type_list
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    PFIELD_NO_INITIAL = object()
    field = MockField(type_list=[int, str], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    result = not isinstance(field.type[0], type) and not isinstance(field.type[0], str)
    assert result == False
    result = not isinstance(field.type[1], type) and not isinstance(field.type[1], str)
    assert result == False


# LLM-generated content at query #22
#--------------------------

def test_serialize_checked_type_with_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    PFIELD_NO_SERIALIZER = object()
    result = serialize(PFIELD_NO_SERIALIZER, "json", CheckedType())
    assert result == "serialized_json"

def test_serialize_checked_type_with_no_serializer_different_format():
    class CheckedType:
        def serialize(self, format):
            return f"checked_{format}"
    
    PFIELD_NO_SERIALIZER = None
    result = serialize(PFIELD_NO_SERIALIZER, "xml", CheckedType())
    assert result == "checked_xml"

def test_serialize_non_checked_type_with_no_serializer():
    PFIELD_NO_SERIALIZER = object()
    mock_serializer = lambda fmt, val: f"serialized_{fmt}_{val}"
    result = serialize(mock_serializer, "json", "test_value")
    assert result == "serialized_json_test_value"

def test_serialize_checked_type_with_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"checked_{format}"
    
    mock_serializer = lambda fmt, val: f"custom_{fmt}_{val.__class__.__name__}"
    result = serialize(mock_serializer, "json", CheckedType())
    assert result == "custom_json_CheckedType"

def test_serialize_non_checked_type_with_serializer():
    mock_serializer = lambda fmt, val: f"serialized_{fmt}_{val}"
    result = serialize(mock_serializer, "yaml", 42)
    assert result == "serialized_yaml_42"


# LLM-generated content at query #23
#--------------------------

def test_is_field_ignore_extra_complaint_false_when_ignore_extra_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from unittest.mock import Mock
    field = Mock()
    result = is_field_ignore_extra_complaint(Mock(), field, False)
    assert result is False

def test_is_field_ignore_extra_complaint_false_when_not_type_cls():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from unittest.mock import Mock
    field = Mock()
    field.type = Mock()
    type_cls = Mock()
    result = is_field_ignore_extra_complaint(type_cls, field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_false_when_no_ignore_extra_param():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from unittest.mock import Mock
    import inspect
    field = Mock()
    field.type = Mock()
    field.factory = Mock()
    signature = Mock()
    signature.parameters = {}
    field.factory.signature = signature
    type_cls = Mock()
    result = is_field_ignore_extra_complaint(type_cls, field, True)
    assert result is False

def test_is_field_ignore_extra_complaint_true_when_all_conditions_met():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from unittest.mock import Mock
    import inspect
    field = Mock()
    field.type = Mock()
    field.factory = Mock()
    signature = Mock()
    signature.parameters = {'ignore_extra': Mock()}
    field.factory.signature = signature
    type_cls = Mock()
    result = is_field_ignore_extra_complaint(type_cls, field, True)
    assert result is True


# LLM-generated content at query #24
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
    _seq_field_types.clear()
    first = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert first is second

def test_make_seq_field_type_sets_name():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, SEQ_FIELD_TYPE_SUFFIXES
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    _seq_field_types.clear()
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    expected_suffix = SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]
    assert result.__name__ == "Int" + expected_suffix

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


# LLM-generated content at query #25
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_type_check_fails():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from unittest.mock import Mock
    mock_field = Mock()
    mock_field.type = "some_type"
    mock_field.factory = Mock()
    type_cls = object()
    result = is_field_ignore_extra_complaint(type_cls, mock_field, True)
    assert result is False


# LLM-generated content at query #26
#--------------------------

def test_pfield_constructor():
    field_type = (int,)
    invariant = lambda x: x > 0
    initial = 1
    mandatory = True
    factory = lambda x: x
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
    initial = 0
    mandatory = True
    factory = lambda x: x
    serializer = lambda x: x
    pfield = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert pfield.type == field_type
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer

def test_pfield_constructor_with_multiple_types():
    field_type = (int, float)
    invariant = lambda x: isinstance(x, (int, float))
    initial = 0.0
    mandatory = False
    factory = lambda x: float(x)
    serializer = lambda x: repr(x)
    pfield = _PField(field_type, invariant, initial, mandatory, factory, serializer)
    assert pfield.type == field_type
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #27
#--------------------------

def test_is_type_cls_with_type_set():
    result = is_type_cls(type, {int, str})
    assert result is True

def test_is_type_cls_with_empty_set():
    result = is_type_cls(type, set())
    assert result is False

def test_is_type_cls_with_single_type_tuple():
    result = is_type_cls(type, (int,))
    assert result is True

def test_is_type_cls_with_multiple_type_tuple():
    result = is_type_cls(type, (int, str))
    assert result is True

def test_is_type_cls_with_non_matching_type_cls():
    result = is_type_cls(str, (int,))
    assert result is False

def test_is_type_cls_with_matching_subclass():
    result = is_type_cls(object, (int,))
    assert result is True

def test_is_type_cls_with_string_type_name():
    result = is_type_cls(type, ('builtins.int',))
    assert result is True

def test_is_type_cls_with_string_type_name_and_subclass():
    result = is_type_cls(object, ('builtins.int',))
    assert result is True

def test_is_type_cls_with_mixed_string_and_type():
    result = is_type_cls(type, ('builtins.int', str))
    assert result is True

def test_is_type_cls_with_empty_tuple():
    result = is_type_cls(type, ())
    assert result is False


# LLM-generated content at query #28
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


# LLM-generated content at query #29
#--------------------------

def test_check_global_invariants_raises_exception_when_error_codes_present():
    class InvariantException(Exception):
        def __init__(self, error_codes, _, message):
            self.error_codes = error_codes
            self.message = message
            super().__init__(message)

    def invariant1(subject):
        return (False, 1001)
    def invariant2(subject):
        return (True, 1002)
    def invariant3(subject):
        return (False, 1003)

    invariants = [invariant1, invariant2, invariant3]
    subject = "test_subject"

    try:
        error_codes = tuple(error_code for is_ok, error_code in
                            (invariant(subject) for invariant in invariants) if not is_ok)
        if error_codes:
            raise InvariantException(error_codes, (), 'Global invariant failed')
    except InvariantException as e:
        assert e.error_codes == (1001, 1003)
        assert e.message == 'Global invariant failed'
    else:
        assert False, "Expected InvariantException to be raised"


# LLM-generated content at query #30
#--------------------------

def test_sequence_field_with_optional_true():
    from pyrsistent import CheckedPVector, optional_type
    from pyrsistent._field_common import _sequence_field
    from pyrsistent._checked_types import optional
    item_type = int
    optional = True
    initial = [1, 2, 3]
    field_result = _sequence_field(CheckedPVector, item_type, optional, initial)
    assert field_result.type == optional_type(field_result._factory.__self__)
    assert field_result.mandatory == True
    assert field_result.initial == field_result._factory(initial)

def test_sequence_field_with_optional_false():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    item_type = str
    optional = False
    initial = ["a", "b"]
    field_result = _sequence_field(CheckedPVector, item_type, optional, initial)
    assert field_result.type == {field_result._factory.__self__}
    assert field_result.mandatory == True
    assert field_result.initial == field_result._factory(initial)

def test_sequence_field_with_none_initial_and_optional_true():
    from pyrsistent import CheckedPSet, optional_type
    from pyrsistent._field_common import _sequence_field
    item_type = float
    optional = True
    initial = None
    field_result = _sequence_field(CheckedPSet, item_type, optional, initial)
    assert field_result.type == optional_type(field_result._factory.__self__)
    assert field_result.mandatory == True
    assert field_result.initial == field_result._factory(initial)

def test_sequence_field_with_item_invariant():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    item_type = int
    optional = False
    initial = []
    item_invariant = lambda x: (x > 0, "positive")
    field_result = _sequence_field(CheckedPVector, item_type, optional, initial, item_invariant=item_invariant)
    assert field_result.type == {field_result._factory.__self__}
    assert field_result.mandatory == True

def test_sequence_field_with_invariant():
    from pyrsistent import CheckedPSet
    from pyrsistent._field_common import _sequence_field
    item_type = str
    optional = True
    initial = None
    invariant = lambda x: (x is None or len(x) > 0, "non-empty")
    field_result = _sequence_field(CheckedPSet, item_type, optional, initial, invariant=invariant)
    assert field_result.type == optional_type(field_result._factory.__self__)
    assert field_result.mandatory == True
    assert field_result.invariant != lambda x: (True, ())

def test_sequence_field_factory_with_optional_true_and_none():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    item_type = int
    optional = True
    initial = None
    field_result = _sequence_field(CheckedPVector, item_type, optional, initial)
    factory = field_result._factory
    result = factory(None)
    assert result is None

def test_sequence_field_factory_with_optional_true_and_non_none():
    from pyrsistent import CheckedPVector
    from pyrsistent._field_common import _sequence_field
    item_type = int
    optional = True
    initial = [1, 2]
    field_result = _sequence_field(CheckedPVector, item_type, optional, initial)
    factory = field_result._factory
    result = factory([3, 4])
    assert isinstance(result, CheckedPVector)
    assert list(result) == [3, 4]

def test_sequence_field_factory_with_optional_false():
    from pyrsistent import CheckedPSet
    from pyrsistent._field_common import _sequence_field
    item_type = str
    optional = False
    initial = ["x"]
    field_result = _sequence_field(CheckedPSet, item_type, optional, initial)
    factory = field_result._factory
    result = factory(["y", "z"])
    assert isinstance(result, CheckedPSet)
    assert set(result) == {"y", "z"}


# LLM-generated content at query #31
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

def test_make_seq_field_type_caches_result():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    key = (MockCheckedClass, item_type)
    _seq_field_types.clear()
    first = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert first is second
    assert _seq_field_types[key] is first

def test_make_seq_field_type_sets_name():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, SEQ_FIELD_TYPE_SUFFIXES
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    suffix = SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    expected_name = "Int" + suffix
    assert result.__name__ == expected_name

def test_make_seq_field_type_has_reduce():
    from pyrsistent._field_common import _make_seq_field_type, _restore_seq_field_pickle
    class MockCheckedClass:
        _checked_types = (int,)
    item_type = str
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    reduce_result = result.__reduce__()
    assert reduce_result[0] is _restore_seq_field_pickle
    assert reduce_result[1][0] is MockCheckedClass
    assert reduce_result[1][1] is item_type


# LLM-generated content at query #32
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
            actual = _restore_seq_field_pickle(mock_checked_class, mock_item_type, mock_data)
            mock_restore.assert_called_once_with(mock_type, mock_data)
            assert actual is mock_result


# LLM-generated content at query #33
#--------------------------

```python
def test_pmap_field_factory_for_optional_true():
    from pyrsistent import pmap_field, optional_type
    from pyrsistent._checked_types import optional
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import get_type
    from pyrsistent._checked_types import maybe_parse_user_type
    from pyrsistent._checked_types import wrap_invariant
    from pyrsistent._checked_types import _merge_invariant_results
    from pyrsistent._checked_types import maybe_parse_many_user_types
    from pyrsistent._checked_types import _get_class
    from pyrsistent._checked_types import _preserved_iterable_types
    from pyrsistent._checked_types import Iterable
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._checked_types import PFIELD_NO_FACTORY
    from pyrsistent._checked_types import PFIELD_NO_INVARIANT
    from pyrsistent._checked_types import field
    from pyrsistent._checked_types import _make_pmap_field_type
    from pyrsistent._checked_types import optional_type
    from pyrsistent._checked_types import TheMap
    from pyrsistent._checked_types import pmap_field
    key_type = str
    value_type = int
    optional = True
    invariant = PFIELD_NO_INVARIANT
    result = pmap_field(key_type, value_type, optional, invariant)
    assert isinstance(result, _PField)
    assert result.mandatory == True
    assert result.initial == TheMap()
    assert result.type == optional_type(TheMap)
    assert result.invariant == invariant
    assert result._factory is not PFIELD_NO_FACTORY
    assert callable(result._factory)
    assert result._factory(None) is None
    assert isinstance(result._factory({}), TheMap)


# LLM-generated content at query #34
#--------------------------

def test_check_field_parameters_initial_invalid_type():
    from dataclasses import dataclass
    from typing import Any
    PFIELD_NO_INITIAL = object()
    @dataclass
    class Field:
        type: Any
        initial: Any
        invariant: Any
        factory: Any
        serializer: Any
    field = Field(type=(int,), initial="not_an_int", invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    field.initial = "not_an_int"
    field.type = (int,)
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)


# LLM-generated content at query #35
#--------------------------

def test_make_pmap_field_type_creates_new_class():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "IntToStrPMap" in result.__name__
    assert issubclass(result, CheckedPMap)

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

def test_make_pmap_field_type_with_tuple_types():
    key_type = (int, str)
    value_type = (bool,)
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert "IntStrToBoolPMap" in result.__name__

def test_make_pmap_field_type_reduce_method():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    instance = result({1: "a"})
    reduced = instance.__reduce__()
    assert reduced[0] is _restore_pmap_field_pickle
    assert reduced[1][0] == key_type
    assert reduced[1][1] == value_type
    assert reduced[1][2] == {1: "a"}


# LLM-generated content at query #36
#--------------------------

def test_check_global_invariants_no_errors():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (True, 1)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = "test"
    invariants = [lambda x: (True, 0), lambda x: (False, 2)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (2,)

def test_check_global_invariants_multiple_errors():
    subject = "test"
    invariants = [lambda x: (False, 1), lambda x: (False, 3), lambda x: (True, 0)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (1, 3)

def test_check_global_invariants_all_errors():
    subject = "test"
    invariants = [lambda x: (False, 5), lambda x: (False, 6)]
    try:
        check_global_invariants(subject, invariants)
        assert False
    except InvariantException as e:
        assert e.error_codes == (5, 6)

def test_check_global_invariants_empty_invariants():
    subject = "test"
    invariants = []
    check_global_invariants(subject, invariants)

def test_check_global_invariants_subject_passed_correctly():
    captured_subject = None
    def invariant(subj):
        nonlocal captured_subject
        captured_subject = subj
        return (True, 0)
    subject = "specific"
    invariants = [invariant]
    check_global_invariants(subject, invariants)
    assert captured_subject == "specific"


# LLM-generated content at query #37
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
            actual_result = _restore_seq_field_pickle(mock_checked_class, mock_item_type, mock_data)
            mock_restore.assert_called_once_with(mock_type, mock_data)
            assert actual_result is mock_result


# LLM-generated content at query #38
#--------------------------

def test_serialize_checked_type_with_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"

    PFIELD_NO_SERIALIZER = object()
    value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == "serialized_json"


