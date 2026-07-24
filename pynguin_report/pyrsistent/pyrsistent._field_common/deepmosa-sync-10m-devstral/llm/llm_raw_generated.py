####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = object()
    invariants = [lambda _: (False, "ERROR1"), lambda _: (True, None)]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'

def test_check_global_invariants_multiple_errors():
    subject = object()
    invariants = [lambda _: (False, "ERROR1"), lambda _: (False, "ERROR2")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR1", "ERROR2")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #2
#--------------------------

```python
def test_restore_seq_field_pickle_calls_restore_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types

    mock_checked_class = type('MockCheckedClass', (), {})
    mock_item_type = type('MockItemType', (), {})
    mock_type = type('MockType', (), {'create': lambda self, data, _factory_fields: data})
    mock_data = [1, 2, 3]

    _seq_field_types[mock_checked_class, mock_item_type] = mock_type
    _restore_pickle(mock_type, mock_data)

    assert _restore_seq_field_pickle(mock_checked_class, mock_item_type, mock_data) == mock_data


# LLM-generated content at query #3
#--------------------------

```python
def test__make_seq_field_type_creates_new_type_with_correct_name():
    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert result.__name__ == "IntSeq"

def test__make_seq_field_type_reuses_existing_type():
    class MockCheckedClass:
        _checked_types = (int,)

    type1 = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    type2 = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert type1 is type2

def test__make_seq_field_type_sets_correct_attributes():
    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert result.__type__ == int
    assert result.__invariant__(5) is True

def test__make_seq_field_type_creates_reduce_method():
    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    instance = result([1, 2, 3])
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_seq_field_pickle
    assert reduced[1][0] == MockCheckedClass
    assert reduced[1][1] == int
    assert reduced[1][2] == [1, 2, 3]


# LLM-generated content at query #4
#--------------------------

```python
def test_types_to_names_with_single_type():
    assert _types_to_names((int,)) == "Int"

def test_types_to_names_with_multiple_types():
    assert _types_to_names((int, str, list)) == "IntStrList"

def test_types_to_names_with_string_type_name():
    assert _types_to_names(("builtins.int", "builtins.str")) == "IntStr"

def test_types_to_names_with_mixed_type_and_string():
    assert _types_to_names((int, "builtins.str")) == "IntStr"

def test_types_to_names_empty_tuple():
    assert _types_to_names(()) == ""


# LLM-generated content at query #5
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    assert is_field_ignore_extra_complaint(str, None, False) is False

def test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_subclass():
    class Field:
        type = int
        factory = lambda: None
    assert is_field_ignore_extra_complaint(str, Field(), True) is False

def test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param():
    class Field:
        type = str
        factory = lambda: None
    assert is_field_ignore_extra_complaint(str, Field(), True) is False

def test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_met():
    class Field:
        type = str
        factory = lambda ignore_extra: None
    assert is_field_ignore_extra_complaint(str, Field(), True) is True

def test_is_field_ignore_extra_complaint_works_with_set_type():
    class Field:
        type = {str}
        factory = lambda ignore_extra: None
    assert is_field_ignore_extra_complaint(str, Field(), True) is True

def test_is_field_ignore_extra_complaint_works_with_empty_type_tuple():
    class Field:
        type = ()
        factory = lambda ignore_extra: None
    assert is_field_ignore_extra_complaint(str, Field(), True) is False


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    checked_value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"serialized_{value}"

    result = serialize(custom_serializer, "format", "test_value")
    assert result == "serialized_test_value"


# LLM-generated content at query #7
#--------------------------

```python
def test__make_seq_field_type_creates_new_type():
    class TestCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(TestCheckedClass, int, None)
    assert isinstance(result, type)
    assert issubclass(result, TestCheckedClass)
    assert result.__name__ == "Int" + SEQ_FIELD_TYPE_SUFFIXES[TestCheckedClass]

def test__make_seq_field_type_reuses_existing_type():
    class TestCheckedClass:
        _checked_types = (int,)

    first_call = _make_seq_field_type(TestCheckedClass, int, None)
    second_call = _make_seq_field_type(TestCheckedClass, int, None)
    assert first_call is second_call

def test__make_seq_field_type_sets_type_and_invariant():
    class TestCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(TestCheckedClass, int, lambda x: x > 0)
    assert result.__type__ == int
    assert result.__invariant__(5) is True
    assert result.__invariant__(-1) is False

def test__make_seq_field_type_reduce_method():
    class TestCheckedClass:
        _checked_types = (int,)

    TheType = _make_seq_field_type(TestCheckedClass, int, None)
    instance = TheType([1, 2, 3])
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_seq_field_pickle
    assert reduced[1][0] == TestCheckedClass
    assert reduced[1][1] == int
    assert reduced[1][2] == [1, 2, 3]


# LLM-generated content at query #8
#--------------------------

```python
def test_check_field_parameters_with_invalid_type_parameter():
    field = MockField(type=[123])
    with pytest.raises(TypeError) as excinfo:
        _check_field_parameters(field)
    assert 'Type parameter expected, not' in str(excinfo.value)

def test_check_field_parameters_with_invalid_initial_type():
    field = MockField(type=[int], initial=123.45)
    with pytest.raises(TypeError) as excinfo:
        _check_field_parameters(field)
    assert 'Initial has invalid type' in str(excinfo.value)

def test_check_field_parameters_with_non_callable_invariant():
    field = MockField(invariant=123)
    with pytest.raises(TypeError) as excinfo:
        _check_field_parameters(field)
    assert 'Invariant must be callable' in str(excinfo.value)

def test_check_field_parameters_with_non_callable_factory():
    field = MockField(factory=123)
    with pytest.raises(TypeError) as excinfo:
        _check_field_parameters(field)
    assert 'Factory must be callable' in str(excinfo.value)

def test_check_field_parameters_with_non_callable_serializer():
    field = MockField(serializer=123)
    with pytest.raises(TypeError) as excinfo:
        _check_field_parameters(field)
    assert 'Serializer must be callable' in str(excinfo.value)

def test_check_field_parameters_with_valid_parameters():
    field = MockField(
        type=[int, str],
        initial=42,
        invariant=lambda x: True,
        factory=lambda: 0,
        serializer=lambda x: str(x)
    )
    _check_field_parameters(field)


# LLM-generated content at query #9
#--------------------------

```python
def test_set_fields_empty_bases():
    dct = {}
    bases = []
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {}}

def test_set_fields_single_base_with_empty_dict():
    dct = {}
    bases = [type("Base", (), {"test": {}})]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {}}

def test_set_fields_single_base_with_items():
    dct = {}
    bases = [type("Base", (), {"test": {"key": "value"}})]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"key": "value"}}

def test_set_fields_multiple_bases():
    dct = {}
    bases = [
        type("Base1", (), {"test": {"key1": "value1"}}),
        type("Base2", (), {"test": {"key2": "value2"}})
    ]
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"key1": "value1", "key2": "value2"}}

def test_set_fields_with_pfield():
    class _PField:
        pass

    dct = {"pf": _PField()}
    bases = []
    name = "test"
    set_fields(dct, bases, name)
    assert dct == {"test": {"pf": dct["pf"]}}


# LLM-generated content at query #10
#--------------------------

```python
def test_sequence_field_with_checked_class_and_item_type():
    checked_class = CheckedPSet
    item_type = int
    optional = False
    initial = [1, 2, 3]
    result = _sequence_field(checked_class, item_type, optional, initial)
    assert isinstance(result, _PField)
    assert result.type == {TheType}  # TheType is the dynamically created class
    assert result.factory == TheType.create
    assert result.mandatory == True
    assert result.initial == TheType.create([1, 2, 3])

def test_sequence_field_with_optional_true():
    checked_class = CheckedPVector
    item_type = str
    optional = True
    initial = ['a', 'b']
    result = _sequence_field(checked_class, item_type, optional, initial)
    assert isinstance(result, _PField)
    assert result.type == {TheType, type(None)}
    assert result.factory(None) is None
    assert result.factory(['a', 'b']) == TheType.create(['a', 'b'])
    assert result.mandatory == True
    assert result.initial == TheType.create(['a', 'b'])

def test_sequence_field_with_invariant():
    checked_class = CheckedPSet
    item_type = int
    optional = False
    initial = [1, 2, 3]
    invariant = lambda x: x > 0
    result = _sequence_field(checked_class, item_type, optional, initial, invariant=invariant)
    assert isinstance(result, _PField)
    assert result.invariant == wrap_invariant(invariant)
    assert result.type == {TheType}
    assert result.factory == TheType.create
    assert result.mandatory == True
    assert result.initial == TheType.create([1, 2, 3])

def test_sequence_field_with_item_invariant():
    checked_class = CheckedPVector
    item_type = str
    optional = False
    initial = ['a', 'b']
    item_invariant = lambda x: len(x) > 0
    result = _sequence_field(checked_class, item_type, optional, initial, item_invariant=item_invariant)
    assert isinstance(result, _PField)
    assert result.type == {TheType}
    assert result.factory == TheType.create
    assert result.mandatory == True
    assert result.initial == TheType.create(['a', 'b'])


# LLM-generated content at query #11
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    check_type(TestClass, Field(), "test_field", 42)

def test_check_type_with_invalid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    try:
        check_type(TestClass, Field(), "test_field", "not_an_int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.destination_cls == TestClass
        assert e.field_name == "test_field"
        assert e.expected_type == (int,)
        assert e.actual_type == str
        assert str(e) == "Invalid type for field TestClass.test_field, was str"

def test_check_type_with_no_type_specified():
    class TestClass:
        pass

    class Field:
        type = None

    check_type(TestClass, Field(), "test_field", "any_value")

def test_check_type_with_multiple_valid_types():
    class TestClass:
        pass

    class Field:
        type = (int, str)

    check_type(TestClass, Field(), "test_field", 42)
    check_type(TestClass, Field(), "test_field", "a_string")

def test_check_type_with_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ("builtins.int",)

    check_type(TestClass, Field(), "test_field", 42)

def test_check_type_with_string_type_name_invalid():
    class TestClass:
        pass

    class Field:
        type = ("builtins.int",)

    try:
        check_type(TestClass, Field(), "test_field", "not_an_int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.destination_cls == TestClass
        assert e.field_name == "test_field"
        assert e.expected_type == ("builtins.int",)
        assert e.actual_type == str
        assert str(e) == "Invalid type for field TestClass.test_field, was str"


# LLM-generated content at query #12
#--------------------------

```python
def test_make_pmap_field_type_creates_new_class():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert result.__name__ == "StrToIntPMap"

def test_make_pmap_field_type_reuses_existing_class():
    key_type = str
    value_type = int
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.str"
    value_type = "builtins.int"
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == str
    assert result.__value_type__ == int
    assert result.__name__ == "StrToIntPMap"


# LLM-generated content at query #13
#--------------------------

```python
def test_check_field_parameters_with_valid_types():
    field = MockField(type=[int, str], initial=42, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_type_parameter():
    field = MockField(type=[42], initial=None, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_invalid_initial_type():
    field = MockField(type=[int], initial="not an int", invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_invariant():
    field = MockField(type=[int], initial=42, invariant="not callable", factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_factory():
    field = MockField(type=[int], initial=42, invariant=lambda x: True, factory="not callable", serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_serializer():
    field = MockField(type=[int], initial=42, invariant=lambda x: True, factory=lambda: None, serializer="not callable")
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #14
#--------------------------

```python
def test_invariant_default_value():
    assert _sequence_field(CheckedPSet, str, False, [])[0].invariant == PFIELD_NO_INVARIANT


# LLM-generated content at query #15
#--------------------------

```python
def test_pfield_constructor():
    type_val = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = lambda: 0
    serializer = lambda x: str(x)

    pfield = _PField(type_val, invariant, initial, mandatory, factory, serializer)

    assert pfield.type == type_val
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #16
#--------------------------

```python
def test_pmap_field_basic():
    result = pmap_field(int, str)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional():
    result = pmap_field(int, str, optional=True)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert callable(result.factory)
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def dummy_invariant(x):
        return True, "dummy"
    result = pmap_field(int, str, invariant=dummy_invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert callable(result.invariant)

def test_pmap_field_optional_with_invariant():
    def dummy_invariant(x):
        return True, "dummy"
    result = pmap_field(int, str, optional=True, invariant=dummy_invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert callable(result.factory)
    assert callable(result.invariant)


# LLM-generated content at query #17
#--------------------------

```python
def test_sequence_field_optional_predicate():
    result = _sequence_field(CheckedPSet, int, True, [])
    assert result.factory is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_pmap_field_with_optional_true_returns_correct_type():
    result = pmap_field(str, int, optional=True)
    assert result.type == optional_type(_make_pmap_field_type(str, int))


# LLM-generated content at query #19
#--------------------------

```python
def test_restore_pmap_field_pickle_calls_restore_pickle_with_correct_args():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    _restore_pmap_field_pickle(key_type, value_type, data)
    assert _restore_pickle.call_args == ((_pmap_field_types[key_type, value_type], data),)


# LLM-generated content at query #20
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    assert isinstance(CheckedType(), CheckedType) and PFIELD_NO_SERIALIZER is PFIELD_NO_SERIALIZER


# LLM-generated content at query #21
#--------------------------

```python
def test_pmap_field_with_non_optional_and_no_invariant():
    result = pmap_field(str, int)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_optional_and_no_invariant():
    result = pmap_field(str, int, optional=True)
    assert result.type == {_make_pmap_field_type(str, int), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_non_optional_and_invariant():
    def invariant_func(x):
        return True, "OK"
    result = pmap_field(str, int, invariant=invariant_func)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create
    assert result.invariant(True, "OK") == (True, "OK")

def test_pmap_field_with_optional_and_invariant():
    def invariant_func(x):
        return True, "OK"
    result = pmap_field(str, int, optional=True, invariant=invariant_func)
    assert result.type == {_make_pmap_field_type(str, int), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})
    assert result.invariant(True, "OK") == (True, "OK")


# LLM-generated content at query #22
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariant_fails():
    subject = object()
    invariants = [lambda _: (False, "ERROR_CODE")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE",)


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_evaluates_to_false():
    field = type('Field', (), {
        'type': [123],  # Not a type or str
        'initial': None,
        'invariant': lambda: True,
        'factory': lambda: None,
        'serializer': lambda x: x
    })()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError but none was raised"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"


# LLM-generated content at query #24
#--------------------------

```python
def test_pfield_init_assigns_factory():
    factory = lambda: None
    pfield = _PField(type=None, invariant=None, initial=None, mandatory=None, factory=factory, serializer=None)
    assert pfield._factory is factory


# LLM-generated content at query #25
#--------------------------

```python
def test_check_global_invariants_with_all_invariants_passing():
    subject = "test_subject"
    invariants = [
        lambda s: (True, "error1"),
        lambda s: (True, "error2")
    ]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #26
#--------------------------

```python
def test_pmap_field_optional_predicate():
    key_type = str
    value_type = int
    optional = True
    invariant = PFIELD_NO_INVARIANT

    result = pmap_field(key_type, value_type, optional, invariant)

    assert result.factory is not None


# LLM-generated content at query #27
#--------------------------

```python
def test_check_type_with_valid_type():
    class Field:
        type = (int,)

    class Destination:
        pass

    check_type(Destination, Field(), "field_name", 42)


# LLM-generated content at query #28
#--------------------------

```python
def test_serialize_with_checked_type_and_pfield_no_serializer():
    assert isinstance(CheckedType(), CheckedType) and PFIELD_NO_SERIALIZER is PFIELD_NO_SERIALIZER


# LLM-generated content at query #29
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda x: (True, None), lambda x: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = object()
    invariants = [lambda x: (True, None), lambda x: (False, "ERROR1")]
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        assert e.error_codes == ("ERROR1",)

def test_check_global_invariants_multiple_errors():
    subject = object()
    invariants = [lambda x: (False, "ERROR1"), lambda x: (False, "ERROR2")]
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        assert e.error_codes == ("ERROR1", "ERROR2")

def test_check_global_invariants_empty_invariants():
    subject = object()
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #30
#--------------------------

```python
def test_pfield_constructor_initialization():
    type_value = (int,)
    invariant = lambda x: x > 0
    initial = 10
    mandatory = True
    factory = lambda: 0
    serializer = str

    pfield = _PField(type_value, invariant, initial, mandatory, factory, serializer)

    assert pfield.type == type_value
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #31
#--------------------------

```python
def test_set_fields_predicate():
    class _PField:
        pass

    class Base1:
        field1 = _PField()
        field2 = "value"

    class Base2:
        field1 = _PField()
        field3 = "another_value"

    dct = {}
    bases = (Base1, Base2)
    name = "__pfields__"

    set_fields(dct, bases, name)

    assert isinstance(dct[name]["field1"], _PField)


# LLM-generated content at query #32
#--------------------------

```python
def test_check_global_invariants_with_no_errors():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #33
#--------------------------

```python
def test_restore_seq_field_pickle_calls_restore_pickle_with_correct_args():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _seq_field_types, _restore_seq_field_pickle

    # Mock data
    checked_class = type('MockCheckedClass', (), {})
    item_type = type('MockItemType', (), {})
    data = {'key': 'value'}

    # Mock _seq_field_types to return a known type
    mock_type = type('MockType', (), {})
    _seq_field_types[checked_class, item_type] = mock_type

    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, data)

    # Assert _restore_pickle was called with the correct arguments
    assert result == _restore_pickle(mock_type, data)


# LLM-generated content at query #34
#--------------------------

```python
def test_pmap_field_optional_factory_returns_none():
    result = pmap_field(str, int, optional=True).factory(None)
    assert result is None


# LLM-generated content at query #35
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    check_type(TestClass, Field(), "test_field", 42)

def test_check_type_with_invalid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    try:
        check_type(TestClass, Field(), "test_field", "not an int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.destination_cls == TestClass
        assert e.field_name == "test_field"
        assert e.expected_type == (int,)
        assert e.actual_type == str

def test_check_type_with_no_type_specified():
    class TestClass:
        pass

    class Field:
        type = None

    check_type(TestClass, Field(), "test_field", "any value")

def test_check_type_with_multiple_valid_types():
    class TestClass:
        pass

    class Field:
        type = (int, float)

    check_type(TestClass, Field(), "test_field", 42)
    check_type(TestClass, Field(), "test_field", 3.14)

def test_check_type_with_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ("builtins.int",)

    check_type(TestClass, Field(), "test_field", 42)


# LLM-generated content at query #36
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    check_type(TestClass, Field(), "test_field", 42)


# LLM-generated content at query #37
#--------------------------

```python
def test_pfield_initialization():
    type_val = (int,)
    invariant_val = lambda x: x > 0
    initial_val = 0
    mandatory_val = True
    factory_val = None
    serializer_val = str

    pfield = _PField(type_val, invariant_val, initial_val, mandatory_val, factory_val, serializer_val)

    assert pfield.type == type_val
    assert pfield.invariant == invariant_val
    assert pfield.initial == initial_val
    assert pfield.mandatory == mandatory_val
    assert pfield._factory == factory_val
    assert pfield.serializer == serializer_val


# LLM-generated content at query #38
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"
    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #39
#--------------------------

```python
def test_set_fields_empty_bases():
    dct = {}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'test_name': {}}

def test_set_fields_with_single_base():
    class Base:
        test_name = {'a': 1}

    dct = {}
    bases = [Base]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'test_name': {'a': 1}}

def test_set_fields_with_multiple_bases():
    class Base1:
        test_name = {'a': 1}

    class Base2:
        test_name = {'b': 2}

    dct = {}
    bases = [Base1, Base2]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'test_name': {'a': 1, 'b': 2}}

def test_set_fields_with_pfield():
    class _PField:
        pass

    class Base:
        test_name = {'a': 1}

    dct = {'field': _PField()}
    bases = [Base]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'test_name': {'a': 1, 'field': _PField()}}


# LLM-generated content at query #40
#--------------------------

```python
def test_check_field_parameters_with_non_callable_invariant():
    field = type('Field', (), {'type': [int], 'initial': 5, 'invariant': 'not_callable', 'factory': lambda: None, 'serializer': lambda x: x})()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError not raised"
    except TypeError as e:
        assert str(e) == 'Invariant must be callable'


# LLM-generated content at query #41
#--------------------------

```python
def test_pfield_constructor_with_all_parameters():
    type_value = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = lambda: 0
    serializer = lambda x: str(x)

    pfield = _PField(type_value, invariant, initial, mandatory, factory, serializer)

    assert pfield.type == type_value
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer

def test_pfield_constructor_with_none_factory():
    type_value = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = PFIELD_NO_FACTORY
    serializer = lambda x: str(x)

    pfield = _PField(type_value, invariant, initial, mandatory, factory, serializer)

    assert pfield.type == type_value
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #42
#--------------------------

```python
def test_set_fields_with_empty_bases():
    dct = {'a': 1, 'b': 2}
    bases = []
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct == {'a': 1, 'b': 2, 'fields': {}}

def test_set_fields_with_non_empty_bases():
    class Base1:
        fields = {'x': 1}

    class Base2:
        fields = {'y': 2}

    dct = {'a': 1, 'b': 2}
    bases = [Base1, Base2]
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct == {'a': 1, 'b': 2, 'fields': {'x': 1, 'y': 2}}

def test_set_fields_with_pfield_instances():
    class _PField:
        pass

    dct = {'a': _PField(), 'b': 2}
    bases = []
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct == {'b': 2, 'fields': {'a': dct['fields']['a']}}
    assert isinstance(dct['fields']['a'], _PField)

def test_set_fields_with_overlapping_fields_in_bases():
    class Base1:
        fields = {'x': 1}

    class Base2:
        fields = {'x': 2, 'y': 3}

    dct = {'a': 1}
    bases = [Base1, Base2]
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct == {'a': 1, 'fields': {'x': 2, 'y': 3}}

def test_set_fields_with_no_fields_in_bases():
    class Base1:
        pass

    class Base2:
        pass

    dct = {'a': 1, 'b': 2}
    bases = [Base1, Base2]
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct == {'a': 1, 'b': 2, 'fields': {}}


# LLM-generated content at query #43
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle

    checked_class = type('MockCheckedClass', (), {})
    item_type = int
    data = [1, 2, 3]

    result = _restore_seq_field_pickle(checked_class, item_type, data)

    assert isinstance(result, type(_restore_pickle(type('MockType', (), {}), data)))


# LLM-generated content at query #44
#--------------------------

```python
def test_pfield_initialization():
    pfield = _PField(type=int, invariant=lambda x: True, initial=0, mandatory=True, factory=None, serializer=None)
    assert pfield.type == int
    assert pfield.invariant(0) is True
    assert pfield.initial == 0
    assert pfield.mandatory is True
    assert pfield._factory is None
    assert pfield.serializer is None


# LLM-generated content at query #45
#--------------------------

```python
def test_valid_field_parameters():
    field = MockField(type=[int], initial=5, invariant=lambda x: True, factory=lambda: 0, serializer=lambda x: str(x))
    _check_field_parameters(field)

def test_invalid_type_parameter():
    field = MockField(type=[123], initial=5, invariant=lambda x: True, factory=lambda: 0, serializer=lambda x: str(x))
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_invalid_initial_type():
    field = MockField(type=[int], initial="not an int", invariant=lambda x: True, factory=lambda: 0, serializer=lambda x: str(x))
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_non_callable_invariant():
    field = MockField(type=[int], initial=5, invariant="not callable", factory=lambda: 0, serializer=lambda x: str(x))
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_non_callable_factory():
    field = MockField(type=[int], initial=5, invariant=lambda x: True, factory="not callable", serializer=lambda x: str(x))
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_non_callable_serializer():
    field = MockField(type=[int], initial=5, invariant=lambda x: True, factory=lambda: 0, serializer="not callable")
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #46
#--------------------------

```python
def test__check_field_parameters_with_invalid_initial_type():
    field = Field(
        type=[int],
        initial="not_an_int",
        invariant=lambda x: True,
        factory=lambda: None,
        serializer=lambda x: x
    )
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #47
#--------------------------

```python
def test_pmap_field_optional_factory_with_none():
    result = pmap_field(str, int, optional=True).factory(None)
    assert result is None


# LLM-generated content at query #48
#--------------------------

```python
def test_make_pmap_field_type_creates_new_subclass():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert result.__name__ == "IntToStrPMap"

def test_make_pmap_field_type_reuses_existing_subclass():
    key_type = int
    value_type = str
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.int"
    value_type = "builtins.str"
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == int
    assert result.__value_type__ == str
    assert result.__name__ == "IntToStrPMap"


# LLM-generated content at query #49
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    assert not is_field_ignore_extra_complaint(str, object(), False)

def test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_a_subset_of_type_cls():
    assert not is_field_ignore_extra_complaint(str, object(), True)

def test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param():
    field = object()
    field.type = {str}
    field.factory = lambda: None
    assert not is_field_ignore_extra_complaint(str, field, True)

def test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_are_met():
    field = object()
    field.type = {str}
    field.factory = lambda ignore_extra: None
    assert is_field_ignore_extra_complaint(str, field, True)


# LLM-generated content at query #50
#--------------------------

```python
def test_sequence_field_optional_predicate():
    checked_class = CheckedPVector
    item_type = int
    optional = True
    initial = [1, 2, 3]
    invariant = PFIELD_NO_INVARIANT
    item_invariant = PFIELD_NO_INVARIANT

    result = _sequence_field(checked_class, item_type, optional, initial, invariant, item_invariant)

    assert result.factory is not None


# LLM-generated content at query #51
#--------------------------

```python
def test_check_field_parameters_with_non_callable_invariant():
    field = type('Field', (), {'type': [int], 'initial': 0, 'invariant': 123, 'factory': lambda: None, 'serializer': lambda x: x})()
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #52
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    check_type(TestClass, Field(), 'test_field', 42)

def test_check_type_with_invalid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    try:
        check_type(TestClass, Field(), 'test_field', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert str(e) == "Invalid type for field TestClass.test_field, was str"

def test_check_type_with_no_type_specified():
    class TestClass:
        pass

    class Field:
        type = None

    check_type(TestClass, Field(), 'test_field', 'any_value')

def test_check_type_with_multiple_valid_types():
    class TestClass:
        pass

    class Field:
        type = (int, str)

    check_type(TestClass, Field(), 'test_field', 42)
    check_type(TestClass, Field(), 'test_field', 'string_value')

def test_check_type_with_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ('builtins.int',)

    check_type(TestClass, Field(), 'test_field', 42)

def test_check_type_with_string_type_name_invalid():
    class TestClass:
        pass

    class Field:
        type = ('builtins.int',)

    try:
        check_type(TestClass, Field(), 'test_field', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert str(e) == "Invalid type for field TestClass.test_field, was str"


# LLM-generated content at query #53
#--------------------------

```python
def test_pfield_constructor():
    type = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = lambda: 0
    serializer = lambda x: str(x)
    pfield = _PField(type, invariant, initial, mandatory, factory, serializer)
    assert pfield.type == type
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #54
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = int
    value_type = str
    data = [(1, "a"), (2, "b")]
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


# LLM-generated content at query #55
#--------------------------

```python
def test_is_field_ignore_extra_complaint_with_false_ignore_extra():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field
    from pyrsistent._checked_types import get_type

    class TestClass:
        pass

    field = Field(type=TestClass, factory=lambda: None)
    assert not is_field_ignore_extra_complaint(get_type(TestClass), field, False)

def test_is_field_ignore_extra_complaint_with_non_matching_type():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field
    from pyrsistent._checked_types import get_type

    class TestClass:
        pass

    class OtherClass:
        pass

    field = Field(type=OtherClass, factory=lambda: None)
    assert not is_field_ignore_extra_complaint(get_type(TestClass), field, True)

def test_is_field_ignore_extra_complaint_without_ignore_extra_in_factory():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field
    from pyrsistent._checked_types import get_type

    class TestClass:
        pass

    field = Field(type=TestClass, factory=lambda: None)
    assert not is_field_ignore_extra_complaint(get_type(TestClass), field, True)

def test_is_field_ignore_extra_complaint_with_ignore_extra_in_factory():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field
    from pyrsistent._checked_types import get_type

    class TestClass:
        pass

    field = Field(type=TestClass, factory=lambda ignore_extra: None)
    assert is_field_ignore_extra_complaint(get_type(TestClass), field, True)

def test_is_field_ignore_extra_complaint_with_set_type():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field
    from pyrsistent._checked_types import get_type

    class TestClass:
        pass

    field = Field(type={TestClass}, factory=lambda ignore_extra: None)
    assert is_field_ignore_extra_complaint(get_type(TestClass), field, True)


# LLM-generated content at query #56
#--------------------------

```python
def test_no_invariants():
    check_global_invariants("subject", [])

def test_all_invariants_pass():
    def invariant1(subject):
        return True, None
    def invariant2(subject):
        return True, None
    check_global_invariants("subject", [invariant1, invariant2])

def test_single_invariant_fails():
    def invariant(subject):
        return False, "ERROR1"
    try:
        check_global_invariants("subject", [invariant])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'

def test_multiple_invariants_fail():
    def invariant1(subject):
        return False, "ERROR1"
    def invariant2(subject):
        return False, "ERROR2"
    try:
        check_global_invariants("subject", [invariant1, invariant2])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR1", "ERROR2")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'

def test_mixed_invariants():
    def invariant1(subject):
        return True, None
    def invariant2(subject):
        return False, "ERROR1"
    def invariant3(subject):
        return True, None
    try:
        check_global_invariants("subject", [invariant1, invariant2, invariant3])
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("ERROR1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_pfield_constructor_initialization():
    type_val = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = lambda: 42
    serializer = lambda x: str(x)

    pfield = _PField(type_val, invariant, initial, mandatory, factory, serializer)

    assert pfield.type == type_val
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #2
#--------------------------

```python
def test__make_seq_field_type_creates_subclass_with_correct_name():
    class MockCheckedClass:
        _checked_types = (int, str)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert result.__name__ == "IntStrSeqFieldType"
    assert issubclass(result, MockCheckedClass)

def test__make_seq_field_type_reuses_existing_type():
    class MockCheckedClass:
        _checked_types = (int,)

    type1 = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    type2 = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert type1 is type2

def test__make_seq_field_type_sets_type_and_invariant():
    class MockCheckedClass:
        _checked_types = (float,)

    result = _make_seq_field_type(MockCheckedClass, float, lambda x: x > 0)
    assert result.__type__ == float
    assert result.__invariant__(5.0) is True
    assert result.__invariant__(-1.0) is False

def test__make_seq_field_type_implements_reduce():
    class MockCheckedClass:
        _checked_types = (str,)

    result_type = _make_seq_field_type(MockCheckedClass, str, lambda x: True)
    instance = result_type()
    instance._data = ["a", "b"]
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_seq_field_pickle
    assert reduced[1][0] == MockCheckedClass
    assert reduced[1][1] == str
    assert reduced[1][2] == ["a", "b"]


# LLM-generated content at query #3
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass:
        pass
    field = type('Field', (), {'type': (int, str)})()
    check_type(TestClass, field, 'test_field', 123)

def test_check_type_with_invalid_type():
    class TestClass:
        pass
    field = type('Field', (), {'type': (int, str)})()
    try:
        check_type(TestClass, field, 'test_field', 123.45)
    except PTypeError as e:
        assert str(e) == "Invalid type for field TestClass.test_field, was float"

def test_check_type_with_no_type_specified():
    class TestClass:
        pass
    field = type('Field', (), {'type': None})()
    check_type(TestClass, field, 'test_field', 123.45)

def test_check_type_with_string_type_name():
    class TestClass:
        pass
    field = type('Field', (), {'type': ('builtins.int', 'builtins.str')})()
    check_type(TestClass, field, 'test_field', "test")


# LLM-generated content at query #4
#--------------------------

```python
def test_check_field_parameters_with_valid_field():
    field = MockField(type=[int, str], initial=42, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_type_parameter():
    field = MockField(type=[int, 42], initial=None, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_invalid_initial_type():
    field = MockField(type=[int, str], initial=42.0, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_invariant():
    field = MockField(type=[int, str], initial=42, invariant=42, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_factory():
    field = MockField(type=[int, str], initial=42, invariant=lambda x: True, factory=42, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_serializer():
    field = MockField(type=[int, str], initial=42, invariant=lambda x: True, factory=lambda: None, serializer=42)
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #5
#--------------------------

```python
def test_check_field_parameters_with_non_callable_invariant():
    field = MockField(invariant=123)
    assert _check_field_parameters(field) == False


# LLM-generated content at query #6
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    assert is_field_ignore_extra_complaint(str, MockField(type=str), False) is False

def test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_subclass():
    assert is_field_ignore_extra_complaint(str, MockField(type=int), True) is False

def test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param():
    assert is_field_ignore_extra_complaint(str, MockField(type=str, factory=lambda x: x), True) is False

def test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_met():
    assert is_field_ignore_extra_complaint(str, MockField(type=str, factory=lambda x, ignore_extra=False: x), True) is True

def test_is_field_ignore_extra_complaint_returns_true_for_set_type():
    assert is_field_ignore_extra_complaint(str, MockField(type={str}), True) is True

def test_is_field_ignore_extra_complaint_returns_false_for_empty_tuple_type():
    assert is_field_ignore_extra_complaint(str, MockField(type=()), True) is False


# LLM-generated content at query #7
#--------------------------

```python
def test__make_seq_field_type_creates_new_type():
    class MockCheckedClass:
        _checked_types = (int,)
    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert isinstance(result, type)
    assert issubclass(result, MockCheckedClass)
    assert result.__name__ == "IntSeq"
    assert result.__type__ == int
    assert result.__invariant__(5) is True

def test__make_seq_field_type_reuses_existing_type():
    class MockCheckedClass:
        _checked_types = (int,)
    type1 = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    type2 = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert type1 is type2

def test__make_seq_field_type_with_different_types():
    class MockCheckedClass:
        _checked_types = (int, str)
    result = _make_seq_field_type(MockCheckedClass, (int, str), lambda x: True)
    assert result.__name__ == "IntStrSeq"
    assert result.__type__ == (int, str)


# LLM-generated content at query #8
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (False, 'ERROR1'), lambda _: (False, 'ERROR2')]
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        assert e.error_codes == ('ERROR1', 'ERROR2')
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #9
#--------------------------

```python
def test__check_field_parameters_with_non_callable_invariant():
    field = MockField(invariant=123)
    with pytest.raises(TypeError, match="Invariant must be callable"):
        _check_field_parameters(field)


# LLM-generated content at query #10
#--------------------------

```python
def test_check_field_parameters_with_valid_types():
    field = type('Field', (), {
        'type': [int, str],
        'initial': 42,
        'invariant': lambda x: True,
        'factory': lambda: None,
        'serializer': lambda x: str(x)
    })()
    _check_field_parameters(field)


# LLM-generated content at query #11
#--------------------------

```python
def test_pmap_field_creates_checked_pmap_field():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type)
    assert isinstance(result, _PField)
    assert result.type == {TheMap}  # Assuming TheMap is the generated class
    assert result.mandatory == True
    assert result.initial == TheMap()
    assert result.factory == TheMap.create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_optional():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type, optional=True)
    assert isinstance(result, _PField)
    assert result.type == {TheMap, type(None)}
    assert result.mandatory == True
    assert result.initial == TheMap()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == TheMap({"a": 1})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    key_type = str
    value_type = int
    invariant = lambda x: True
    result = pmap_field(key_type, value_type, invariant=invariant)
    assert isinstance(result, _PField)
    assert result.type == {TheMap}
    assert result.mandatory == True
    assert result.initial == TheMap()
    assert result.factory == TheMap.create
    assert result.invariant == invariant

def test_pmap_field_with_optional_and_invariant():
    key_type = str
    value_type = int
    invariant = lambda x: True
    result = pmap_field(key_type, value_type, optional=True, invariant=invariant)
    assert isinstance(result, _PField)
    assert result.type == {TheMap, type(None)}
    assert result.mandatory == True
    assert result.initial == TheMap()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == TheMap({"a": 1})
    assert result.invariant == invariant


# LLM-generated content at query #12
#--------------------------

```python
def test__make_pmap_field_type_creates_new_class_with_correct_name():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int

def test__make_pmap_field_type_returns_cached_class():
    key_type = str
    value_type = int
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call

def test__make_pmap_field_type_with_string_type_names():
    key_type = "builtins.str"
    value_type = "builtins.int"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


# LLM-generated content at query #13
#--------------------------

```python
def test_check_field_parameters_with_non_callable_invariant():
    field = type('Field', (), {'type': [str], 'initial': PFIELD_NO_INITIAL, 'invariant': 'not_callable', 'factory': lambda: None, 'serializer': lambda x: x})()
    with pytest.raises(TypeError, match='Invariant must be callable'):
        _check_field_parameters(field)


# LLM-generated content at query #14
#--------------------------

```python
def test_set_fields_merges_base_class_fields():
    class Base1:
        fields = {'a': 1, 'b': 2}

    class Base2:
        fields = {'b': 3, 'c': 4}

    dct = {}
    set_fields(dct, [Base1, Base2], 'fields')
    assert dct['fields'] == {'a': 1, 'b': 3, 'c': 4}

def test_set_fields_moves_pfield_instances_to_fields_dict():
    class _PField:
        pass

    class Base:
        fields = {}

    dct = {'x': _PField(), 'y': 'not a field'}
    set_fields(dct, [Base], 'fields')
    assert 'x' in dct['fields']
    assert 'x' not in dct
    assert 'y' in dct


# LLM-generated content at query #15
#--------------------------

```python
def test_set_fields_predicate_false():
    dct = {'a': 1}
    bases = []
    name = 'a'
    assert not (name in dct and any(isinstance(b, dict) and name in b for b in bases))


# LLM-generated content at query #16
#--------------------------

```python
def test_make_pmap_field_type_creates_new_class():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert isinstance(result, type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type

def test_make_pmap_field_type_reuses_existing_class():
    key_type = str
    value_type = int
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call

def test_make_pmap_field_type_sets_correct_name():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "StrToIntPMap"

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.str"
    value_type = "builtins.int"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == str
    assert result.__value_type__ == int
    assert result.__name__ == "StrToIntPMap"


# LLM-generated content at query #17
#--------------------------

```python
def test_pfield_constructor_initialization():
    type_value = (int,)
    invariant_value = lambda x: x > 0
    initial_value = 0
    mandatory_value = True
    factory_value = None
    serializer_value = str

    pfield = _PField(type_value, invariant_value, initial_value, mandatory_value, factory_value, serializer_value)

    assert pfield.type == type_value
    assert pfield.invariant == invariant_value
    assert pfield.initial == initial_value
    assert pfield.mandatory == mandatory_value
    assert pfield._factory == factory_value
    assert pfield.serializer == serializer_value


# LLM-generated content at query #18
#--------------------------

```python
def test_check_type_with_valid_type():
    class Field:
        type = (int,)

    class Destination:
        pass

    check_type(Destination, Field(), "field_name", 42)


# LLM-generated content at query #19
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"
    assert result.__key_type__ == int
    assert result.__value_type__ == str

def test_make_pmap_field_type_reuses_existing_type():
    key_type = int
    value_type = str
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.int"
    value_type = "builtins.str"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"
    assert result.__key_type__ == int
    assert result.__value_type__ == str

def test_make_pmap_field_type_pickle_support():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    instance = result({1: "a", 2: "b"})
    assert instance.__reduce__() == (_restore_pmap_field_pickle, (int, str, {1: "a", 2: "b"}))


# LLM-generated content at query #20
#--------------------------

```python
def test_pmap_field_docstring_exists():
    assert pmap_field.__doc__ is not None
    assert len(pmap_field.__doc__) > 0


# LLM-generated content at query #21
#--------------------------

```python
def test_restore_seq_field_pickle_returns_correct_type():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle

    class MockClass:
        pass

    mock_data = [1, 2, 3]
    _restore_pickle.return_value = mock_data

    result = _restore_seq_field_pickle(MockClass, int, mock_data)

    assert result == mock_data
    _restore_pickle.assert_called_once_with(_seq_field_types[MockClass, int], mock_data)


# LLM-generated content at query #22
#--------------------------

```python
def test_sequence_field_creates_checked_class():
    result = _sequence_field(CheckedPSet, int, False, [])
    assert isinstance(result, _PField)
    assert result.type == {CheckedPSet}
    assert result.factory == CheckedPSet.create
    assert result.mandatory is True
    assert result.initial == CheckedPSet.create([])

def test_sequence_field_with_optional():
    result = _sequence_field(CheckedPSet, int, True, [])
    assert isinstance(result, _PField)
    assert result.type == {CheckedPSet, type(None)}
    assert result.mandatory is True
    assert result.initial == result.factory([])

def test_sequence_field_with_initial_none():
    result = _sequence_field(CheckedPSet, int, True, None)
    assert isinstance(result, _PField)
    assert result.type == {CheckedPSet, type(None)}
    assert result.mandatory is True
    assert result.initial is None

def test_sequence_field_with_invariant():
    def invariant(x):
        return x is not None
    result = _sequence_field(CheckedPSet, int, False, [], invariant=invariant)
    assert isinstance(result, _PField)
    assert result.invariant == invariant
    assert result.type == {CheckedPSet}
    assert result.factory == CheckedPSet.create
    assert result.mandatory is True
    assert result.initial == CheckedPSet.create([])

def test_sequence_field_with_item_invariant():
    def item_invariant(x):
        return x > 0
    result = _sequence_field(CheckedPSet, int, False, [], item_invariant=item_invariant)
    assert isinstance(result, _PField)
    assert result.type == {CheckedPSet}
    assert result.factory == CheckedPSet.create
    assert result.mandatory is True
    assert result.initial == CheckedPSet.create([])

def test_sequence_field_with_checked_pvector():
    result = _sequence_field(CheckedPVector, str, False, [])
    assert isinstance(result, _PField)
    assert result.type == {CheckedPVector}
    assert result.factory == CheckedPVector.create
    assert result.mandatory is True
    assert result.initial == CheckedPVector.create([])

def test_sequence_field_with_optional_checked_pvector():
    result = _sequence_field(CheckedPVector, str, True, [])
    assert isinstance(result, _PField)
    assert result.type == {CheckedPVector, type(None)}
    assert result.mandatory is True
    assert result.initial == result.factory([])

def test_sequence_field_with_custom_initial():
    result = _sequence_field(CheckedPSet, int, False, [1, 2, 3])
    assert isinstance(result, _PField)
    assert result.type == {CheckedPSet}
    assert result.factory == CheckedPSet.create
    assert result.mandatory is True
    assert result.initial == CheckedPSet.create([1, 2, 3])


# LLM-generated content at query #23
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    check_type(TestClass, Field(), 'test_field', 42)

def test_check_type_with_invalid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    try:
        check_type(TestClass, Field(), 'test_field', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.destination_cls == TestClass
        assert e.field_name == 'test_field'
        assert e.expected_types == (int,)
        assert e.actual_type == str

def test_check_type_with_no_type_specified():
    class TestClass:
        pass

    class Field:
        type = None

    check_type(TestClass, Field(), 'test_field', 'any_value')

def test_check_type_with_multiple_valid_types():
    class TestClass:
        pass

    class Field:
        type = (int, float)

    check_type(TestClass, Field(), 'test_field', 3.14)

def test_check_type_with_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ('builtins.int',)

    check_type(TestClass, Field(), 'test_field', 100)

def test_check_type_with_string_type_name_invalid():
    class TestClass:
        pass

    class Field:
        type = ('builtins.int',)

    try:
        check_type(TestClass, Field(), 'test_field', [1, 2, 3])
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.expected_types == ('builtins.int',)
        assert e.actual_type == list


# LLM-generated content at query #24
#--------------------------

```python
def test_sequence_field_optional_false():
    checked_class = type('CheckedClass', (), {})
    item_type = str
    optional = False
    initial = ['a', 'b']
    invariant = lambda x: True
    item_invariant = lambda x: True

    result = _sequence_field(checked_class, item_type, optional, initial, invariant, item_invariant)

    assert result.factory == checked_class.create


# LLM-generated content at query #25
#--------------------------

```python
def test_sequence_field_optional_false():
    checked_class = type('CheckedClass', (), {})
    item_type = int
    optional = False
    initial = [1, 2, 3]
    result = _sequence_field(checked_class, item_type, optional, initial)
    assert result.factory == TheType.create


# LLM-generated content at query #26
#--------------------------

```python
def test__PField___init___assigns__factory_to_factory_parameter():
    field = _PField(type=None, invariant=None, initial=None, mandatory=None, factory="test_factory", serializer=None)
    assert field._factory == "test_factory"


# LLM-generated content at query #27
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    checked_value = CheckedType()
    checked_value.serialize = lambda fmt: "serialized"
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == "serialized"

def test_serialize_with_custom_serializer():
    serializer = lambda fmt, val: f"{fmt}:{val}"
    result = serialize(serializer, "json", "data")
    assert result == "json:data"


# LLM-generated content at query #28
#--------------------------

```python
def test_pmap_field_basic():
    result = pmap_field(int, str)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory == True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional():
    result = pmap_field(int, str, optional=True)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str), type(None)}
    assert result.mandatory == True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory(None) is None
    assert result.factory({1: "a"}) == _make_pmap_field_type(int, str).create({1: "a"})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    invariant = lambda x: (True, "OK")
    result = pmap_field(int, str, invariant=invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory == True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert result.invariant == wrap_invariant(invariant)

def test_pmap_field_optional_with_invariant():
    invariant = lambda x: (True, "OK")
    result = pmap_field(int, str, optional=True, invariant=invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str), type(None)}
    assert result.mandatory == True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory(None) is None
    assert result.factory({1: "a"}) == _make_pmap_field_type(int, str).create({1: "a"})
    assert result.invariant == wrap_invariant(invariant)


# LLM-generated content at query #29
#--------------------------

```python
def test_set_fields_predicate_false():
    dct = {}
    bases = []
    name = "test_name"
    assert not (dct and bases and name)


# LLM-generated content at query #30
#--------------------------

```python
def test_sequence_field_optional_false():
    checked_class = type('CheckedClass', (), {})
    item_type = str
    optional = False
    initial = ['a', 'b']
    invariant = lambda x: (True, None)
    item_invariant = lambda x: (True, None)

    result = _sequence_field(checked_class, item_type, optional, initial, invariant, item_invariant)

    assert result.factory == checked_class.create


# LLM-generated content at query #31
#--------------------------

```python
def test_pmap_field_optional_type_predicate():
    key_type = str
    value_type = int
    optional = True
    invariant = PFIELD_NO_INVARIANT
    TheMap = _make_pmap_field_type(key_type, value_type)
    result = pmap_field(key_type, value_type, optional, invariant)
    assert result.type == optional_type(TheMap)


# LLM-generated content at query #32
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    result = is_field_ignore_extra_complaint(str, None, False)
    assert result is False

def test_is_field_ignore_extra_complaint_returns_false_when_type_mismatch():
    result = is_field_ignore_extra_complaint(str, None, True)
    assert result is False


# LLM-generated content at query #33
#--------------------------

```python
def test_is_type_cls_with_set_field_type():
    assert is_type_cls(object, set())

def test_is_type_cls_with_empty_tuple():
    assert not is_type_cls(object, ())

def test_is_type_cls_with_non_empty_tuple_and_valid_subclass():
    assert is_type_cls(object, (int,))

def test_is_type_cls_with_non_empty_tuple_and_invalid_subclass():
    assert not is_type_cls(int, (str,))

def test_is_type_cls_with_type_directly():
    assert is_type_cls(object, int)

def test_is_type_cls_with_type_directly_and_invalid_subclass():
    assert not is_type_cls(int, str)


# LLM-generated content at query #34
#--------------------------

```python
def test__make_seq_field_type_creates_new_type():
    class MockCheckedClass:
        _checked_types = (int,)
    result = _make_seq_field_type(MockCheckedClass, int, None)
    assert result.__name__ == "IntSeq"
    assert result.__type__ == int
    assert result.__invariant__ is None

def test__make_seq_field_type_reuses_existing_type():
    class MockCheckedClass:
        _checked_types = (int,)
    first_call = _make_seq_field_type(MockCheckedClass, int, None)
    second_call = _make_seq_field_type(MockCheckedClass, int, None)
    assert first_call is second_call

def test__make_seq_field_type_with_different_item_type():
    class MockCheckedClass:
        _checked_types = (int, str)
    int_type = _make_seq_field_type(MockCheckedClass, int, None)
    str_type = _make_seq_field_type(MockCheckedClass, str, None)
    assert int_type.__name__ == "IntStrSeq"
    assert str_type.__name__ == "IntStrSeq"
    assert int_type.__type__ == int
    assert str_type.__type__ == str

def test__make_seq_field_type_with_invariant():
    class MockCheckedClass:
        _checked_types = (int,)
    def mock_invariant(x):
        return x > 0
    result = _make_seq_field_type(MockCheckedClass, int, mock_invariant)
    assert result.__invariant__ == mock_invariant

def test__make_seq_field_type_pickle_support():
    class MockCheckedClass:
        _checked_types = (int,)
        def __init__(self, items):
            self._items = items
        def __iter__(self):
            return iter(self._items)
    result = _make_seq_field_type(MockCheckedClass, int, None)
    instance = result([1, 2, 3])
    assert instance.__reduce__() == (_restore_seq_field_pickle, (MockCheckedClass, int, [1, 2, 3]))


# LLM-generated content at query #35
#--------------------------

```python
def test_pmap_field_basic():
    result = pmap_field(str, int)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional():
    result = pmap_field(str, int, optional=True)
    assert result.type == {_make_pmap_field_type(str, int), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def invariant_func(x):
        return True, "test"
    result = pmap_field(str, int, invariant=invariant_func)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create
    assert result.invariant(True, "test") == (True, "test")

def test_pmap_field_optional_with_invariant():
    def invariant_func(x):
        return True, "test"
    result = pmap_field(str, int, optional=True, invariant=invariant_func)
    assert result.type == {_make_pmap_field_type(str, int), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})
    assert result.invariant(True, "test") == (True, "test")


# LLM-generated content at query #36
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda x: (True, None), lambda x: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = object()
    invariants = [lambda x: (False, 'E1'), lambda x: (True, None), lambda x: (False, 'E2')]
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        assert e.error_codes == ('E1', 'E2')
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #37
#--------------------------

```python
def test_restore_pmap_field_pickle_calls_restore_pickle_with_correct_args():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    _pmap_field_types = {(str, int): type("MockType", (), {"create": lambda self, data, _factory_fields: data})}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert result == data


# LLM-generated content at query #38
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    assert not is_field_ignore_extra_complaint(object, object(), False)


# LLM-generated content at query #39
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"

    assert isinstance(value, CheckedType)
    assert serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #40
#--------------------------

```python
def test_pfield_constructor():
    type_val = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = lambda: 0
    serializer = lambda x: x

    pfield = _PField(type_val, invariant, initial, mandatory, factory, serializer)

    assert pfield.type == type_val
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #41
#--------------------------

```python
def test_check_global_invariants_with_empty_invariants():
    subject = object()
    invariants = []
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #42
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == data


# LLM-generated content at query #43
#--------------------------

```python
def test_check_global_invariants_no_error():
    subject = "test_subject"
    invariants = [lambda x: (True, "OK1"), lambda x: (True, "OK2")]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = "test_subject"
    invariants = [lambda x: (False, "ERROR1"), lambda x: (True, "OK2")]
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        assert e.error_codes == ("ERROR1",)

def test_check_global_invariants_multiple_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, "ERROR1"), lambda x: (False, "ERROR2")]
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        assert e.error_codes == ("ERROR1", "ERROR2")

def test_check_global_invariants_empty_invariants():
    subject = "test_subject"
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #44
#--------------------------

```python
def test_check_global_invariants_with_valid_subject():
    subject = "valid_subject"
    invariants = [lambda s: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #45
#--------------------------

```python
def test_set_fields_predicate_false():
    dct = {'a': 1, 'b': 2}
    bases = []
    name = 'new_key'
    assert not (isinstance(dct.get('a'), _PField) or isinstance(dct.get('b'), _PField))


# LLM-generated content at query #46
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda _: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = object()
    invariants = [lambda _: (False, "ERROR1"), lambda _: (True, None)]
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        assert e.error_codes == ("ERROR1",)
    else:
        assert False, "Expected InvariantException"

def test_check_global_invariants_multiple_errors():
    subject = object()
    invariants = [lambda _: (False, "ERROR1"), lambda _: (False, "ERROR2")]
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        assert e.error_codes == ("ERROR1", "ERROR2")
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #47
#--------------------------

```python
def test_sequence_field_optional_predicate():
    result = _sequence_field(CheckedPSet, int, True, [])
    assert result.factory is not None


# LLM-generated content at query #48
#--------------------------

```python
def test_predicate_at_line_4_evaluates_to_false():
    dct = {}
    bases = []
    name = "test_name"
    set_fields(dct, bases, name)
    assert not list(dct.items())


# LLM-generated content at query #49
#--------------------------

```python
def test__PField__init__assigns__factory_to_factory_parameter():
    type_ = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = lambda: 0
    serializer = str
    pfield = _PField(type_, invariant, initial, mandatory, factory, serializer)
    assert pfield._factory is factory


# LLM-generated content at query #50
#--------------------------

```python
def test_sequence_field_checked_pset():
    result = _sequence_field(CheckedPSet, int, False, [])
    assert result.type == {CheckedPSet}
    assert result.factory is not None
    assert result.mandatory is True
    assert result.invariant == PFIELD_NO_INVARIANT
    assert result.initial == CheckedPSet.create([])

def test_sequence_field_checked_pvector():
    result = _sequence_field(CheckedPVector, str, False, [])
    assert result.type == {CheckedPVector}
    assert result.factory is not None
    assert result.mandatory is True
    assert result.invariant == PFIELD_NO_INVARIANT
    assert result.initial == CheckedPVector.create([])

def test_sequence_field_optional():
    result = _sequence_field(CheckedPSet, int, True, None)
    assert result.type == {CheckedPSet, type(None)}
    assert result.factory is not None
    assert result.mandatory is True
    assert result.invariant == PFIELD_NO_INVARIANT
    assert result.initial is None

def test_sequence_field_with_invariant():
    def custom_invariant(x):
        return True, "OK"
    result = _sequence_field(CheckedPVector, str, False, [], invariant=custom_invariant)
    assert result.type == {CheckedPVector}
    assert result.factory is not None
    assert result.mandatory is True
    assert result.invariant == custom_invariant
    assert result.initial == CheckedPVector.create([])

def test_sequence_field_with_item_invariant():
    def custom_item_invariant(x):
        return True, "OK"
    result = _sequence_field(CheckedPSet, int, False, [], item_invariant=custom_item_invariant)
    assert result.type == {CheckedPSet}
    assert result.factory is not None
    assert result.mandatory is True
    assert result.invariant == PFIELD_NO_INVARIANT
    assert result.initial == CheckedPSet.create([])


# LLM-generated content at query #51
#--------------------------

```python
def test_predicate_evaluates_to_true():
    dct = {'a': _PField(), 'b': 1}
    bases = []
    name = 'test'
    set_fields(dct, bases, name)
    assert isinstance(dct['a'], _PField)


# LLM-generated content at query #52
#--------------------------

```python
def test_pmap_field_docstring_exists():
    assert pmap_field.__doc__ is not None
    assert len(pmap_field.__doc__) > 0


# LLM-generated content at query #53
#--------------------------

```python
def test_pmap_field_optional_false():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type, optional=False)
    assert result.factory is not None


# LLM-generated content at query #54
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


# LLM-generated content at query #55
#--------------------------

```python
def test_check_global_invariants_with_valid_invariants():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #56
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types

    # Mock data
    checked_class = type('MockCheckedClass', (), {})
    item_type = int
    data = [1, 2, 3]

    # Mock _seq_field_types to return a mock type
    mock_type = type('MockType', (), {'create': lambda self, data, **kwargs: data})
    _seq_field_types[checked_class, item_type] = mock_type

    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, data)

    # Assertions
    assert result == data


# LLM-generated content at query #57
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types

    # Mock data
    checked_class = type('MockCheckedClass', (), {})
    item_type = type('MockItemType', (), {})
    data = [1, 2, 3]

    # Mock _seq_field_types to return a mock type
    mock_type = type('MockType', (), {'create': lambda self, data, **kwargs: data})
    _seq_field_types[checked_class, item_type] = mock_type

    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, data)

    # Assertions
    assert result == data


# LLM-generated content at query #58
#--------------------------

```python
def test_check_global_invariants_with_no_errors():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #59
#--------------------------

```python
def test_check_global_invariants_with_valid_subject():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #60
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    value = CheckedType()
    value.serialize = lambda fmt: "serialized"
    assert serialize(PFIELD_NO_SERIALIZER, "format", value) == "serialized"

def test_serialize_with_serializer():
    serializer = lambda fmt, val: f"{fmt}:{val}"
    assert serialize(serializer, "json", "data") == "json:data"


# LLM-generated content at query #61
#--------------------------

```python
def test_sequence_field_creates_correct_field_with_non_optional_type():
    result = _sequence_field(CheckedPVector, int, False, [])
    assert isinstance(result, _PField)
    assert result.type == {_make_seq_field_type(CheckedPVector, int, PFIELD_NO_INVARIANT)}
    assert result.factory == _make_seq_field_type(CheckedPVector, int, PFIELD_NO_INVARIANT).create
    assert result.mandatory is True
    assert result.invariant == PFIELD_NO_INVARIANT
    assert result.initial == _make_seq_field_type(CheckedPVector, int, PFIELD_NO_INVARIANT).create([])

def test_sequence_field_creates_correct_field_with_optional_type():
    result = _sequence_field(CheckedPSet, str, True, set())
    assert isinstance(result, _PField)
    assert result.type == {_make_seq_field_type(CheckedPSet, str, PFIELD_NO_INVARIANT), type(None)}
    assert callable(result.factory)
    assert result.mandatory is True
    assert result.invariant == PFIELD_NO_INVARIANT
    assert result.initial == result.factory(set())

def test_sequence_field_with_custom_invariant():
    def custom_invariant(x):
        return len(x) > 0
    result = _sequence_field(CheckedPVector, int, False, [1, 2], invariant=custom_invariant)
    assert result.invariant == custom_invariant
    assert result.initial == _make_seq_field_type(CheckedPVector, int, PFIELD_NO_INVARIANT).create([1, 2])

def test_sequence_field_with_item_invariant():
    def item_invariant(x):
        return x > 0
    result = _sequence_field(CheckedPSet, int, False, {1, 2}, item_invariant=item_invariant)
    assert result.type == {_make_seq_field_type(CheckedPSet, int, item_invariant)}
    assert result.initial == _make_seq_field_type(CheckedPSet, int, item_invariant).create({1, 2})

def test_sequence_field_with_optional_and_none_initial():
    result = _sequence_field(CheckedPVector, str, True, None)
    assert result.type == {_make_seq_field_type(CheckedPVector, str, PFIELD_NO_INVARIANT), type(None)}
    assert result.initial is None


# LLM-generated content at query #62
#--------------------------

```python
def test__make_seq_field_type_creates_subclass_with_correct_name():
    class MockCheckedClass:
        _checked_types = (int, str)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert result.__name__ == "IntStrSeq"

def test__make_seq_field_type_stores_item_type_and_invariant():
    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: x > 0)
    assert result.__type__ == int
    assert result.__invariant__(5) is True
    assert result.__invariant__(-1) is False

def test__make_seq_field_type_caches_created_types():
    class MockCheckedClass:
        _checked_types = (int,)

    type1 = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    type2 = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert type1 is type2

def test__make_seq_field_type_preserves_checked_class_behavior():
    class MockCheckedClass:
        _checked_types = (int,)
        def __init__(self, items):
            self._items = items

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    instance = result([1, 2, 3])
    assert isinstance(instance, MockCheckedClass)
    assert instance._items == [1, 2, 3]


# LLM-generated content at query #63
#--------------------------

```python
def test_set_fields_predicate():
    dct = {'a': _PField(), 'b': 1}
    bases = []
    name = 'fields'
    set_fields(dct, bases, name)
    assert isinstance(dct['a'], _PField)


# LLM-generated content at query #64
#--------------------------

```python
def test__make_pmap_field_type_creates_new_type():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert isinstance(result, type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert result.__name__ == "StrToIntPMap"

def test__make_pmap_field_type_reuses_existing_type():
    key_type = str
    value_type = int
    first_result = _make_pmap_field_type(key_type, value_type)
    second_result = _make_pmap_field_type(key_type, value_type)
    assert first_result is second_result

def test__make_pmap_field_type_with_custom_types():
    key_type = "collections.abc.Hashable"
    value_type = "numbers.Number"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "HashableToNumberPMap"


# LLM-generated content at query #65
#--------------------------

```python
def test_is_field_ignore_extra_complaint_with_ignore_extra_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type

    class Field:
        type = int
        factory = lambda: None

    assert not is_field_ignore_extra_complaint(get_type(int), Field, False)


# LLM-generated content at query #66
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"

    assert isinstance(value, CheckedType)
    assert serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #67
#--------------------------

```python
def test_make_pmap_field_type_creates_new_subclass():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"
    assert result.__key_type__ == int
    assert result.__value_type__ == str
    assert issubclass(result, CheckedPMap)

def test_make_pmap_field_type_reuses_existing_subclass():
    key_type = int
    value_type = str
    first_result = _make_pmap_field_type(key_type, value_type)
    second_result = _make_pmap_field_type(key_type, value_type)
    assert first_result is second_result

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.int"
    value_type = "builtins.str"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"
    assert result.__key_type__ == int
    assert result.__value_type__ == str

def test_make_pmap_field_type_with_custom_type_names():
    key_type = "collections.OrderedDict"
    value_type = "decimal.Decimal"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "OrdereddictToDecimalPMap"
    assert result.__key_type__ == OrderedDict
    assert result.__value_type__ == Decimal


