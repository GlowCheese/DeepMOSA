####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR1",)


# LLM-generated content at query #2
#--------------------------

```python
def test_check_field_parameters_with_valid_types():
    field = Mock()
    field.type = [int, str]
    field.initial = PFIELD_NO_INITIAL
    field.invariant = lambda: True
    field.factory = lambda: None
    field.serializer = lambda x: x
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_type_parameter():
    field = Mock()
    field.type = [123]
    field.initial = PFIELD_NO_INITIAL
    field.invariant = lambda: True
    field.factory = lambda: None
    field.serializer = lambda x: x
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_invalid_initial_type():
    field = Mock()
    field.type = [int]
    field.initial = "not an int"
    field.invariant = lambda: True
    field.factory = lambda: None
    field.serializer = lambda x: x
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_valid_initial_type():
    field = Mock()
    field.type = [int, str]
    field.initial = 42
    field.invariant = lambda: True
    field.factory = lambda: None
    field.serializer = lambda x: x
    _check_field_parameters(field)

def test_check_field_parameters_with_callable_initial():
    field = Mock()
    field.type = [int]
    field.initial = lambda: 42
    field.invariant = lambda: True
    field.factory = lambda: None
    field.serializer = lambda x: x
    _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_invariant():
    field = Mock()
    field.type = [int]
    field.initial = PFIELD_NO_INITIAL
    field.invariant = "not callable"
    field.factory = lambda: None
    field.serializer = lambda x: x
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_factory():
    field = Mock()
    field.type = [int]
    field.initial = PFIELD_NO_INITIAL
    field.invariant = lambda: True
    field.factory = "not callable"
    field.serializer = lambda x: x
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_serializer():
    field = Mock()
    field.type = [int]
    field.initial = PFIELD_NO_INITIAL
    field.invariant = lambda: True
    field.factory = lambda: None
    field.serializer = "not callable"
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #3
#--------------------------

```python
def test_is_field_ignore_extra_complaint_with_false_ignore_extra():
    assert is_field_ignore_extra_complaint(str, MockField(type=str), False) is False

def test_is_field_ignore_extra_complaint_with_non_matching_type():
    assert is_field_ignore_extra_complaint(str, MockField(type=int), True) is False

def test_is_field_ignore_extra_complaint_with_matching_type_and_no_ignore_extra_param():
    field = MockField(type=str, factory=lambda: None)
    assert is_field_ignore_extra_complaint(str, field, True) is False

def test_is_field_ignore_extra_complaint_with_matching_type_and_ignore_extra_param():
    field = MockField(type=str, factory=lambda *, ignore_extra: None)
    assert is_field_ignore_extra_complaint(str, field, True) is True

def test_is_field_ignore_extra_complaint_with_set_type():
    field = MockField(type={str}, factory=lambda *, ignore_extra: None)
    assert is_field_ignore_extra_complaint(str, field, True) is True

def test_is_field_ignore_extra_complaint_with_empty_tuple_type():
    field = MockField(type=(), factory=lambda *, ignore_extra: None)
    assert is_field_ignore_extra_complaint(str, field, True) is False


# LLM-generated content at query #4
#--------------------------

```python
def test__sequence_field_with_checked_class_and_item_type():
    checked_class = CheckedPVector
    item_type = int
    optional = False
    initial = [1, 2, 3]
    result = _sequence_field(checked_class, item_type, optional, initial)
    assert isinstance(result, _PField)
    assert result.type == {CheckedPVector}
    assert result.factory == CheckedPVector.create
    assert result.mandatory is True
    assert result.initial == CheckedPVector.create([1, 2, 3])
    assert result.invariant == PFIELD_NO_INVARIANT

def test__sequence_field_with_optional_true():
    checked_class = CheckedPSet
    item_type = str
    optional = True
    initial = {'a', 'b'}
    result = _sequence_field(checked_class, item_type, optional, initial)
    assert isinstance(result, _PField)
    assert result.type == {CheckedPSet, type(None)}
    assert result.factory(None) is None
    assert result.mandatory is True
    assert result.initial == CheckedPSet.create({'a', 'b'})
    assert result.invariant == PFIELD_NO_INVARIANT

def test__sequence_field_with_item_invariant():
    checked_class = CheckedPVector
    item_type = int
    optional = False
    initial = [1, 2, 3]
    item_invariant = lambda x: x > 0
    result = _sequence_field(checked_class, item_type, optional, initial, item_invariant=item_invariant)
    assert isinstance(result, _PField)
    assert result.type == {CheckedPVector}
    assert result.factory == CheckedPVector.create
    assert result.mandatory is True
    assert result.initial == CheckedPVector.create([1, 2, 3])
    assert result.invariant == PFIELD_NO_INVARIANT

def test__sequence_field_with_invariant():
    checked_class = CheckedPSet
    item_type = str
    optional = False
    initial = {'a', 'b'}
    invariant = lambda x: len(x) > 0
    result = _sequence_field(checked_class, item_type, optional, initial, invariant=invariant)
    assert isinstance(result, _PField)
    assert result.type == {CheckedPSet}
    assert result.factory == CheckedPSet.create
    assert result.mandatory is True
    assert result.initial == CheckedPSet.create({'a', 'b'})
    assert result.invariant == invariant

def test__sequence_field_with_optional_type():
    checked_class = CheckedPVector
    item_type = int
    optional = True
    initial = [1, 2, 3]
    result = _sequence_field(checked_class, item_type, optional, initial)
    assert isinstance(result, _PField)
    assert result.type == {CheckedPVector, type(None)}
    assert result.factory(None) is None
    assert result.mandatory is True
    assert result.initial == CheckedPVector.create([1, 2, 3])
    assert result.invariant == PFIELD_NO_INVARIANT


# LLM-generated content at query #5
#--------------------------

```python
def test_types_to_names_with_single_type():
    assert _types_to_names((int,)) == "Int"

def test_types_to_names_with_multiple_types():
    assert _types_to_names((int, str, float)) == "IntStrFloat"

def test_types_to_names_with_type_string():
    assert _types_to_names(("builtins.int", "builtins.str")) == "IntStr"

def test_types_to_names_with_mixed_types_and_strings():
    assert _types_to_names((int, "builtins.str", float)) == "IntStrFloat"

def test_types_to_names_empty_tuple():
    assert _types_to_names(()) == ""

def test_types_to_names_with_custom_type():
    class CustomClass:
        pass
    assert _types_to_names((CustomClass,)) == "Customclass"


# LLM-generated content at query #6
#--------------------------

```python
def test_check_field_parameters_initial_valid_type():
    field = Field(type=[int], initial=42, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)


# LLM-generated content at query #7
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    checked_value = CheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

def test_serialize_with_custom_serializer():
    serializer = lambda fmt, val: f"serialized_{val}"
    result = serialize(serializer, "format", "value")
    assert result == "serialized_value"


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_evaluates_to_false():
    field = Mock()
    field.initial = "valid_initial"
    field.type = []
    assert not (field.initial is not PFIELD_NO_INITIAL and not callable(field.initial) and field.type and not any(isinstance(field.initial, t) for t in field.type))


# LLM-generated content at query #9
#--------------------------

```python
def test_check_field_parameters_with_valid_field():
    field = MockField(type=[int], initial=0, invariant=lambda x: True, factory=lambda: 0, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_type_parameter():
    field = MockField(type=[123], initial=0, invariant=lambda x: True, factory=lambda: 0, serializer=lambda x: x)
    with pytest.raises(TypeError, match="Type parameter expected, not <class 'int'>"):
        _check_field_parameters(field)

def test_check_field_parameters_with_invalid_initial_type():
    field = MockField(type=[int], initial="not an int", invariant=lambda x: True, factory=lambda: 0, serializer=lambda x: x)
    with pytest.raises(TypeError, match="Initial has invalid type <class 'str'>"):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_invariant():
    field = MockField(type=[int], initial=0, invariant="not callable", factory=lambda: 0, serializer=lambda x: x)
    with pytest.raises(TypeError, match="Invariant must be callable"):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_factory():
    field = MockField(type=[int], initial=0, invariant=lambda x: True, factory="not callable", serializer=lambda x: x)
    with pytest.raises(TypeError, match="Factory must be callable"):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_serializer():
    field = MockField(type=[int], initial=0, invariant=lambda x: True, factory=lambda: 0, serializer="not callable")
    with pytest.raises(TypeError, match="Serializer must be callable"):
        _check_field_parameters(field)


# LLM-generated content at query #10
#--------------------------

```python
def test_restore_seq_field_pickle():
    checked_class = type('MockCheckedClass', (), {})
    item_type = str
    data = ['a', 'b', 'c']

    result = _restore_seq_field_pickle(checked_class, item_type, data)

    assert isinstance(result, _seq_field_types[checked_class, item_type])
    assert result == data


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_evaluates_to_false():
    field = type('Field', (), {'type': [123], 'initial': None, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})()
    with pytest.raises(TypeError) as excinfo:
        _check_field_parameters(field)
    assert 'Type parameter expected, not' in str(excinfo.value)


# LLM-generated content at query #12
#--------------------------

```python
def test_check_field_parameters_with_invalid_type_parameter():
    field = type('Field', (), {'type': [123], 'initial': PFIELD_NO_INITIAL, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_valid_type_parameters():
    field = type('Field', (), {'type': [int, str], 'initial': PFIELD_NO_INITIAL, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_initial_type():
    field = type('Field', (), {'type': [int], 'initial': 'not_an_int', 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_valid_initial_type():
    field = type('Field', (), {'type': [int, str], 'initial': 42, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})
    _check_field_parameters(field)

def test_check_field_parameters_with_callable_initial():
    field = type('Field', (), {'type': [], 'initial': lambda: 42, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})
    _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_invariant():
    field = type('Field', (), {'type': [int], 'initial': PFIELD_NO_INITIAL, 'invariant': 'not_callable', 'factory': lambda: None, 'serializer': lambda x: x})
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_factory():
    field = type('Field', (), {'type': [int], 'initial': PFIELD_NO_INITIAL, 'invariant': lambda x: True, 'factory': 'not_callable', 'serializer': lambda x: x})
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_serializer():
    field = type('Field', (), {'type': [int], 'initial': PFIELD_NO_INITIAL, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': 'not_callable'})
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #13
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda x: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = object()
    invariants = [lambda x: (False, 'ERROR1'), lambda x: (True, None)]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('ERROR1',)

def test_check_global_invariants_multiple_errors():
    subject = object()
    invariants = [lambda x: (False, 'ERROR1'), lambda x: (False, 'ERROR2')]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ('ERROR1', 'ERROR2')


# LLM-generated content at query #14
#--------------------------

```python
def test_check_field_parameters_with_valid_field():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = lambda x: True
        factory = lambda: 0
        serializer = lambda x: str(x)

    _check_field_parameters(MockField())

def test_check_field_parameters_with_invalid_type_parameter():
    class MockField:
        type = [int, 123]  # 123 is not a type or str
        initial = 5
        invariant = lambda x: True
        factory = lambda: 0
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_with_invalid_initial_type():
    class MockField:
        type = [int, str]
        initial = 5.5  # float not in type
        invariant = lambda x: True
        factory = lambda: 0
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'float'>"

def test_check_field_parameters_with_non_callable_invariant():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = "not callable"
        factory = lambda: 0
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_with_non_callable_factory():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = lambda x: True
        factory = "not callable"
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_with_non_callable_serializer():
    class MockField:
        type = [int, str]
        initial = 5
        invariant = lambda x: True
        factory = lambda: 0
        serializer = "not callable"

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #15
#--------------------------

```python
def test__make_seq_field_type_creates_subclass_with_correct_name():
    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert result.__name__ == "IntSeq"

def test__make_seq_field_type_creates_subclass_with_correct_type():
    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert issubclass(result, MockCheckedClass)

def test__make_seq_field_type_creates_subclass_with_correct_attributes():
    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert result.__type__ == int
    assert result.__invariant__(5) is True

def test__make_seq_field_type_returns_cached_type():
    class MockCheckedClass:
        _checked_types = (int,)

    first_call = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    second_call = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    assert first_call is second_call

def test__make_seq_field_type_creates_different_types_for_different_item_types():
    class MockCheckedClass:
        _checked_types = (int,)

    int_type = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    str_type = _make_seq_field_type(MockCheckedClass, str, lambda x: True)
    assert int_type is not str_type
    assert int_type.__name__ == "IntSeq"
    assert str_type.__name__ == "StrSeq"

def test__make_seq_field_type_creates_different_types_for_different_checked_classes():
    class MockCheckedClass1:
        _checked_types = (int,)

    class MockCheckedClass2:
        _checked_types = (int,)

    type1 = _make_seq_field_type(MockCheckedClass1, int, lambda x: True)
    type2 = _make_seq_field_type(MockCheckedClass2, int, lambda x: True)
    assert type1 is not type2

def test__make_seq_field_type_creates_type_with_correct_reduce():
    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    instance = result([1, 2, 3])
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_seq_field_pickle
    assert reduced[1][0] == MockCheckedClass
    assert reduced[1][1] == int
    assert reduced[1][2] == [1, 2, 3]


# LLM-generated content at query #16
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = [("a", 1), ("b", 2)]
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert dict(result) == {"a": 1, "b": 2}


# LLM-generated content at query #17
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


# LLM-generated content at query #18
#--------------------------

```python
def test_pmap_field_creates_checked_pmap_with_correct_types():
    result = pmap_field(int, str)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_optional_creates_checked_pmap_with_none_type():
    result = pmap_field(int, str, optional=True)
    assert result.type == {_make_pmap_field_type(int, str), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory(None) is None
    assert result.factory({1: "a"}) == _make_pmap_field_type(int, str).create({1: "a"})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant_preserves_invariant():
    def test_invariant(pmap):
        return True, "test"

    result = pmap_field(int, str, invariant=test_invariant)
    assert result.invariant == wrap_invariant(test_invariant)


# LLM-generated content at query #19
#--------------------------

```python
def test_pmap_field_basic():
    result = pmap_field(int, str)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional():
    result = pmap_field(int, str, optional=True)
    assert result.type == {_make_pmap_field_type(int, str), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert callable(result.factory)
    assert result.factory(None) is None
    assert isinstance(result.factory({1: 'a'}), _make_pmap_field_type(int, str))
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def test_invariant(m):
        return len(m) > 0
    result = pmap_field(int, str, invariant=test_invariant)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert callable(result.invariant)
    assert result.invariant(_make_pmap_field_type(int, str)({1: 'a'})) == (True, ())
    assert result.invariant(_make_pmap_field_type(int, str)()) == (False, (_make_pmap_field_type(int, str)(),))

def test_pmap_field_optional_with_invariant():
    def test_invariant(m):
        return len(m) > 0
    result = pmap_field(int, str, optional=True, invariant=test_invariant)
    assert result.type == {_make_pmap_field_type(int, str), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert callable(result.factory)
    assert result.factory(None) is None
    assert isinstance(result.factory({1: 'a'}), _make_pmap_field_type(int, str))
    assert callable(result.invariant)
    assert result.invariant(_make_pmap_field_type(int, str)({1: 'a'})) == (True, ())
    assert result.invariant(_make_pmap_field_type(int, str)()) == (False, (_make_pmap_field_type(int, str)(),))


# LLM-generated content at query #20
#--------------------------

```python
def test_make_seq_field_type_creates_new_type():
    class MockCheckedClass:
        pass

    item_type = int
    item_invariant = lambda x: x > 0

    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)

    assert isinstance(result, type)
    assert issubclass(result, MockCheckedClass)
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant

def test_make_seq_field_type_reuses_existing_type():
    class MockCheckedClass:
        pass

    item_type = int
    item_invariant = lambda x: x > 0

    first_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)

    assert first_call is second_call

def test_make_seq_field_type_sets_correct_name():
    class MockCheckedClass:
        pass

    item_type = int
    item_invariant = lambda x: x > 0

    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)

    assert result.__name__.endswith(SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass])
    assert "Int" in result.__name__

def test_make_seq_field_type_with_string_type():
    class MockCheckedClass:
        pass

    item_type = "builtins.str"
    item_invariant = lambda x: len(x) > 0

    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)

    assert result.__type__ == str
    assert result.__invariant__ == item_invariant


# LLM-generated content at query #21
#--------------------------

```python
def test_check_field_parameters_with_non_callable_invariant():
    field = MockField(type=[int], initial=5, invariant="not_callable", factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError, match="Invariant must be callable"):
        _check_field_parameters(field)


# LLM-generated content at query #22
#--------------------------

```python
def test_check_global_invariants_with_valid_invariants():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #23
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


# LLM-generated content at query #24
#--------------------------

```python
def test_check_global_invariants_with_valid_invariants():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #25
#--------------------------

```python
def test_restore_seq_field_pickle_returns_correct_type():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types

    checked_class = type('MockCheckedClass', (), {})
    item_type = int
    data = [1, 2, 3]
    expected_type = _seq_field_types[checked_class, item_type]

    result = _restore_seq_field_pickle(checked_class, item_type, data)

    assert result == _restore_pickle(expected_type, data)


# LLM-generated content at query #26
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
    assert callable(result.factory)
    assert result.factory(None) is None
    assert isinstance(result.factory({1: "a"}), _make_pmap_field_type(int, str))
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def test_invariant(pmap):
        return True, "OK"
    result = pmap_field(int, str, invariant=test_invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory == True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert result.invariant == wrap_invariant(test_invariant)


# LLM-generated content at query #27
#--------------------------

```python
def test__make_seq_field_type_creates_subclass_with_correct_name():
    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, None)
    assert result.__name__ == "IntSeq"
    assert issubclass(result, MockCheckedClass)

def test__make_seq_field_type_sets_type_and_invariant():
    class MockCheckedClass:
        pass

    result = _make_seq_field_type(MockCheckedClass, str, lambda x: x)
    assert result.__type__ == str
    assert result.__invariant__ == lambda x: x

def test__make_seq_field_type_caches_result():
    class MockCheckedClass:
        pass

    first_call = _make_seq_field_type(MockCheckedClass, int, None)
    second_call = _make_seq_field_type(MockCheckedClass, int, None)
    assert first_call is second_call

def test__make_seq_field_type_different_types_different_classes():
    class MockCheckedClass:
        pass

    first_call = _make_seq_field_type(MockCheckedClass, int, None)
    second_call = _make_seq_field_type(MockCheckedClass, str, None)
    assert first_call is not second_call
    assert first_call.__type__ == int
    assert second_call.__type__ == str

def test__make_seq_field_type_with_string_type_name():
    class MockCheckedClass:
        pass

    result = _make_seq_field_type(MockCheckedClass, "builtins.int", None)
    assert result.__type__ == int
    assert result.__name__ == "IntSeq"


# LLM-generated content at query #28
#--------------------------

```python
def test_pmap_field_with_optional_false():
    result = pmap_field(key_type=str, value_type=int, optional=False)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create

def test_pmap_field_with_optional_true():
    result = pmap_field(key_type=str, value_type=int, optional=True)
    assert result.type == {_make_pmap_field_type(str, int), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

def test_pmap_field_with_invariant():
    def test_invariant(pmap):
        return True, "Test"
    result = pmap_field(key_type=str, value_type=int, optional=False, invariant=test_invariant)
    assert result.invariant == wrap_invariant(test_invariant)


# LLM-generated content at query #29
#--------------------------

```python
def test_pmap_field_optional_type_predicate():
    result = pmap_field(str, int, optional=True)
    assert result.type == optional_type(_make_pmap_field_type(str, int))


# LLM-generated content at query #30
#--------------------------

```python
def test_check_field_parameters_with_invalid_type():
    field = Mock()
    field.type = [123]
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_invalid_initial_type():
    field = Mock()
    field.type = [int, str]
    field.initial = 123.45
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_invariant():
    field = Mock()
    field.type = [int]
    field.initial = 123
    field.invariant = 123
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_factory():
    field = Mock()
    field.type = [int]
    field.initial = 123
    field.invariant = lambda x: True
    field.factory = 123
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_serializer():
    field = Mock()
    field.type = [int]
    field.initial = 123
    field.invariant = lambda x: True
    field.factory = lambda: 123
    field.serializer = 123
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_valid_parameters():
    field = Mock()
    field.type = [int, str]
    field.initial = 123
    field.invariant = lambda x: True
    field.factory = lambda: 123
    field.serializer = lambda x: str(x)
    _check_field_parameters(field)


# LLM-generated content at query #31
#--------------------------

```python
def test_set_fields_with_empty_bases():
    dct = {'field1': 'value1'}
    bases = []
    name = 'new_name'
    set_fields(dct, bases, name)
    assert dct == {'field1': 'value1', 'new_name': {}}

def test_set_fields_with_non_empty_bases():
    class Base1:
        __dict__ = {'field1': 'value1', 'field2': 'value2'}
    class Base2:
        __dict__ = {'field3': 'value3'}
    dct = {'field4': 'value4'}
    bases = [Base1, Base2]
    name = 'new_name'
    set_fields(dct, bases, name)
    assert dct == {'field4': 'value4', 'new_name': {'field1': 'value1', 'field2': 'value2', 'field3': 'value3'}}

def test_set_fields_with_pfield_instances():
    class _PField:
        pass
    dct = {'field1': _PField(), 'field2': 'value2'}
    bases = []
    name = 'new_name'
    set_fields(dct, bases, name)
    assert dct == {'field2': 'value2', 'new_name': {'field1': _PField()}}


# LLM-generated content at query #32
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


# LLM-generated content at query #33
#--------------------------

```python
def test_optional_factory_returns_none():
    result = pmap_field(str, int, optional=True).factory(None)
    assert result is None


# LLM-generated content at query #34
#--------------------------

```python
def test_pfield_constructor():
    type_value = (str,)
    invariant = lambda x: True
    initial = "initial"
    mandatory = True
    factory = lambda: "factory"
    serializer = lambda x: x

    pfield = _PField(type_value, invariant, initial, mandatory, factory, serializer)

    assert pfield.type == type_value
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #35
#--------------------------

```python
def test_pfield_constructor():
    type_val = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = lambda: 0
    serializer = str

    pfield = _PField(type_val, invariant, initial, mandatory, factory, serializer)

    assert pfield.type == type_val
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #36
#--------------------------

```python
def test_make_pmap_field_type_creates_new_class():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert isinstance(result, type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type

def test_make_pmap_field_type_returns_cached_class():
    key_type = int
    value_type = str
    first_result = _make_pmap_field_type(key_type, value_type)
    second_result = _make_pmap_field_type(key_type, value_type)
    assert first_result is second_result

def test_make_pmap_field_type_sets_correct_name():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"

def test_make_pmap_field_type_handles_string_types():
    key_type = "builtins.str"
    value_type = "builtins.int"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == str
    assert result.__value_type__ == int
    assert result.__name__ == "StrToIntPMap"

def test_make_pmap_field_type_reduce_method():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    instance = result({1: "one"})
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_pmap_field_pickle
    assert reduced[1][0] == key_type
    assert reduced[1][1] == value_type
    assert reduced[1][2] == {1: "one"}


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_at_line_5_evaluates_to_true():
    class _PField:
        pass

    class Base1:
        __dict__ = {'field1': _PField(), 'field2': 'value'}

    class Base2:
        __dict__ = {'field3': _PField(), 'field4': 'value'}

    dct = {'field1': _PField(), 'field5': 'value'}
    bases = [Base1, Base2]
    name = 'name'

    set_fields(dct, bases, name)

    assert isinstance(dct[name]['field1'], _PField)


# LLM-generated content at query #38
#--------------------------

```python
def test_check_global_invariants_with_valid_invariants():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #39
#--------------------------

```python
def test_set_fields_with_empty_bases():
    dct = {}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'test_name': {}}

def test_set_fields_with_single_base():
    dct = {}
    class Base:
        test_name = {'key1': 'value1'}
    bases = [Base]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'test_name': {'key1': 'value1'}}

def test_set_fields_with_multiple_bases():
    dct = {}
    class Base1:
        test_name = {'key1': 'value1'}
    class Base2:
        test_name = {'key2': 'value2'}
    bases = [Base1, Base2]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'test_name': {'key1': 'value1', 'key2': 'value2'}}

def test_set_fields_with_pfield_in_dct():
    dct = {'field1': _PField(), 'field2': 'value'}
    class Base:
        test_name = {'key1': 'value1'}
    bases = [Base]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'test_name': {'key1': 'value1', 'field1': _PField()}, 'field2': 'value'}

def test_set_fields_with_pfield_and_overlapping_keys():
    dct = {'key1': _PField(), 'key2': 'value'}
    class Base:
        test_name = {'key1': 'base_value'}
    bases = [Base]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {'test_name': {'key1': _PField()}, 'key2': 'value'}


# LLM-generated content at query #40
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


# LLM-generated content at query #41
#--------------------------

```python
def test_pmap_field_optional_factory_returns_none():
    result = pmap_field(str, int, optional=True).factory(None)
    assert result is None


# LLM-generated content at query #42
#--------------------------

```python
def test_pmap_field_optional_type_predicate():
    key_type = str
    value_type = int
    optional = True
    invariant = PFIELD_NO_INVARIANT
    result = pmap_field(key_type, value_type, optional, invariant)
    assert isinstance(result.type, tuple)
    assert len(result.type) == 2
    assert type(None) in result.type


# LLM-generated content at query #43
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"
    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #44
#--------------------------

```python
def test__make_seq_field_type_with_checked_class_and_item_type():
    checked_class = list
    item_type = int
    item_invariant = lambda x: x > 0
    result = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert result.__name__ == "IntList"
    assert result.__type__ == int
    assert result.__invariant__ == item_invariant
    assert issubclass(result, checked_class)

def test__make_seq_field_type_with_cached_type():
    checked_class = list
    item_type = int
    item_invariant = lambda x: x > 0
    first_call = _make_seq_field_type(checked_class, item_type, item_invariant)
    second_call = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert first_call is second_call

def test__make_seq_field_type_with_different_item_types():
    checked_class = list
    int_type = _make_seq_field_type(checked_class, int, lambda x: x > 0)
    str_type = _make_seq_field_type(checked_class, str, lambda x: len(x) > 0)
    assert int_type.__name__ == "IntList"
    assert str_type.__name__ == "StrList"
    assert int_type is not str_type

def test__make_seq_field_type_with_custom_checked_class():
    checked_class = tuple
    item_type = float
    item_invariant = lambda x: x >= 0.0
    result = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert result.__name__ == "FloatTuple"
    assert result.__type__ == float
    assert result.__invariant__ == item_invariant
    assert issubclass(result, checked_class)


# LLM-generated content at query #45
#--------------------------

```python
def test_make_pmap_field_type_creates_subclass_with_correct_name():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"

def test_make_pmap_field_type_sets_key_and_value_types():
    key_type = float
    value_type = bool
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == float
    assert result.__value_type__ == bool

def test_make_pmap_field_type_reuses_existing_type():
    key_type = str
    value_type = int
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


# LLM-generated content at query #46
#--------------------------

```python
def test_pfield_initialization_with_factory():
    pfield = _PField(type=int, invariant=lambda x: True, initial=0, mandatory=True, factory=lambda: 0, serializer=str)
    assert pfield._factory is not PFIELD_NO_FACTORY


# LLM-generated content at query #47
#--------------------------

```python
def test_predicate_at_line_3_evaluates_to_false():
    field = type('Field', (), {'type': [1]})()
    assert not (isinstance(1, type) or isinstance(1, str))


# LLM-generated content at query #48
#--------------------------

```python
def test_make_pmap_field_type_creates_new_class():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type

def test_make_pmap_field_type_reuses_existing_class():
    key_type = str
    value_type = int
    first_result = _make_pmap_field_type(key_type, value_type)
    second_result = _make_pmap_field_type(key_type, value_type)
    assert first_result is second_result

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.str"
    value_type = "builtins.int"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int

def test_make_pmap_field_type_with_custom_type_names():
    key_type = "my.module.CustomKey"
    value_type = "my.module.CustomValue"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "CustomKeyToCustomValuePMap"
    assert result.__key_type__ == _get_class(key_type)
    assert result.__value_type__ == _get_class(value_type)


# LLM-generated content at query #49
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _seq_field_types, _restore_seq_field_pickle

    # Mock data
    checked_class = type('MockCheckedClass', (), {})
    item_type = type('MockItemType', (), {})
    data = [1, 2, 3]

    # Mock _seq_field_types to return a mock type
    mock_type = type('MockType', (), {'create': lambda self, data, **kwargs: data})
    _seq_field_types[checked_class, item_type] = mock_type

    # Mock _restore_pickle to return the data
    _restore_pickle(mock_type, data) == data

    # Test the function
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    assert result == data


# LLM-generated content at query #50
#--------------------------

```python
def test_check_field_parameters_valid():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)

    field = MockField()
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = [123]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.initial = "not an int"
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'str'>"

def test_check_field_parameters_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.initial = PFIELD_NO_INITIAL
            self.invariant = 123
            self.factory = lambda: None
            self.serializer = lambda x: str(x)

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = 123
            self.serializer = lambda x: str(x)

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = 123

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #51
#--------------------------

```python
def test_check_field_parameters_with_invalid_type_parameter():
    from your_module import _check_field_parameters
    from your_module import PFIELD_NO_INITIAL
    from your_module import Field

    field = Field(type=[int, 123], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError for invalid type parameter"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_with_invalid_initial_type():
    from your_module import _check_field_parameters
    from your_module import PFIELD_NO_INITIAL
    from your_module import Field

    field = Field(type=[int, str], initial=1.5, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError for invalid initial type"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'float'>"

def test_check_field_parameters_with_non_callable_invariant():
    from your_module import _check_field_parameters
    from your_module import PFIELD_NO_INITIAL
    from your_module import Field

    field = Field(type=[int], initial=PFIELD_NO_INITIAL, invariant=123, factory=lambda: None, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError for non-callable invariant"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_with_non_callable_factory():
    from your_module import _check_field_parameters
    from your_module import PFIELD_NO_INITIAL
    from your_module import Field

    field = Field(type=[int], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=123, serializer=lambda x: x)
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError for non-callable factory"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_with_non_callable_serializer():
    from your_module import _check_field_parameters
    from your_module import PFIELD_NO_INITIAL
    from your_module import Field

    field = Field(type=[int], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda: None, serializer=123)
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError for non-callable serializer"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"

def test_check_field_parameters_with_valid_parameters():
    from your_module import _check_field_parameters
    from your_module import PFIELD_NO_INITIAL
    from your_module import Field

    field = Field(type=[int, str], initial=5, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)


# LLM-generated content at query #52
#--------------------------

```python
def test_valid_field_parameters():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x

    field = MockField()
    _check_field_parameters(field)

def test_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = (int, 5)  # 5 is not a type or str
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_invalid_initial_type():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = 5.5  # float not in type
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'float'>"

def test_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = 5
            self.invariant = "not callable"
            self.factory = lambda: None
            self.serializer = lambda x: x

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = "not callable"
            self.serializer = lambda x: x

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = 5
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = "not callable"

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #53
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized"

    value = CheckedType()
    PFIELD_NO_SERIALIZER = None
    result = serialize(PFIELD_NO_SERIALIZER, "format", value)
    assert result == "serialized"


# LLM-generated content at query #54
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle

    # Mock the _seq_field_types dictionary
    import sys
    sys.modules['pyrsistent._field_common']._seq_field_types = {
        (type, int): type('MockType', (), {'create': lambda self, data, **kwargs: data})
    }

    # Test with a mock checked_class and item_type
    result = _restore_seq_field_pickle(type, int, [1, 2, 3])
    assert result == [1, 2, 3]


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    value = CheckedType()
    assert serialize(PFIELD_NO_SERIALIZER, "format", value) == value.serialize("format")

def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f"serialized_{value}"
    assert serialize(custom_serializer, "format", "test_value") == "serialized_test_value"


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_restore_pmap_field_pickle_calls_restore_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    _restore_pmap_field_pickle(key_type, value_type, data)
    _restore_pickle.assert_called_once_with(_pmap_field_types[key_type, value_type], data)


# LLM-generated content at query #4
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    check_type(TestClass, Field(), "test_field", 123)

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
        type = (int, float)

    check_type(TestClass, Field(), "test_field", 123)
    check_type(TestClass, Field(), "test_field", 123.45)

def test_check_type_with_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ("builtins.int",)

    check_type(TestClass, Field(), "test_field", 123)


# LLM-generated content at query #5
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
    assert result.__key_type__ == str
    assert result.__value_type__ == int
    assert result.__name__ == "StrToIntPMap"

def test_make_pmap_field_type_custom_class_names():
    class CustomKey:
        pass
    class CustomValue:
        pass
    result = _make_pmap_field_type(CustomKey, CustomValue)
    assert result.__name__ == "CustomKeyToCustomValuePMap"


# LLM-generated content at query #6
#--------------------------

```python
def test_valid_field_parameters():
    class MockField:
        type = (int, str)
        initial = 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: str(x)

    _check_field_parameters(MockField())

def test_invalid_type_parameter():
    class MockField:
        type = (int, 123)  # 123 is not a type or str
        initial = 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_invalid_initial_type():
    class MockField:
        type = (int, str)
        initial = 5.5  # float not in type
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'float'>"

def test_non_callable_invariant():
    class MockField:
        type = (int, str)
        initial = 5
        invariant = "not callable"
        factory = lambda: None
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_non_callable_factory():
    class MockField:
        type = (int, str)
        initial = 5
        invariant = lambda x: True
        factory = "not callable"
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_non_callable_serializer():
    class MockField:
        type = (int, str)
        initial = 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = "not callable"

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_false():
    field = Mock()
    field.initial = PFIELD_NO_INITIAL
    field.type = [int]
    field.invariant = lambda x: True
    field.factory = lambda: 0
    field.serializer = lambda x: str(x)

    _check_field_parameters(field)


# LLM-generated content at query #8
#--------------------------

```python
def test_pfield_initialization():
    pfield = _PField(type="test_type", invariant="test_invariant", initial="test_initial", mandatory=True, factory="test_factory", serializer="test_serializer")
    assert pfield.type == "test_type"
    assert pfield.invariant == "test_invariant"
    assert pfield.initial == "test_initial"
    assert pfield.mandatory is True
    assert pfield._factory == "test_factory"
    assert pfield.serializer == "test_serializer"


# LLM-generated content at query #9
#--------------------------

```python
def test_is_type_cls_with_set():
    assert is_type_cls(object, {int, str})

def test_is_type_cls_with_empty_tuple():
    assert not is_type_cls(object, ())

def test_is_type_cls_with_single_type_in_tuple():
    assert is_type_cls(int, (int,))

def test_is_type_cls_with_multiple_types_in_tuple():
    assert is_type_cls(object, (int, str))

def test_is_type_cls_with_non_matching_type():
    assert not is_type_cls(str, (int,))

def test_is_type_cls_with_type_object():
    assert is_type_cls(object, object)

def test_is_type_cls_with_string_type_name():
    assert is_type_cls(Exception, 'builtins.ValueError')

def test_is_type_cls_with_non_matching_string_type_name():
    assert not is_type_cls(ValueError, 'builtins.Exception')


# LLM-generated content at query #10
#--------------------------

```python
def test__make_seq_field_type_returns_existing_type():
    checked_class = list
    item_type = int
    item_invariant = lambda x: x > 0
    type1 = _make_seq_field_type(checked_class, item_type, item_invariant)
    type2 = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert type1 is type2

def test__make_seq_field_type_creates_new_type():
    checked_class = list
    item_type = int
    item_invariant = lambda x: x > 0
    new_type = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert issubclass(new_type, checked_class)
    assert new_type.__type__ == item_type
    assert new_type.__invariant__ == item_invariant

def test__make_seq_field_type_sets_correct_name():
    checked_class = list
    item_type = int
    item_invariant = lambda x: x > 0
    new_type = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert new_type.__name__ == "IntList"

def test__make_seq_field_type_with_string_type():
    checked_class = tuple
    item_type = str
    item_invariant = lambda x: len(x) > 0
    new_type = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert new_type.__name__ == "StrTuple"

def test__make_seq_field_type_with_custom_type():
    checked_class = list
    item_type = "collections.OrderedDict"
    item_invariant = lambda x: True
    new_type = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert new_type.__name__ == "OrdereddictList"

def test__make_seq_field_type_pickle_support():
    checked_class = list
    item_type = int
    item_invariant = lambda x: x > 0
    new_type = _make_seq_field_type(checked_class, item_type, item_invariant)
    instance = new_type([1, 2, 3])
    assert instance.__reduce__() == (_restore_seq_field_pickle, (checked_class, item_type, [1, 2, 3]))


# LLM-generated content at query #11
#--------------------------

```python
def test_make_pmap_field_type_creates_new_class():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int

def test_make_pmap_field_type_reuses_existing_class():
    key_type = str
    value_type = int
    first_result = _make_pmap_field_type(key_type, value_type)
    second_result = _make_pmap_field_type(key_type, value_type)
    assert first_result is second_result

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.str"
    value_type = "builtins.int"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


# LLM-generated content at query #12
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return "serialized"

    PFIELD_NO_SERIALIZER = None
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER

    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #13
#--------------------------

```python
def test_factory_property_returns_no_factory_when_type_is_not_checked_type():
    pfield = _PField(type=(), invariant=None, initial=None, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=None)
    assert pfield.factory is PFIELD_NO_FACTORY


# LLM-generated content at query #14
#--------------------------

```python
def test_make_pmap_field_type_creates_new_class():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"
    assert result.__key_type__ == int
    assert result.__value_type__ == str

def test_make_pmap_field_type_reuses_existing_class():
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

def test_make_pmap_field_type_with_custom_types():
    class CustomKey:
        pass

    class CustomValue:
        pass

    result = _make_pmap_field_type(CustomKey, CustomValue)
    assert result.__name__ == "CustomKeyToCustomValuePMap"
    assert result.__key_type__ == CustomKey
    assert result.__value_type__ == CustomValue


# LLM-generated content at query #15
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"Serialized {format}"

    value = CheckedType()
    PFIELD_NO_SERIALIZER = None
    format = "test_format"

    result = serialize(PFIELD_NO_SERIALIZER, format, value)

    assert result == "Serialized test_format"


# LLM-generated content at query #16
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda _: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = object()
    invariants = [lambda _: (False, "E1")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("E1",)

def test_check_global_invariants_multiple_errors():
    subject = object()
    invariants = [lambda _: (False, "E1"), lambda _: (False, "E2")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("E1", "E2")

def test_check_global_invariants_mixed_results():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (False, "E1"), lambda _: (True, None)]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("E1",)


# LLM-generated content at query #17
#--------------------------

```python
def test_pfield_initialization():
    pfield = _PField(type=int, invariant=lambda x: x > 0, initial=0, mandatory=True, factory=None, serializer=str)
    assert pfield.type == int
    assert pfield.invariant(5) == True
    assert pfield.initial == 0
    assert pfield.mandatory == True
    assert pfield._factory == None
    assert pfield.serializer == str


# LLM-generated content at query #18
#--------------------------

```python
def test_is_field_ignore_extra_complaint_with_ignore_extra_false():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field

    field = Field('test', type=int, factory=lambda: 0)
    assert not is_field_ignore_extra_complaint(int, field, False)

def test_is_field_ignore_extra_complaint_with_non_matching_type():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field

    field = Field('test', type=str, factory=lambda: '')
    assert not is_field_ignore_extra_complaint(int, field, True)

def test_is_field_ignore_extra_complaint_with_matching_type_and_no_ignore_extra_param():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field

    field = Field('test', type=int, factory=lambda: 0)
    assert not is_field_ignore_extra_complaint(int, field, True)

def test_is_field_ignore_extra_complaint_with_matching_type_and_ignore_extra_param():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field

    field = Field('test', type=int, factory=lambda *, ignore_extra: 0)
    assert is_field_ignore_extra_complaint(int, field, True)

def test_is_field_ignore_extra_complaint_with_set_type():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._field_common import Field

    field = Field('test', type={int, str}, factory=lambda: 0)
    assert not is_field_ignore_extra_complaint(int, field, True)


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    check_type(TestClass, Field(), "test_field", 123)

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

    check_type(TestClass, Field(), "test_field", 123)
    check_type(TestClass, Field(), "test_field", "string")

def test_check_type_with_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ("builtins.int",)

    check_type(TestClass, Field(), "test_field", 123)

def test_check_type_with_invalid_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ("builtins.int",)

    try:
        check_type(TestClass, Field(), "test_field", "not_an_int")
        assert False, "Expected PTypeError"
    except PTypeError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    assert is_field_ignore_extra_complaint(str, object(), False) is False

def test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_a_subset_of_type_cls():
    assert is_field_ignore_extra_complaint(str, object(), True) is False

def test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param():
    field = object()
    field.type = {str}
    field.factory = lambda: None
    assert is_field_ignore_extra_complaint(str, field, True) is False

def test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_are_met():
    field = object()
    field.type = {str}
    field.factory = lambda ignore_extra: None
    assert is_field_ignore_extra_complaint(str, field, True) is True


# LLM-generated content at query #22
#--------------------------

```python
def test__make_seq_field_type_creates_new_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, SEQ_FIELD_TYPE_SUFFIXES

    class MockCheckedClass:
        _checked_types = (int,)

    _seq_field_types.clear()
    SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass] = "Seq"

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)

    assert result.__name__ == "IntSeq"
    assert result.__type__ == int
    assert result.__invariant__(5)
    assert (MockCheckedClass, int) in _seq_field_types
    assert _seq_field_types[(MockCheckedClass, int)] is result

def test__make_seq_field_type_returns_cached_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, SEQ_FIELD_TYPE_SUFFIXES

    class MockCheckedClass:
        _checked_types = (int,)

    _seq_field_types.clear()
    SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass] = "Seq"

    first_call = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    second_call = _make_seq_field_type(MockCheckedClass, int, lambda x: True)

    assert first_call is second_call

def test__make_seq_field_type_with_string_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, SEQ_FIELD_TYPE_SUFFIXES

    class MockCheckedClass:
        _checked_types = (str,)

    _seq_field_types.clear()
    SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass] = "Seq"

    result = _make_seq_field_type(MockCheckedClass, "builtins.str", lambda x: True)

    assert result.__name__ == "StrSeq"
    assert result.__type__ == str
    assert result.__invariant__("test")
    assert ("builtins.str",) in _seq_field_types
    assert _seq_field_types[("builtins.str",)] is result


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_evaluates_to_false():
    field = MockField(type=[123], initial=None, invariant=lambda: True, factory=lambda: True, serializer=lambda: True)
    assert _check_field_parameters(field) == False


# LLM-generated content at query #24
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass: pass
    class Field:
        type = (int,)
    check_type(TestClass, Field(), 'test_field', 42)

def test_check_type_with_invalid_type():
    class TestClass: pass
    class Field:
        type = (int,)
    try:
        check_type(TestClass, Field(), 'test_field', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.destination_cls == TestClass
        assert e.field_name == 'test_field'
        assert e.expected_type == (int,)
        assert e.actual_type == str

def test_check_type_with_no_type_specified():
    class TestClass: pass
    class Field:
        type = None
    check_type(TestClass, Field(), 'test_field', 'any_value')

def test_check_type_with_multiple_valid_types():
    class TestClass: pass
    class Field:
        type = (int, str)
    check_type(TestClass, Field(), 'test_field', 42)
    check_type(TestClass, Field(), 'test_field', 'string')

def test_check_type_with_string_type_name():
    class TestClass: pass
    class Field:
        type = ('builtins.int',)
    check_type(TestClass, Field(), 'test_field', 42)


# LLM-generated content at query #25
#--------------------------

```python
def test__make_seq_field_type_with_built_in_type():
    class TestCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(TestCheckedClass, int, lambda x: True)
    assert result.__name__ == "IntSeq"
    assert result.__type__ == int
    assert result.__invariant__(5)

def test__make_seq_field_type_with_string_type():
    class TestCheckedClass:
        _checked_types = ("builtins.int",)

    result = _make_seq_field_type(TestCheckedClass, "builtins.int", lambda x: True)
    assert result.__name__ == "IntSeq"
    assert result.__type__ == int
    assert result.__invariant__(5)

def test__make_seq_field_type_caching():
    class TestCheckedClass:
        _checked_types = (int,)

    type1 = _make_seq_field_type(TestCheckedClass, int, lambda x: True)
    type2 = _make_seq_field_type(TestCheckedClass, int, lambda x: True)
    assert type1 is type2

def test__make_seq_field_type_pickle_support():
    class TestCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(TestCheckedClass, int, lambda x: True)
    instance = result([1, 2, 3])
    assert instance.__reduce__() == (_restore_seq_field_pickle, (TestCheckedClass, int, [1, 2, 3]))


# LLM-generated content at query #26
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = object()
    invariants = [lambda _: (False, "ERROR1"), lambda _: (True, None)]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR1",)

def test_check_global_invariants_multiple_errors():
    subject = object()
    invariants = [lambda _: (False, "ERROR1"), lambda _: (False, "ERROR2")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR1", "ERROR2")


# LLM-generated content at query #27
#--------------------------

```python
def test_pfield_initialization():
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


# LLM-generated content at query #28
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"
    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #29
#--------------------------

```python
def test_factory_assignment():
    pfield = _PField(type=(int,), invariant=lambda x: True, initial=0, mandatory=True, factory=None, serializer=str)
    assert pfield._factory is None


# LLM-generated content at query #30
#--------------------------

```python
def test_pmap_field_basic():
    result = pmap_field(int, str)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert callable(result.factory)
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional():
    result = pmap_field(int, str, optional=True)
    assert result.type == optional(_make_pmap_field_type(int, str))
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert callable(result.factory)
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def test_invariant(x):
        return True, "test"
    result = pmap_field(int, str, invariant=test_invariant)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert callable(result.factory)
    assert callable(result.invariant)

def test_pmap_field_factory_none():
    result = pmap_field(int, str, optional=True)
    assert result.factory(None) is None

def test_pmap_field_factory_with_value():
    result = pmap_field(int, str, optional=True)
    test_map = {1: "a", 2: "b"}
    assert isinstance(result.factory(test_map), _make_pmap_field_type(int, str))


# LLM-generated content at query #31
#--------------------------

```python
def test_restore_seq_field_pickle():
    _seq_field_types[TestClass, int] = TestType
    data = [1, 2, 3]
    result = _restore_seq_field_pickle(TestClass, int, data)
    assert result == TestType.create(data, _factory_fields=set())


# LLM-generated content at query #32
#--------------------------

```python
def test_pmap_field_basic():
    result = pmap_field(int, str)
    assert isinstance(result.type, set)
    assert len(result.type) == 1
    assert result.mandatory is True
    assert result.initial == TheMap()
    assert result.factory == TheMap.create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional():
    result = pmap_field(int, str, optional=True)
    assert isinstance(result.type, set)
    assert len(result.type) == 2
    assert type(None) in result.type
    assert result.mandatory is True
    assert result.initial == TheMap()
    assert result.factory(None) is None
    assert result.factory({1: "a"}) == TheMap.create({1: "a"})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def test_invariant(x):
        return True, "test"
    result = pmap_field(int, str, invariant=test_invariant)
    assert result.invariant == wrap_invariant(test_invariant)

def test_pmap_field_optional_with_invariant():
    def test_invariant(x):
        return True, "test"
    result = pmap_field(int, str, optional=True, invariant=test_invariant)
    assert result.invariant == wrap_invariant(test_invariant)


# LLM-generated content at query #33
#--------------------------

```python
def test_check_field_parameters_with_valid_type():
    field = type('Field', (), {'type': [int], 'initial': PFIELD_NO_INITIAL, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_type():
    field = type('Field', (), {'type': [123], 'initial': PFIELD_NO_INITIAL, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})
    try:
        _check_field_parameters(field)
    except TypeError as e:
        assert str(e) == 'Type parameter expected, not <class \'int\'>'

def test_check_field_parameters_with_valid_initial():
    field = type('Field', (), {'type': [int], 'initial': 5, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_initial():
    field = type('Field', (), {'type': [int], 'initial': 'string', 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})
    try:
        _check_field_parameters(field)
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'

def test_check_field_parameters_with_non_callable_invariant():
    field = type('Field', (), {'type': [int], 'initial': PFIELD_NO_INITIAL, 'invariant': 'not callable', 'factory': lambda: None, 'serializer': lambda x: x})
    try:
        _check_field_parameters(field)
    except TypeError as e:
        assert str(e) == 'Invariant must be callable'

def test_check_field_parameters_with_non_callable_factory():
    field = type('Field', (), {'type': [int], 'initial': PFIELD_NO_INITIAL, 'invariant': lambda x: True, 'factory': 'not callable', 'serializer': lambda x: x})
    try:
        _check_field_parameters(field)
    except TypeError as e:
        assert str(e) == 'Factory must be callable'

def test_check_field_parameters_with_non_callable_serializer():
    field = type('Field', (), {'type': [int], 'initial': PFIELD_NO_INITIAL, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': 'not callable'})
    try:
        _check_field_parameters(field)
    except TypeError as e:
        assert str(e) == 'Serializer must be callable'


# LLM-generated content at query #34
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
    except PTypeError as e:
        assert e.args[0] == "Invalid type for field TestClass.test_field, was str"

def test_check_type_with_multiple_valid_types():
    class TestClass:
        pass

    class Field:
        type = (int, str)

    check_type(TestClass, Field(), "test_field", 42)
    check_type(TestClass, Field(), "test_field", "a_string")

def test_check_type_with_no_type_specified():
    class TestClass:
        pass

    class Field:
        type = None

    check_type(TestClass, Field(), "test_field", 42)
    check_type(TestClass, Field(), "test_field", "a_string")
    check_type(TestClass, Field(), "test_field", {"a": "dict"})

def test_check_type_with_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ("builtins.int",)

    check_type(TestClass, Field(), "test_field", 42)

def test_check_type_with_invalid_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ("builtins.int",)

    try:
        check_type(TestClass, Field(), "test_field", "not_an_int")
    except PTypeError as e:
        assert e.args[0] == "Invalid type for field TestClass.test_field, was str"


# LLM-generated content at query #35
#--------------------------

```python
def test_check_global_invariants_with_valid_invariants():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #36
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestField:
        type = (int,)

    class TestClass:
        pass

    check_type(TestClass, TestField(), 'test_field', 42)


# LLM-generated content at query #37
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"
    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #38
#--------------------------

```python
def test_serialize_with_checked_type_and_pfield_no_serializer():
    assert isinstance(CheckedType(), CheckedType) and PFIELD_NO_SERIALIZER is PFIELD_NO_SERIALIZER


# LLM-generated content at query #39
#--------------------------

```python
def test_check_field_parameters_with_valid_field():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = 10
            self.invariant = lambda x: True
            self.factory = lambda: 0
            self.serializer = lambda x: str(x)

    field = Field()
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_type_parameter():
    class Field:
        def __init__(self):
            self.type = [123]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: 0
            self.serializer = lambda x: str(x)

    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_with_invalid_initial_type():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = 12.5
            self.invariant = lambda x: True
            self.factory = lambda: 0
            self.serializer = lambda x: str(x)

    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'float'>"

def test_check_field_parameters_with_non_callable_invariant():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = 123
            self.factory = lambda: 0
            self.serializer = lambda x: str(x)

    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_with_non_callable_factory():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = 123
            self.serializer = lambda x: str(x)

    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_with_non_callable_serializer():
    class Field:
        def __init__(self):
            self.type = [int, str]
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: 0
            self.serializer = 123

    field = Field()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #40
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _seq_field_types, _restore_seq_field_pickle

    # Mock data
    checked_class = type('MockCheckedClass', (), {})
    item_type = int
    data = [1, 2, 3]

    # Mock _seq_field_types to return a mock type
    mock_type = type('MockType', (), {'create': lambda self, data, **kwargs: data})
    _seq_field_types[checked_class, item_type] = mock_type

    # Mock _restore_pickle to return the data as-is
    with patch('pyrsistent._field_common._restore_pickle', side_effect=lambda cls, data: data):
        result = _restore_seq_field_pickle(checked_class, item_type, data)

    assert result == data


# LLM-generated content at query #41
#--------------------------

```python
def test_check_field_parameters_with_valid_type():
    field = Mock(type=[int], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_type():
    field = Mock(type=[123], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_valid_initial():
    field = Mock(type=[int], initial=5, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_initial():
    field = Mock(type=[int], initial="not an int", invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_callable_initial():
    field = Mock(type=[int], initial=lambda: 5, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_invariant():
    field = Mock(type=[int], initial=PFIELD_NO_INITIAL, invariant="not callable", factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_factory():
    field = Mock(type=[int], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory="not callable", serializer=lambda x: x)
    with pytest.raises(TypeError):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_serializer():
    field = Mock(type=[int], initial=PFIELD_NO_INITIAL, invariant=lambda x: True, factory=lambda: None, serializer="not callable")
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #42
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    assert is_field_ignore_extra_complaint(str, type('Field', (), {'type': str, 'factory': lambda: None}), False) is False

def test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_subclass():
    assert is_field_ignore_extra_complaint(str, type('Field', (), {'type': int, 'factory': lambda: None}), True) is False

def test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param():
    assert is_field_ignore_extra_complaint(str, type('Field', (), {'type': str, 'factory': lambda: None}), True) is False

def test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_are_met():
    def factory(ignore_extra=True):
        return None
    assert is_field_ignore_extra_complaint(str, type('Field', (), {'type': str, 'factory': factory}), True) is True

def test_is_field_ignore_extra_complaint_works_with_set_type():
    assert is_field_ignore_extra_complaint(str, type('Field', (), {'type': {str}, 'factory': lambda: None}), True) is False

def test_is_field_ignore_extra_complaint_works_with_empty_tuple_type():
    assert is_field_ignore_extra_complaint(str, type('Field', (), {'type': tuple(), 'factory': lambda: None}), True) is False


# LLM-generated content at query #43
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


# LLM-generated content at query #44
#--------------------------

```python
def test_restore_pmap_field_pickle_calls_restore_pickle_with_correct_args():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}

    result = _restore_pmap_field_pickle(key_type, value_type, data)

    assert result == _restore_pickle(_pmap_field_types[key_type, value_type], data)


# LLM-generated content at query #45
#--------------------------

```python
def test_set_fields_with_empty_bases_and_empty_dct():
    dct = {}
    bases = []
    name = "test_name"
    set_fields(dct, bases, name)
    assert dct == {name: {}}

def test_set_fields_with_non_empty_bases_and_empty_dct():
    class Base1:
        test_name = {"key1": "value1"}

    class Base2:
        test_name = {"key2": "value2"}

    dct = {}
    bases = [Base1, Base2]
    name = "test_name"
    set_fields(dct, bases, name)
    assert dct == {name: {"key1": "value1", "key2": "value2"}}

def test_set_fields_with_pfield_in_dct():
    class _PField:
        pass

    dct = {"field1": _PField(), "field2": "value2"}
    bases = []
    name = "test_name"
    set_fields(dct, bases, name)
    assert dct == {name: {"field1": dct["field1"]}, "field2": "value2"}


# LLM-generated content at query #46
#--------------------------

```python
def test_restore_pmap_field_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _pmap_field_types

    # Mock data
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}

    # Mock _pmap_field_types
    mock_type = type("MockType", (), {"create": lambda self, data, **kwargs: data})
    _pmap_field_types[key_type, value_type] = mock_type

    # Call function
    result = _restore_pmap_field_pickle(key_type, value_type, data)

    # Assertions
    assert result == data


# LLM-generated content at query #47
#--------------------------

```python
def test_predicate_evaluates_to_true():
    dct = {'a': _PField(), 'b': 1}
    bases = []
    name = 'fields'
    set_fields(dct, bases, name)
    assert isinstance(dct['a'], _PField)


# LLM-generated content at query #48
#--------------------------

```python
def test_valid_field_parameters():
    class MockField:
        type = [int, str]
        initial = 10
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x

    _check_field_parameters(MockField())

def test_invalid_type_parameter():
    class MockField:
        type = [123]
        initial = PFIELD_NO_INITIAL
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_invalid_initial_type():
    class MockField:
        type = [int]
        initial = "not an int"
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'str'>"

def test_non_callable_invariant():
    class MockField:
        type = [int]
        initial = PFIELD_NO_INITIAL
        invariant = "not callable"
        factory = lambda: None
        serializer = lambda x: x

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_non_callable_factory():
    class MockField:
        type = [int]
        initial = PFIELD_NO_INITIAL
        invariant = lambda x: True
        factory = "not callable"
        serializer = lambda x: x

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_non_callable_serializer():
    class MockField:
        type = [int]
        initial = PFIELD_NO_INITIAL
        invariant = lambda x: True
        factory = lambda: None
        serializer = "not callable"

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #49
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_true_but_type_cls_not_compatible():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type

    class MockField:
        type = 'invalid.type.name'
        factory = lambda: None

    assert not is_field_ignore_extra_complaint(get_type(int), MockField, True)


# LLM-generated content at query #50
#--------------------------

```python
def test_check_global_invariants_with_valid_invariants():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #51
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"

    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #52
#--------------------------

```python
def test_set_fields_predicate():
    dct = {'a': _PField(), 'b': 1}
    bases = []
    name = 'fields'
    set_fields(dct, bases, name)
    assert 'a' not in dct and 'fields' in dct and 'a' in dct['fields']


# LLM-generated content at query #53
#--------------------------

```python
def test_restore_seq_field_pickle_calls_restore_pickle_with_correct_args():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types

    checked_class = type('TestClass', (), {})
    item_type = int
    data = [1, 2, 3]

    _seq_field_types[checked_class, item_type] = type('TestType', (), {'create': lambda self, data, _factory_fields: data})

    result = _restore_seq_field_pickle(checked_class, item_type, data)

    assert result == data
    assert _restore_pickle.called_with(_seq_field_types[checked_class, item_type], data)


# LLM-generated content at query #54
#--------------------------

```python
def test_predicate_evaluates_to_false():
    field = type('Field', (), {'type': [123], 'initial': 'value', 'invariant': lambda: True, 'factory': lambda: True, 'serializer': lambda: True})()
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #55
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (False, "ERROR1"), lambda _: (False, "ERROR2")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR1", "ERROR2")


# LLM-generated content at query #56
#--------------------------

```python
def test_pmap_field_docstring_exists():
    assert pmap_field.__doc__ is not None
    assert len(pmap_field.__doc__) > 0


# LLM-generated content at query #57
#--------------------------

```python
def test_make_seq_field_type_with_builtin_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, SEQ_FIELD_TYPE_SUFFIXES

    checked_class = list
    item_type = int
    item_invariant = lambda x: x > 0

    result = _make_seq_field_type(checked_class, item_type, item_invariant)

    assert result.__name__ == "Int" + SEQ_FIELD_TYPE_SUFFIXES[checked_class]
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant
    assert (checked_class, item_type) in _seq_field_types
    assert _seq_field_types[(checked_class, item_type)] == result

def test_make_seq_field_type_with_custom_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types, SEQ_FIELD_TYPE_SUFFIXES

    checked_class = list
    item_type = "collections.abc.Sequence"
    item_invariant = lambda x: len(x) > 0

    result = _make_seq_field_type(checked_class, item_type, item_invariant)

    assert result.__name__ == "Sequence" + SEQ_FIELD_TYPE_SUFFIXES[checked_class]
    assert result.__type__ == get_type(item_type)
    assert result.__invariant__ == item_invariant
    assert (checked_class, item_type) in _seq_field_types
    assert _seq_field_types[(checked_class, item_type)] == result

def test_make_seq_field_type_caching():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types

    checked_class = tuple
    item_type = str
    item_invariant = lambda x: len(x) > 0

    result1 = _make_seq_field_type(checked_class, item_type, item_invariant)
    result2 = _make_seq_field_type(checked_class, item_type, item_invariant)

    assert result1 is result2
    assert len(_seq_field_types) == 1

def test_make_seq_field_type_reduce():
    from pyrsistent._field_common import _make_seq_field_type, _restore_seq_field_pickle

    checked_class = list
    item_type = float
    item_invariant = lambda x: x != 0.0

    result = _make_seq_field_type(checked_class, item_type, item_invariant)
    instance = result([1.0, 2.0, 3.0])

    reduced = instance.__reduce__()
    assert reduced[0] == _restore_seq_field_pickle
    assert reduced[1][0] == checked_class
    assert reduced[1][1] == item_type
    assert reduced[1][2] == [1.0, 2.0, 3.0]


# LLM-generated content at query #58
#--------------------------

```python
def test_pmap_field_predicate():
    assert pmap_field.__doc__ is not None


# LLM-generated content at query #59
#--------------------------

```python
def test_pmap_field_optional_false():
    result = pmap_field(str, int, optional=False)
    assert not result.factory is None


# LLM-generated content at query #60
#--------------------------

```python
def test__make_seq_field_type_creates_new_type():
    from pyrsistent._field_common import _make_seq_field_type
    from pyrsistent._checked_types import get_type

    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, None)
    assert isinstance(result, type)
    assert issubclass(result, MockCheckedClass)
    assert result.__name__ == "IntSeq"
    assert result.__type__ == int
    assert result.__invariant__ is None

def test__make_seq_field_type_reuses_existing_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types

    class MockCheckedClass:
        _checked_types = (int,)

    type1 = _make_seq_field_type(MockCheckedClass, int, None)
    type2 = _make_seq_field_type(MockCheckedClass, int, None)
    assert type1 is type2
    assert (MockCheckedClass, int) in _seq_field_types


# LLM-generated content at query #61
#--------------------------

```python
def test_set_fields_empty_bases():
    dct = {}
    bases = []
    name = "test_name"
    set_fields(dct, bases, name)
    assert dct == {name: {}}

def test_set_fields_with_single_base():
    class Base:
        test_name = {"key1": "value1"}

    dct = {}
    bases = [Base]
    name = "test_name"
    set_fields(dct, bases, name)
    assert dct == {name: {"key1": "value1"}}

def test_set_fields_with_multiple_bases():
    class Base1:
        test_name = {"key1": "value1"}

    class Base2:
        test_name = {"key2": "value2"}

    dct = {}
    bases = [Base1, Base2]
    name = "test_name"
    set_fields(dct, bases, name)
    assert dct == {name: {"key1": "value1", "key2": "value2"}}

def test_set_fields_with_pfield():
    class _PField:
        pass

    dct = {"field1": _PField(), "field2": "value2"}
    bases = []
    name = "test_name"
    set_fields(dct, bases, name)
    assert dct == {name: {"field1": _PField()}, "field2": "value2"}


# LLM-generated content at query #62
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
        assert e.error_codes == ("ERROR1",)


# LLM-generated content at query #63
#--------------------------

```python
def test_isinstance_v_pfield():
    dct = {'key': _PField()}
    bases = []
    name = 'test_name'
    assert isinstance(dct['key'], _PField)


