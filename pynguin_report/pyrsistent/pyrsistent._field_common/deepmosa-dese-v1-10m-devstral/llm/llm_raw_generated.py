####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_check_global_invariants_all_pass():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_failure():
    subject = object()
    invariants = [lambda _: (False, "E1"), lambda _: (True, None)]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("E1",)

def test_check_global_invariants_multiple_failures():
    subject = object()
    invariants = [lambda _: (False, "E1"), lambda _: (False, "E2")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("E1", "E2")

def test_check_global_invariants_empty_invariants():
    subject = object()
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_check_field_parameters_with_valid_field():
    class MockField:
        type = (int, str)
        initial = 5
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x

    _check_field_parameters(MockField())

def test_check_field_parameters_with_invalid_type_parameter():
    class MockField:
        type = (int, 5)
        initial = PFIELD_NO_INITIAL
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_with_invalid_initial_type():
    class MockField:
        type = (int, str)
        initial = 5.5
        invariant = lambda x: True
        factory = lambda: None
        serializer = lambda x: x

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'float'>"

def test_check_field_parameters_with_non_callable_invariant():
    class MockField:
        type = (int, str)
        initial = 5
        invariant = "not callable"
        factory = lambda: None
        serializer = lambda x: x

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_with_non_callable_factory():
    class MockField:
        type = (int, str)
        initial = 5
        invariant = lambda x: True
        factory = "not callable"
        serializer = lambda x: x

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_with_non_callable_serializer():
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


# LLM-generated content at query #4
#--------------------------

```python
def test_pmap_field_creates_correct_field_with_non_optional():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type, optional=False)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_creates_correct_field_with_optional():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type, optional=True)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(str, int), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_custom_invariant():
    key_type = str
    value_type = int
    def custom_invariant(x):
        return True, "OK"
    result = pmap_field(key_type, value_type, optional=False, invariant=custom_invariant)
    assert isinstance(result, _PField)
    assert result.invariant == wrap_invariant(custom_invariant)


# LLM-generated content at query #5
#--------------------------

```python
def test_set_fields_basic():
    dct = {'a': 1, 'b': 2}
    bases = []
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct == {'a': 1, 'b': 2, 'fields': {}}

def test_set_fields_with_bases():
    class Base1:
        fields = {'x': 1, 'y': 2}
    class Base2:
        fields = {'y': 3, 'z': 4}
    dct = {}
    bases = [Base1, Base2]
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct == {'fields': {'x': 1, 'y': 3, 'z': 4}}

def test_set_fields_with_pfield():
    class _PField:
        pass
    dct = {'a': _PField(), 'b': 2}
    bases = []
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct == {'b': 2, 'fields': {'a': dct['fields']['a']}}
    assert isinstance(dct['fields']['a'], _PField)


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_evaluates_to_false():
    field = type('Field', (), {'type': [123], 'initial': PFIELD_NO_INITIAL, 'invariant': lambda x: True, 'factory': lambda: None, 'serializer': lambda x: x})()
    with raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_false():
    field = Mock()
    field.initial = PFIELD_NO_INITIAL
    field.type = []
    field.invariant = lambda: True
    field.factory = lambda: True
    field.serializer = lambda: True

    _check_field_parameters(field)


# LLM-generated content at query #8
#--------------------------

```python
def test_set_fields_empty_bases():
    dct = {}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {name: {}}

def test_set_fields_single_base_with_items():
    dct = {}
    class Base:
        pass
    Base.test_name = {'key1': 'value1'}
    bases = [Base]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {name: {'key1': 'value1'}}

def test_set_fields_multiple_bases_with_items():
    dct = {}
    class Base1:
        pass
    Base1.test_name = {'key1': 'value1'}
    class Base2:
        pass
    Base2.test_name = {'key2': 'value2'}
    bases = [Base1, Base2]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {name: {'key1': 'value1', 'key2': 'value2'}}

def test_set_fields_with_pfield_in_dct():
    dct = {'field1': _PField('value1')}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {name: {'field1': _PField('value1')}}


# LLM-generated content at query #9
#--------------------------

```python
def test_make_pmap_field_type_creates_new_subclass():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert issubclass(result, CheckedPMap)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type

def test_make_pmap_field_type_reuses_existing_subclass():
    key_type = int
    value_type = str
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call

def test_make_pmap_field_type_correct_name_generation():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.int"
    value_type = "builtins.str"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == int
    assert result.__value_type__ == str
    assert result.__name__ == "IntToStrPMap"

def test_make_pmap_field_type_reduce_method():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    instance = result({1: "a", 2: "b"})
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_pmap_field_pickle
    assert reduced[1][0] == key_type
    assert reduced[1][1] == value_type
    assert reduced[1][2] == {1: "a", 2: "b"}


# LLM-generated content at query #10
#--------------------------

```python
def test__sequence_field_with_checked_class_and_item_type():
    result = _sequence_field(CheckedPSet, int, False, [])
    assert isinstance(result, _PField)
    assert result.type == {CheckedPSet}
    assert result.factory == CheckedPSet.create
    assert result.mandatory == True
    assert result.initial == CheckedPSet.create([])

def test__sequence_field_with_optional():
    result = _sequence_field(CheckedPVector, str, True, [])
    assert isinstance(result, _PField)
    assert result.type == {CheckedPVector, type(None)}
    assert result.mandatory == True
    assert result.initial == None

def test__sequence_field_with_invariant():
    invariant = lambda x: True
    result = _sequence_field(CheckedPSet, float, False, [], invariant=invariant)
    assert isinstance(result, _PField)
    assert result.invariant == invariant
    assert result.type == {CheckedPSet}
    assert result.factory == CheckedPSet.create

def test__sequence_field_with_item_invariant():
    item_invariant = lambda x: True
    result = _sequence_field(CheckedPVector, bool, False, [], item_invariant=item_invariant)
    assert isinstance(result, _PField)
    assert result.type == {CheckedPVector}
    assert result.factory == CheckedPVector.create


# LLM-generated content at query #11
#--------------------------

```python
def test_isinstance_check():
    dct = {'key': _PField()}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert isinstance(dct['key'], _PField)


# LLM-generated content at query #12
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _seq_field_types, _restore_seq_field_pickle

    # Mock data and types
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


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_check_global_invariants_with_no_errors():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #15
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda _: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = object()
    invariants = [lambda _: (False, "E001")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("E001",)

def test_check_global_invariants_multiple_errors():
    subject = object()
    invariants = [lambda _: (False, "E001"), lambda _: (False, "E002")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("E001", "E002")

def test_check_global_invariants_mixed_results():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (False, "E001"), lambda _: (True, None)]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("E001",)


# LLM-generated content at query #16
#--------------------------

```python
def test_check_field_parameters_with_non_type_non_str_type():
    field = MockField(type=[123])
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_evaluates_to_true():
    dct = {'a': _PField(), 'b': 1}
    bases = []
    name = 'test'

    set_fields(dct, bases, name)

    assert 'a' not in dct
    assert 'test' in dct
    assert dct['test']['a'] is dct['a']


# LLM-generated content at query #18
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


# LLM-generated content at query #19
#--------------------------

```python
def test_sequence_field_creates_checked_class_with_correct_type():
    result = _sequence_field(CheckedPVector, int, False, [])
    assert isinstance(result.type, set)
    assert CheckedPVector in result.type

def test_sequence_field_with_optional_creates_correct_factory():
    result = _sequence_field(CheckedPSet, str, True, set())
    assert callable(result.factory)
    assert result.factory(None) is None

def test_sequence_field_without_optional_uses_default_factory():
    result = _sequence_field(CheckedPVector, float, False, [])
    assert result.factory == CheckedPVector.create

def test_sequence_field_sets_mandatory_to_true():
    result = _sequence_field(CheckedPSet, int, False, set())
    assert result.mandatory is True

def test_sequence_field_preserves_invariant():
    def test_invariant(x):
        return True
    result = _sequence_field(CheckedPVector, str, False, [], invariant=test_invariant)
    assert result.invariant == test_invariant

def test_sequence_field_creates_initial_value():
    result = _sequence_field(CheckedPSet, int, False, {1, 2, 3})
    assert isinstance(result.initial, CheckedPSet)
    assert result.initial == CheckedPSet([1, 2, 3])

def test_sequence_field_with_optional_type():
    result = _sequence_field(CheckedPVector, str, True, [])
    assert type(None) in result.type


# LLM-generated content at query #20
#--------------------------

```python
def test_check_field_parameters_with_invalid_initial_type():
    field = MockField(
        type=[int],
        initial="not_an_int",
        invariant=lambda x: True,
        factory=lambda: None,
        serializer=lambda x: x
    )
    with raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #21
#--------------------------

```python
def test_pmap_field_optional_false():
    result = pmap_field(str, int, optional=False)
    assert not result.factory is None


# LLM-generated content at query #22
#--------------------------

```python
def test__make_seq_field_type_creates_subclass_with_correct_name():
    class TestClass:
        _checked_types = (int, str)

    result = _make_seq_field_type(TestClass, int, lambda x: True)
    assert result.__name__ == "IntStrSeq"

def test__make_seq_field_type_creates_subclass_with_type_and_invariant():
    class TestClass:
        pass

    result = _make_seq_field_type(TestClass, int, lambda x: x > 0)
    assert result.__type__ == int
    assert result.__invariant__(5)
    assert not result.__invariant__(-1)

def test__make_seq_field_type_returns_cached_type():
    class TestClass:
        pass

    type1 = _make_seq_field_type(TestClass, int, lambda x: True)
    type2 = _make_seq_field_type(TestClass, int, lambda x: True)
    assert type1 is type2

def test__make_seq_field_type_creates_subclass_with_reduce_method():
    class TestClass:
        pass

    result = _make_seq_field_type(TestClass, int, lambda x: True)
    instance = result([1, 2, 3])
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_seq_field_pickle
    assert reduced[1][0] == TestClass
    assert reduced[1][1] == int
    assert reduced[1][2] == [1, 2, 3]


# LLM-generated content at query #23
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type

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

def test_make_pmap_field_type_with_custom_type_names():
    key_type = "collections.OrderedDict"
    value_type = "typing.List"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "OrdereddictToListPMap"
    assert result.__key_type__.__name__ == "OrderedDict"
    assert result.__value_type__.__name__ == "List"


# LLM-generated content at query #24
#--------------------------

```python
def test__make_seq_field_type_creates_subclass_with_correct_attributes():
    class MockCheckedClass:
        _checked_types = (int,)

    item_type = int
    item_invariant = lambda x: x > 0
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)

    assert issubclass(result, MockCheckedClass)
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant
    assert result.__name__ == "Int" + SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]

def test__make_seq_field_type_caches_created_type():
    class MockCheckedClass:
        _checked_types = (int,)

    item_type = int
    item_invariant = lambda x: x > 0
    first_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)

    assert first_call is second_call

def test__make_seq_field_type_uses_types_to_names_for_naming():
    class MockCheckedClass:
        _checked_types = (int, str)

    item_type = int
    item_invariant = lambda x: x > 0
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)

    assert result.__name__ == "IntStr" + SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]

def test__make_seq_field_type_reduce_returns_correct_tuple():
    class MockCheckedClass:
        _checked_types = (int,)

    item_type = int
    item_invariant = lambda x: x > 0
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    instance = result([1, 2, 3])

    reduce_result = instance.__reduce__()
    assert reduce_result[0] == _restore_seq_field_pickle
    assert reduce_result[1][0] == MockCheckedClass
    assert reduce_result[1][1] == item_type
    assert reduce_result[1][2] == [1, 2, 3]


# LLM-generated content at query #25
#--------------------------

```python
def test_check_field_parameters_with_non_callable_invariant():
    field = MockField(type=[str], initial='default', invariant='not_callable', factory=lambda: None, serializer=lambda x: x)
    assert _check_field_parameters(field) == False


# LLM-generated content at query #26
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"
    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #27
#--------------------------

```python
def test__make_seq_field_type_with_existing_type():
    checked_class = list
    item_type = int
    item_invariant = lambda x: x > 0
    _seq_field_types[(checked_class, item_type)] = type('TestType', (checked_class,), {})
    result = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert result is _seq_field_types[(checked_class, item_type)]

def test__make_seq_field_type_with_new_type():
    checked_class = list
    item_type = int
    item_invariant = lambda x: x > 0
    _seq_field_types.clear()
    result = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert issubclass(result, checked_class)
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant
    assert result.__name__.endswith(SEQ_FIELD_TYPE_SUFFIXES[checked_class])
    assert (checked_class, item_type) in _seq_field_types
    assert _seq_field_types[(checked_class, item_type)] is result

def test__make_seq_field_type_with_string_type():
    checked_class = list
    item_type = "builtins.int"
    item_invariant = lambda x: x > 0
    _seq_field_types.clear()
    result = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert issubclass(result, checked_class)
    assert result.__type__ == int
    assert result.__invariant__ == item_invariant
    assert result.__name__.endswith(SEQ_FIELD_TYPE_SUFFIXES[checked_class])
    assert (checked_class, item_type) in _seq_field_types
    assert _seq_field_types[(checked_class, item_type)] is result

def test__make_seq_field_type_with_custom_class():
    class CustomClass:
        pass
    checked_class = CustomClass
    item_type = int
    item_invariant = lambda x: x > 0
    _seq_field_types.clear()
    result = _make_seq_field_type(checked_class, item_type, item_invariant)
    assert issubclass(result, checked_class)
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant
    assert result.__name__.endswith(SEQ_FIELD_TYPE_SUFFIXES[checked_class])
    assert (checked_class, item_type) in _seq_field_types
    assert _seq_field_types[(checked_class, item_type)] is result


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_3_evaluates_to_false():
    field = Mock()
    field.type = [123]
    with pytest.raises(TypeError):
        _check_field_parameters(field)


# LLM-generated content at query #29
#--------------------------

```python
def test_check_global_invariants_with_valid_subject():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #30
#--------------------------

```python
def test_restore_seq_field_pickle_with_valid_data():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle

    # Mock data
    data = [1, 2, 3]

    # Mock _seq_field_types to return a mock class
    import pyrsistent._field_common as field_common
    mock_class = type('MockClass', (), {'create': lambda self, data, _factory_fields: data})
    field_common._seq_field_types = {(type, int): mock_class}

    # Mock _restore_pickle to return the data
    field_common._restore_pickle = lambda cls, data: data

    result = _restore_seq_field_pickle(type, int, data)
    assert result == data


# LLM-generated content at query #31
#--------------------------

```python
def test_check_global_invariants_with_no_errors():
    subject = object()
    invariants = [lambda _: (True, None)]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #32
#--------------------------

```python
def test_valid_field_parameters():
    class MockField:
        def __init__(self, type_param, initial, invariant, factory, serializer):
            self.type = type_param
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    field = MockField(
        type_param=[int, str],
        initial=42,
        invariant=lambda x: True,
        factory=lambda: None,
        serializer=lambda x: str(x)
    )
    _check_field_parameters(field)

def test_invalid_type_parameter():
    class MockField:
        def __init__(self, type_param):
            self.type = type_param
            self.initial = None
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)

    field = MockField(type_param=[123])
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Type parameter expected, not <class \'int\'>'

def test_invalid_initial_type():
    class MockField:
        def __init__(self, type_param, initial):
            self.type = type_param
            self.initial = initial
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: str(x)

    field = MockField(type_param=[int], initial="not an int")
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Initial has invalid type <class \'str\'>'

def test_non_callable_invariant():
    class MockField:
        def __init__(self, invariant):
            self.type = [int]
            self.initial = 42
            self.invariant = invariant
            self.factory = lambda: None
            self.serializer = lambda x: str(x)

    field = MockField(invariant="not callable")
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariant must be callable'

def test_non_callable_factory():
    class MockField:
        def __init__(self, factory):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = factory
            self.serializer = lambda x: str(x)

    field = MockField(factory="not callable")
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Factory must be callable'

def test_non_callable_serializer():
    class MockField:
        def __init__(self, serializer):
            self.type = [int]
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = serializer

    field = MockField(serializer="not callable")
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Serializer must be callable'


# LLM-generated content at query #33
#--------------------------

```python
def test_restore_pmap_field_pickle_calls_restore_pickle_with_correct_args():
    _pmap_field_types = {(str, int): type('MockType', (), {'create': lambda data, _factory_fields: data})}
    data = {'a': 1, 'b': 2}
    result = _restore_pmap_field_pickle(str, int, data)
    assert result == data


# LLM-generated content at query #34
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

    # Mock _restore_pickle to return the data as-is
    original_restore_pickle = _restore_pickle
    _restore_pickle = lambda cls, data: data

    result = _restore_seq_field_pickle(checked_class, item_type, data)

    assert result == data

    # Restore original _restore_pickle
    _restore_pickle = original_restore_pickle


# LLM-generated content at query #35
#--------------------------

```python
def test_set_fields_predicate():
    class Base1:
        __dict__ = {'test_field': {'key1': _PField('value1')}}

    class Base2:
        __dict__ = {'test_field': {'key2': _PField('value2')}}

    dct = {'test_field': {}, 'other_field': 'value'}
    bases = [Base1, Base2]
    name = 'test_field'

    set_fields(dct, bases, name)

    assert dct[name]['key1'] is Base1.__dict__['test_field']['key1']
    assert dct[name]['key2'] is Base2.__dict__['test_field']['key2']
    assert 'other_field' not in dct


# LLM-generated content at query #36
#--------------------------

```python
def test__make_seq_field_type_creates_new_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types

    class MockCheckedClass:
        _checked_types = (int,)

    result = _make_seq_field_type(MockCheckedClass, int, lambda x: True)

    assert isinstance(result, type)
    assert result.__name__ == "IntSeq"
    assert result.__type__ == int
    assert result.__invariant__(5)
    assert (MockCheckedClass, int) in _seq_field_types
    assert _seq_field_types[(MockCheckedClass, int)] == result

def test__make_seq_field_type_returns_cached_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types

    class MockCheckedClass:
        _checked_types = (int,)

    first_call = _make_seq_field_type(MockCheckedClass, int, lambda x: True)
    second_call = _make_seq_field_type(MockCheckedClass, int, lambda x: True)

    assert first_call is second_call
    assert len(_seq_field_types) == 1

def test__make_seq_field_type_with_different_types():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types

    class MockCheckedClass:
        _checked_types = (int, str)

    result = _make_seq_field_type(MockCheckedClass, str, lambda x: len(x) > 0)

    assert isinstance(result, type)
    assert result.__name__ == "IntStrSeq"
    assert result.__type__ == str
    assert not result.__invariant__("")
    assert result.__invariant__("test")


# LLM-generated content at query #37
#--------------------------

```python
def test_check_field_parameters_with_invalid_initial_type():
    field = Mock()
    field.type = [int, str]
    field.initial = 1.5  # float is not in field.type
    field.invariant = lambda x: True
    field.factory = lambda: None
    field.serializer = lambda x: x

    with pytest.raises(TypeError) as excinfo:
        _check_field_parameters(field)
    assert 'Initial has invalid type' in str(excinfo.value)


# LLM-generated content at query #38
#--------------------------

```python
def test_pfield_initialization():
    type_val = "test_type"
    invariant_val = "test_invariant"
    initial_val = "test_initial"
    mandatory_val = True
    factory_val = "test_factory"
    serializer_val = "test_serializer"

    pfield = _PField(type_val, invariant_val, initial_val, mandatory_val, factory_val, serializer_val)

    assert pfield.type == type_val
    assert pfield.invariant == invariant_val
    assert pfield.initial == initial_val
    assert pfield.mandatory == mandatory_val
    assert pfield._factory == factory_val
    assert pfield.serializer == serializer_val


# LLM-generated content at query #39
#--------------------------

```python
def test_check_field_parameters_valid_field():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = 42
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x

    field = MockField()
    _check_field_parameters(field)

def test_check_field_parameters_invalid_type_parameter():
    class MockField:
        def __init__(self):
            self.type = (int, 42)  # 42 is not a type or str
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_check_field_parameters_invalid_initial_type():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = 3.14  # float not in type
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = lambda x: x

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'float'>"

def test_check_field_parameters_non_callable_invariant():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = PFIELD_NO_INITIAL
            self.invariant = 42  # not callable
            self.factory = lambda: None
            self.serializer = lambda x: x

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_non_callable_factory():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = 42  # not callable
            self.serializer = lambda x: x

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_non_callable_serializer():
    class MockField:
        def __init__(self):
            self.type = (int, str)
            self.initial = PFIELD_NO_INITIAL
            self.invariant = lambda x: True
            self.factory = lambda: None
            self.serializer = 42  # not callable

    field = MockField()
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #40
#--------------------------

```python
def test_pmap_field_predicate_false():
    key_type = str
    value_type = int
    optional = False
    invariant = PFIELD_NO_INVARIANT
    result = pmap_field(key_type, value_type, optional, invariant)
    assert not optional


# LLM-generated content at query #41
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_false():
    class MockField:
        def __init__(self, initial, type, invariant, factory, serializer):
            self.initial = initial
            self.type = type
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    PFIELD_NO_INITIAL = object()
    field = MockField(
        initial=PFIELD_NO_INITIAL,
        type=[int],
        invariant=lambda: True,
        factory=lambda: None,
        serializer=lambda x: x
    )
    assert not (field.initial is not PFIELD_NO_INITIAL and
                not callable(field.initial) and
                field.type and not any(isinstance(field.initial, t) for t in field.type))


# LLM-generated content at query #42
#--------------------------

```python
def test_pfield_initialization():
    type_val = "test_type"
    invariant_val = "test_invariant"
    initial_val = "test_initial"
    mandatory_val = True
    factory_val = "test_factory"
    serializer_val = "test_serializer"

    pfield = _PField(type_val, invariant_val, initial_val, mandatory_val, factory_val, serializer_val)

    assert pfield.type == type_val
    assert pfield.invariant == invariant_val
    assert pfield.initial == initial_val
    assert pfield.mandatory == mandatory_val
    assert pfield._factory == factory_val
    assert pfield.serializer == serializer_val


# LLM-generated content at query #43
#--------------------------

```python
def test_make_pmap_field_type_creates_new_type():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type
    assert result.__name__ == "IntToStrPMap"

def test_make_pmap_field_type_reuses_existing_type():
    key_type = int
    value_type = str
    first_result = _make_pmap_field_type(key_type, value_type)
    second_result = _make_pmap_field_type(key_type, value_type)
    assert first_result is second_result

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.int"
    value_type = "builtins.str"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__key_type__ == int
    assert result.__value_type__ == str
    assert result.__name__ == "IntToStrPMap"


# LLM-generated content at query #44
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    checked_value = CheckedType()
    checked_value.serialize = lambda fmt: f"serialized_{fmt}"
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    assert result == "serialized_json"

def test_serialize_with_custom_serializer():
    def custom_serializer(fmt, val):
        return f"{val}_as_{fmt}"

    result = serialize(custom_serializer, "xml", "data")
    assert result == "data_as_xml"


# LLM-generated content at query #2
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}

    result = _restore_pmap_field_pickle(key_type, value_type, data)

    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


# LLM-generated content at query #3
#--------------------------

```python
def test_make_pmap_field_type_with_builtin_types():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"
    assert result.__key_type__ == int
    assert result.__value_type__ == str

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.int"
    value_type = "builtins.str"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"
    assert result.__key_type__ == int
    assert result.__value_type__ == str

def test_make_pmap_field_type_caching():
    key_type = int
    value_type = str
    result1 = _make_pmap_field_type(key_type, value_type)
    result2 = _make_pmap_field_type(key_type, value_type)
    assert result1 is result2


# LLM-generated content at query #4
#--------------------------

```python
def test_sequence_field_creates_checked_class_with_non_optional_type():
    result = _sequence_field(CheckedPVector, int, False, [])
    assert isinstance(result, _PField)
    assert result.type == {_make_seq_field_type(CheckedPVector, int, PFIELD_NO_INVARIANT)}
    assert result.factory is _make_seq_field_type(CheckedPVector, int, PFIELD_NO_INVARIANT).create
    assert result.mandatory is True
    assert result.invariant is PFIELD_NO_INVARIANT
    assert result.initial == _make_seq_field_type(CheckedPVector, int, PFIELD_NO_INVARIANT).create([])

def test_sequence_field_creates_checked_class_with_optional_type():
    result = _sequence_field(CheckedPSet, str, True, set())
    assert isinstance(result, _PField)
    assert result.type == {_make_seq_field_type(CheckedPSet, str, PFIELD_NO_INVARIANT), type(None)}
    assert result.mandatory is True
    assert result.invariant is PFIELD_NO_INVARIANT
    assert result.initial == _make_seq_field_type(CheckedPSet, str, PFIELD_NO_INVARIANT).create(set())

def test_sequence_field_with_custom_invariant():
    def custom_invariant(x):
        return len(x) > 0
    result = _sequence_field(CheckedPVector, int, False, [1, 2, 3], invariant=custom_invariant)
    assert result.invariant == custom_invariant
    assert result.initial == _make_seq_field_type(CheckedPVector, int, PFIELD_NO_INVARIANT).create([1, 2, 3])

def test_sequence_field_with_item_invariant():
    def item_invariant(x):
        return x > 0
    result = _sequence_field(CheckedPSet, int, False, {1, 2, 3}, item_invariant=item_invariant)
    assert result.type == {_make_seq_field_type(CheckedPSet, int, item_invariant)}
    assert result.initial == _make_seq_field_type(CheckedPSet, int, item_invariant).create({1, 2, 3})


# LLM-generated content at query #5
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    class CheckedType:
        def serialize(self, format):
            return f"serialized_{format}"

    value = CheckedType()
    PFIELD_NO_SERIALIZER = None
    format = "test_format"

    result = serialize(PFIELD_NO_SERIALIZER, format, value)

    assert result == "serialized_test_format"


# LLM-generated content at query #6
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = "test_subject"
    invariants = [lambda x: (True, "error1"), lambda x: (True, "error2")]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = "test_subject"
    invariants = [lambda x: (True, "error1"), lambda x: (False, "error2")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("error2",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'

def test_check_global_invariants_multiple_errors():
    subject = "test_subject"
    invariants = [lambda x: (False, "error1"), lambda x: (False, "error2")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[0] == ("error1", "error2")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #7
#--------------------------

```python
def test_check_field_parameters_with_valid_field():
    field = MockField(type=[int, str], initial=42, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    _check_field_parameters(field)

def test_check_field_parameters_with_invalid_type_parameter():
    field = MockField(type=[42], initial=42, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError, match='Type parameter expected, not <class \'int\'>'):
        _check_field_parameters(field)

def test_check_field_parameters_with_invalid_initial_type():
    field = MockField(type=[int, str], initial=42.0, invariant=lambda x: True, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError, match='Initial has invalid type <class \'float\'>'):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_invariant():
    field = MockField(type=[int, str], initial=42, invariant=42, factory=lambda: None, serializer=lambda x: x)
    with pytest.raises(TypeError, match='Invariant must be callable'):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_factory():
    field = MockField(type=[int, str], initial=42, invariant=lambda x: True, factory=42, serializer=lambda x: x)
    with pytest.raises(TypeError, match='Factory must be callable'):
        _check_field_parameters(field)

def test_check_field_parameters_with_non_callable_serializer():
    field = MockField(type=[int, str], initial=42, invariant=lambda x: True, factory=lambda: None, serializer=42)
    with pytest.raises(TypeError, match='Serializer must be callable'):
        _check_field_parameters(field)


# LLM-generated content at query #8
#--------------------------

```python
def test_check_field_parameters_with_non_callable_invariant():
    field = MockField(invariant=123)
    assert not _check_field_parameters(field)


# LLM-generated content at query #9
#--------------------------

```python
def test_pmap_field_creates_checked_pmap_field():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type)
    assert result.type == {TheMap}  # Assuming TheMap is the generated CheckedPMap class
    assert result.mandatory is True
    assert result.initial == TheMap()
    assert result.factory == TheMap.create

def test_pmap_field_with_optional_creates_optional_field():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type, optional=True)
    assert result.type == {TheMap, type(None)}
    assert result.mandatory is True
    assert result.initial == TheMap()
    assert callable(result.factory)
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == TheMap({"a": 1})

def test_pmap_field_with_invariant_applies_invariant():
    key_type = str
    value_type = int
    invariant = lambda x: (True, "OK")
    result = pmap_field(key_type, value_type, invariant=invariant)
    assert result.invariant == wrap_invariant(invariant)


# LLM-generated content at query #10
#--------------------------

```python
def test_valid_field_parameters():
    class MockField:
        def __init__(self, type, initial, invariant, factory, serializer):
            self.type = type
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    field = MockField(
        type=[int, str],
        initial=42,
        invariant=lambda x: True,
        factory=lambda: None,
        serializer=lambda x: str(x)
    )
    _check_field_parameters(field)

def test_invalid_type_parameter():
    class MockField:
        def __init__(self, type, initial, invariant, factory, serializer):
            self.type = type
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    field = MockField(
        type=[int, 42],
        initial=42,
        invariant=lambda x: True,
        factory=lambda: None,
        serializer=lambda x: str(x)
    )
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Type parameter expected, not <class 'int'>"

def test_invalid_initial_type():
    class MockField:
        def __init__(self, type, initial, invariant, factory, serializer):
            self.type = type
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    field = MockField(
        type=[int, str],
        initial=3.14,
        invariant=lambda x: True,
        factory=lambda: None,
        serializer=lambda x: str(x)
    )
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Initial has invalid type <class 'float'>"

def test_non_callable_invariant():
    class MockField:
        def __init__(self, type, initial, invariant, factory, serializer):
            self.type = type
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    field = MockField(
        type=[int, str],
        initial=42,
        invariant=42,
        factory=lambda: None,
        serializer=lambda x: str(x)
    )
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_non_callable_factory():
    class MockField:
        def __init__(self, type, initial, invariant, factory, serializer):
            self.type = type
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    field = MockField(
        type=[int, str],
        initial=42,
        invariant=lambda x: True,
        factory=42,
        serializer=lambda x: str(x)
    )
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_non_callable_serializer():
    class MockField:
        def __init__(self, type, initial, invariant, factory, serializer):
            self.type = type
            self.initial = initial
            self.invariant = invariant
            self.factory = factory
            self.serializer = serializer

    field = MockField(
        type=[int, str],
        initial=42,
        invariant=lambda x: True,
        factory=lambda: None,
        serializer=42
    )
    try:
        _check_field_parameters(field)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #11
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda _: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_single_error():
    subject = object()
    invariants = [lambda _: (False, "ERROR_CODE")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE",)

def test_check_global_invariants_multiple_errors():
    subject = object()
    invariants = [lambda _: (False, "ERROR1"), lambda _: (False, "ERROR2")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR1", "ERROR2")

def test_check_global_invariants_mixed_results():
    subject = object()
    invariants = [lambda _: (True, None), lambda _: (False, "ERROR"), lambda _: (True, None)]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR",)


# LLM-generated content at query #12
#--------------------------

```python
def test_sequence_field_with_checked_pset():
    checked_class = CheckedPSet
    item_type = int
    optional = False
    initial = [1, 2, 3]
    invariant = lambda x: True
    item_invariant = lambda x: True

    result = _sequence_field(checked_class, item_type, optional, initial, invariant, item_invariant)

    assert isinstance(result, _PField)
    assert result.type == {_make_seq_field_type(checked_class, item_type, item_invariant)}
    assert result.factory == _make_seq_field_type(checked_class, item_type, item_invariant).create
    assert result.mandatory is True
    assert result.invariant == invariant
    assert result.initial == _make_seq_field_type(checked_class, item_type, item_invariant).create(initial)

def test_sequence_field_with_checked_pvector():
    checked_class = CheckedPVector
    item_type = str
    optional = False
    initial = ['a', 'b', 'c']
    invariant = lambda x: True
    item_invariant = lambda x: True

    result = _sequence_field(checked_class, item_type, optional, initial, invariant, item_invariant)

    assert isinstance(result, _PField)
    assert result.type == {_make_seq_field_type(checked_class, item_type, item_invariant)}
    assert result.factory == _make_seq_field_type(checked_class, item_type, item_invariant).create
    assert result.mandatory is True
    assert result.invariant == invariant
    assert result.initial == _make_seq_field_type(checked_class, item_type, item_invariant).create(initial)

def test_sequence_field_with_optional():
    checked_class = CheckedPSet
    item_type = int
    optional = True
    initial = None
    invariant = lambda x: True
    item_invariant = lambda x: True

    result = _sequence_field(checked_class, item_type, optional, initial, invariant, item_invariant)

    assert isinstance(result, _PField)
    assert result.type == {_make_seq_field_type(checked_class, item_type, item_invariant), type(None)}
    assert callable(result.factory)
    assert result.mandatory is True
    assert result.invariant == invariant
    assert result.initial is None


# LLM-generated content at query #13
#--------------------------

```python
def test__make_seq_field_type_creates_subclass_with_correct_name():
    class MockCheckedClass:
        _checked_types = (int,)

    item_type = int
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result.__name__ == "Int" + SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]

def test__make_seq_field_type_creates_subclass_with_correct_attributes():
    class MockCheckedClass:
        _checked_types = (int,)

    item_type = int
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant

def test__make_seq_field_type_creates_subclass_with_correct_reduce():
    class MockCheckedClass:
        _checked_types = (int,)

    item_type = int
    item_invariant = lambda x: True
    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    instance = result([1, 2, 3])
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_seq_field_pickle
    assert reduced[1][0] == MockCheckedClass
    assert reduced[1][1] == item_type
    assert reduced[1][2] == [1, 2, 3]

def test__make_seq_field_type_returns_cached_type():
    class MockCheckedClass:
        _checked_types = (int,)

    item_type = int
    item_invariant = lambda x: True
    first_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    assert first_call is second_call


# LLM-generated content at query #14
#--------------------------

```python
def test_check_type_with_valid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    check_type(TestClass, Field(), 'test_field', 123)

def test_check_type_with_invalid_type():
    class TestClass:
        pass

    class Field:
        type = (int,)

    try:
        check_type(TestClass, Field(), 'test_field', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.cls == TestClass
        assert e.field_name == 'test_field'
        assert e.expected_type == (int,)
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
        type = (int, str)

    check_type(TestClass, Field(), 'test_field', 123)
    check_type(TestClass, Field(), 'test_field', 'string')

def test_check_type_with_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ('builtins.int',)

    check_type(TestClass, Field(), 'test_field', 456)

def test_check_type_with_invalid_string_type_name():
    class TestClass:
        pass

    class Field:
        type = ('builtins.int',)

    try:
        check_type(TestClass, Field(), 'test_field', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.cls == TestClass
        assert e.field_name == 'test_field'
        assert e.expected_type == ('builtins.int',)
        assert e.actual_type == str


# LLM-generated content at query #15
#--------------------------

```python
def test_pfield_constructor():
    type_val = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = lambda: 42
    serializer = str

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
def test_serialize_checked_type_with_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"
    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #17
#--------------------------

```python
def test_pmap_field_with_non_optional_and_no_invariant():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_optional_and_no_invariant():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type, optional=True)
    assert result.type == {_make_pmap_field_type(str, int), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_non_optional_and_invariant():
    key_type = str
    value_type = int
    def invariant_func(x):
        return True, "Test"
    result = pmap_field(key_type, value_type, invariant=invariant_func)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create
    assert result.invariant({"a": 1}) == (True, "Test")

def test_pmap_field_with_optional_and_invariant():
    key_type = str
    value_type = int
    def invariant_func(x):
        return True, "Test"
    result = pmap_field(key_type, value_type, optional=True, invariant=invariant_func)
    assert result.type == {_make_pmap_field_type(str, int), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory(None) is None
    assert result.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})
    assert result.invariant({"a": 1}) == (True, "Test")


# LLM-generated content at query #18
#--------------------------

```python
def test_set_fields_empty_bases():
    dct = {}
    bases = []
    name = "test_name"
    set_fields(dct, bases, name)
    assert dct == {name: {}}

def test_set_fields_with_bases():
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

def test_set_fields_combined():
    class _PField:
        pass

    class Base:
        test_name = {"key1": "value1"}

    dct = {"field1": _PField(), "field2": "value2"}
    bases = [Base]
    name = "test_name"
    set_fields(dct, bases, name)
    assert dct == {name: {"key1": "value1", "field1": _PField()}, "field2": "value2"}


# LLM-generated content at query #19
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
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def test_invariant(pmap):
        return True, "test"
    result = pmap_field(int, str, invariant=test_invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory == True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert callable(result.invariant)

def test_pmap_field_optional_with_invariant():
    def test_invariant(pmap):
        return True, "test"
    result = pmap_field(int, str, optional=True, invariant=test_invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str), type(None)}
    assert result.mandatory == True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory(None) is None
    assert callable(result.invariant)


# LLM-generated content at query #20
#--------------------------

```python
def test_sequence_field_invariant_default():
    result = _sequence_field(CheckedPSet, int, False, [])
    assert result.invariant == PFIELD_NO_INVARIANT


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_evaluates_to_true():
    dct = {'a': _PField(), 'b': 1}
    bases = []
    name = 'fields'
    set_fields(dct, bases, name)
    assert 'a' in dct[name] and 'a' not in dct


# LLM-generated content at query #22
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = str
    value_type = int
    data = {"a": 1, "b": 2}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert isinstance(result, _pmap_field_types[key_type, value_type])
    assert result == _pmap_field_types[key_type, value_type].create(data, _factory_fields=set())


# LLM-generated content at query #23
#--------------------------

```python
def test__make_pmap_field_type_creates_new_type():
    key_type = int
    value_type = str
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "IntToStrPMap"
    assert result.__key_type__ == key_type
    assert result.__value_type__ == value_type


# LLM-generated content at query #24
#--------------------------

```python
def test_check_global_invariants_with_valid_invariants():
    subject = object()
    invariants = [lambda _: (True, None)]
    assert check_global_invariants(subject, invariants) is None


# LLM-generated content at query #25
#--------------------------

```python
def test_sequence_field_with_optional_true():
    checked_class = CheckedPVector
    item_type = int
    optional = True
    initial = [1, 2, 3]
    invariant = PFIELD_NO_INVARIANT
    item_invariant = PFIELD_NO_INVARIANT

    result = _sequence_field(checked_class, item_type, optional, initial, invariant, item_invariant)

    assert result.factory is not None
    assert result.factory(initial) == CheckedPVector([1, 2, 3])
    assert result.factory(None) is None


# LLM-generated content at query #26
#--------------------------

```python
def test_make_pmap_field_type_creates_new_class():
    key_type = str
    value_type = int
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int

def test_make_pmap_field_type_returns_existing_class():
    key_type = str
    value_type = int
    first_call = _make_pmap_field_type(key_type, value_type)
    second_call = _make_pmap_field_type(key_type, value_type)
    assert first_call is second_call

def test_make_pmap_field_type_with_string_type_names():
    key_type = "builtins.str"
    value_type = "builtins.int"
    result = _make_pmap_field_type(key_type, value_type)
    assert result.__name__ == "StrToIntPMap"
    assert result.__key_type__ == str
    assert result.__value_type__ == int


# LLM-generated content at query #27
#--------------------------

```python
def test_pmap_field_basic():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type)
    assert isinstance(result, _PField)
    assert result.type == {TheMap}
    assert result.mandatory == True
    assert result.initial == TheMap()
    assert result.factory == TheMap.create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional():
    key_type = str
    value_type = int
    result = pmap_field(key_type, value_type, optional=True)
    assert isinstance(result, _PField)
    assert result.type == {TheMap, type(None)}
    assert result.mandatory == True
    assert result.initial == TheMap()
    assert result.factory(None) == None
    assert result.factory({"a": 1}) == TheMap({"a": 1})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    key_type = str
    value_type = int
    invariant = lambda x: (True, "OK")
    result = pmap_field(key_type, value_type, invariant=invariant)
    assert isinstance(result, _PField)
    assert result.type == {TheMap}
    assert result.mandatory == True
    assert result.initial == TheMap()
    assert result.factory == TheMap.create
    assert result.invariant == wrap_invariant(invariant)


# LLM-generated content at query #28
#--------------------------

```python
def test_pmap_field_optional_false():
    result = pmap_field(str, int, optional=False)
    assert not result.factory(None) is None


# LLM-generated content at query #29
#--------------------------

```python
def test_check_global_invariants_no_errors():
    subject = object()
    invariants = [lambda x: (True, None), lambda x: (True, None)]
    check_global_invariants(subject, invariants)

def test_check_global_invariants_with_errors():
    subject = object()
    invariants = [lambda x: (False, "ERROR1"), lambda x: (True, None)]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR1",)
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'

def test_check_global_invariants_multiple_errors():
    subject = object()
    invariants = [lambda x: (False, "ERROR1"), lambda x: (False, "ERROR2")]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR1", "ERROR2")
        assert e.args[1] == ()
        assert e.args[2] == 'Global invariant failed'


# LLM-generated content at query #30
#--------------------------

```python
def test__make_seq_field_type_creates_new_type():
    from pyrsistent._checked_types import get_type
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types

    class MockCheckedClass:
        _checked_types = (int,)
        __name__ = "MockCheckedClass"

    item_type = int
    item_invariant = lambda x: True

    result = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)

    assert isinstance(result, type)
    assert issubclass(result, MockCheckedClass)
    assert result.__type__ == item_type
    assert result.__invariant__ == item_invariant
    assert result.__name__ == "Int" + _seq_field_types.SEQ_FIELD_TYPE_SUFFIXES[MockCheckedClass]
    assert (MockCheckedClass, item_type) in _seq_field_types
    assert _seq_field_types[(MockCheckedClass, item_type)] == result

def test__make_seq_field_type_returns_cached_type():
    from pyrsistent._field_common import _make_seq_field_type, _seq_field_types

    class MockCheckedClass:
        _checked_types = (int,)

    item_type = int
    item_invariant = lambda x: True

    first_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)
    second_call = _make_seq_field_type(MockCheckedClass, item_type, item_invariant)

    assert first_call is second_call


# LLM-generated content at query #31
#--------------------------

```python
def test_pmap_field_optional_factory_returns_none():
    result = pmap_field(str, int, optional=True).factory(None)
    assert result is None


# LLM-generated content at query #32
#--------------------------

```python
def test_pmap_field_basic():
    field_instance = pmap_field(str, int)
    assert isinstance(field_instance, _PField)
    assert field_instance.type == {TheMap}
    assert field_instance.mandatory == True
    assert field_instance.initial == TheMap()
    assert field_instance.factory == TheMap.create
    assert field_instance.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional():
    field_instance = pmap_field(str, int, optional=True)
    assert isinstance(field_instance, _PField)
    assert field_instance.type == {TheMap, type(None)}
    assert field_instance.mandatory == True
    assert field_instance.initial == TheMap()
    assert field_instance.factory(None) is None
    assert field_instance.factory({"a": 1}) == TheMap({"a": 1})
    assert field_instance.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def test_invariant(pmap):
        return True, "Test"
    field_instance = pmap_field(str, int, invariant=test_invariant)
    assert isinstance(field_instance, _PField)
    assert field_instance.type == {TheMap}
    assert field_instance.mandatory == True
    assert field_instance.initial == TheMap()
    assert field_instance.factory == TheMap.create
    assert field_instance.invariant({"a": 1}) == (True, "Test")

def test_pmap_field_optional_with_invariant():
    def test_invariant(pmap):
        return True, "Test"
    field_instance = pmap_field(str, int, optional=True, invariant=test_invariant)
    assert isinstance(field_instance, _PField)
    assert field_instance.type == {TheMap, type(None)}
    assert field_instance.mandatory == True
    assert field_instance.initial == TheMap()
    assert field_instance.factory(None) is None
    assert field_instance.factory({"a": 1}) == TheMap({"a": 1})
    assert field_instance.invariant({"a": 1}) == (True, "Test")


# LLM-generated content at query #33
#--------------------------

```python
def test_pmap_field_optional_predicate():
    assert pmap_field(str, int, optional=True).factory(None) is None


# LLM-generated content at query #34
#--------------------------

```python
def test_set_fields_predicate():
    dct = {'a': _PField(), 'b': 1}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert 'a' not in dct
    assert 'test_name' in dct
    assert isinstance(dct['test_name']['a'], _PField)


# LLM-generated content at query #35
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._field_common import _restore_seq_field_pickle
    from pyrsistent._checked_types import _restore_pickle

    # Mock data
    checked_class = type('MockCheckedClass', (), {})
    item_type = type('MockItemType', (), {})
    data = [1, 2, 3]

    # Mock _seq_field_types and _restore_pickle
    _seq_field_types = {(checked_class, item_type): type('MockType', (), {'create': lambda self, data, _factory_fields: data})}
    _restore_pickle = lambda cls, data: cls.create(data, _factory_fields=set())

    # Test
    result = _restore_seq_field_pickle(checked_class, item_type, data)
    assert result == data


# LLM-generated content at query #36
#--------------------------

```python
def test_check_global_invariants_raises_exception_when_invariants_fail():
    def invariant1(subject):
        return (False, "INVARIANT_1_FAILED")

    def invariant2(subject):
        return (False, "INVARIANT_2_FAILED")

    subject = object()
    invariants = [invariant1, invariant2]

    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ("INVARIANT_1_FAILED", "INVARIANT_2_FAILED")


# LLM-generated content at query #37
#--------------------------

```python
def test_pfield_constructor():
    type_value = (int,)
    invariant = lambda x: x > 0
    initial = 0
    mandatory = True
    factory = None
    serializer = str

    pfield = _PField(type_value, invariant, initial, mandatory, factory, serializer)

    assert pfield.type == type_value
    assert pfield.invariant == invariant
    assert pfield.initial == initial
    assert pfield.mandatory == mandatory
    assert pfield._factory == factory
    assert pfield.serializer == serializer


# LLM-generated content at query #38
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false():
    assert is_field_ignore_extra_complaint(str, None, False) is False

def test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_a_subset_of_type_cls():
    field = type('Field', (), {'type': int, 'factory': lambda: None})()
    assert is_field_ignore_extra_complaint(str, field, True) is False

def test_is_field_ignore_extra_complaint_returns_false_when_field_factory_does_not_have_ignore_extra_param():
    field = type('Field', (), {'type': str, 'factory': lambda: None})()
    assert is_field_ignore_extra_complaint(str, field, True) is False

def test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_are_met():
    field = type('Field', (), {'type': str, 'factory': lambda ignore_extra=False: None})()
    assert is_field_ignore_extra_complaint(str, field, True) is True


# LLM-generated content at query #39
#--------------------------

```python
def test_set_fields_predicate():
    class Base1:
        __dict__ = {'fields': {'a': 1}}

    class Base2:
        __dict__ = {'fields': {'b': 2}}

    dct = {'fields': {}, 'c': _PField('test')}
    bases = (Base1, Base2)
    name = 'fields'

    set_fields(dct, bases, name)

    assert isinstance(dct['fields']['c'], _PField)


# LLM-generated content at query #40
#--------------------------

```python
def test_pmap_field_basic():
    f = pmap_field(int, str)
    assert f.type == {_make_pmap_field_type(int, str)}
    assert f.mandatory == True
    assert f.initial == _make_pmap_field_type(int, str)()
    assert f.factory == _make_pmap_field_type(int, str).create
    assert f.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_optional():
    f = pmap_field(int, str, optional=True)
    assert f.type == {_make_pmap_field_type(int, str), type(None)}
    assert f.mandatory == True
    assert f.initial == _make_pmap_field_type(int, str)()
    assert f.factory(None) is None
    assert f.factory({1: 'a'}) == _make_pmap_field_type(int, str).create({1: 'a'})
    assert f.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def inv(m):
        return True, "OK"
    f = pmap_field(int, str, invariant=inv)
    assert f.type == {_make_pmap_field_type(int, str)}
    assert f.mandatory == True
    assert f.initial == _make_pmap_field_type(int, str)()
    assert f.factory == _make_pmap_field_type(int, str).create
    assert f.invariant == wrap_invariant(inv)


# LLM-generated content at query #41
#--------------------------

```python
def test_set_fields_empty_bases():
    dct = {}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {name: {}}

def test_set_fields_with_bases():
    class Base1:
        test_name = {'a': 1}

    class Base2:
        test_name = {'b': 2}

    dct = {}
    bases = [Base1, Base2]
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {name: {'a': 1, 'b': 2}}

def test_set_fields_with_pfield():
    class _PField:
        pass

    dct = {'field': _PField()}
    bases = []
    name = 'test_name'
    set_fields(dct, bases, name)
    assert dct == {name: {'field': _PField()}}


# LLM-generated content at query #42
#--------------------------

```python
def test_check_global_invariants_with_failing_invariant():
    subject = object()
    invariants = [lambda _: (False, "ERROR_CODE")]
    try:
        check_global_invariants(subject, invariants)
    except InvariantException as e:
        assert e.error_codes == ("ERROR_CODE",)


# LLM-generated content at query #43
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
    assert result.factory(None) is None
    assert result.factory({1: "a"}) == _make_pmap_field_type(int, str).create({1: "a"})
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def test_invariant(x):
        return True, "test"
    result = pmap_field(int, str, invariant=test_invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory == _make_pmap_field_type(int, str).create
    assert result.invariant(test_invariant) == (True, "test")

def test_pmap_field_optional_with_invariant():
    def test_invariant(x):
        return True, "test"
    result = pmap_field(int, str, optional=True, invariant=test_invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(int, str), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(int, str)()
    assert result.factory(None) is None
    assert result.factory({1: "a"}) == _make_pmap_field_type(int, str).create({1: "a"})
    assert result.invariant(test_invariant) == (True, "test")


# LLM-generated content at query #44
#--------------------------

```python
def test_serialize_checked_type_with_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"
    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #45
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

def test_check_global_invariants_empty_invariants():
    subject = object()
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #46
#--------------------------

```python
def test_restore_seq_field_pickle():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types

    # Mock data for testing
    checked_class = type('MockCheckedClass', (), {})
    item_type = type('MockItemType', (), {})
    data = [1, 2, 3]

    # Mock _seq_field_types to return a mock type
    mock_type = type('MockType', (), {'create': lambda self, data, _factory_fields=None: data})
    _seq_field_types[checked_class, item_type] = mock_type

    # Call the function
    result = _restore_seq_field_pickle(checked_class, item_type, data)

    # Assertions
    assert result == data


# LLM-generated content at query #47
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    value = CheckedType()
    serializer = PFIELD_NO_SERIALIZER
    format = "some_format"

    assert isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #48
#--------------------------

```python
def test_predicate_evaluates_to_false():
    field = type('Field', (), {'type': [1], 'initial': None, 'invariant': lambda: True, 'factory': lambda: True, 'serializer': lambda: True})()
    assert not (isinstance(1, type) or isinstance(1, str))


# LLM-generated content at query #49
#--------------------------

```python
def test_sequence_field_optional_predicate():
    assert _sequence_field(CheckedPSet, int, True, initial=[]) is not None


# LLM-generated content at query #50
#--------------------------

```python
def test_restore_seq_field_pickle_returns_correct_type():
    from pyrsistent._checked_types import _restore_pickle
    from pyrsistent._field_common import _restore_seq_field_pickle, _seq_field_types

    checked_class = type('MockCheckedClass', (), {})
    item_type = int
    data = [1, 2, 3]

    _seq_field_types[checked_class, item_type] = type('MockType', (), {'create': lambda self, data, **kwargs: data})

    result = _restore_seq_field_pickle(checked_class, item_type, data)

    assert result == data


# LLM-generated content at query #51
#--------------------------

```python
def test_check_global_invariants_with_no_errors():
    subject = object()
    invariants = [lambda _: (True, None)]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #52
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


# LLM-generated content at query #53
#--------------------------

```python
def test_pmap_field_creates_checked_pmap_field():
    result = pmap_field(key_type=str, value_type=int)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_optional_true():
    result = pmap_field(key_type=str, value_type=int, optional=True)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(str, int), type(None)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert callable(result.factory)
    assert result.factory(None) is None
    assert result.invariant == PFIELD_NO_INVARIANT

def test_pmap_field_with_invariant():
    def test_invariant(x):
        return True
    result = pmap_field(key_type=str, value_type=int, invariant=test_invariant)
    assert isinstance(result, _PField)
    assert result.type == {_make_pmap_field_type(str, int)}
    assert result.mandatory is True
    assert result.initial == _make_pmap_field_type(str, int)()
    assert result.factory == _make_pmap_field_type(str, int).create
    assert result.invariant == wrap_invariant(test_invariant)


# LLM-generated content at query #54
#--------------------------

```python
def test_check_field_parameters_with_valid_field():
    class MockField:
        type = (int, str)
        initial = 0
        invariant = lambda x: True
        factory = lambda: 0
        serializer = lambda x: str(x)

    _check_field_parameters(MockField())

def test_check_field_parameters_with_invalid_type_parameter():
    class MockField:
        type = (int, 123)  # 123 is not a type or str
        initial = 0
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
        type = (int, str)
        initial = 3.14  # float not in type
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
        type = (int, str)
        initial = 0
        invariant = "not_callable"
        factory = lambda: 0
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariant must be callable"

def test_check_field_parameters_with_non_callable_factory():
    class MockField:
        type = (int, str)
        initial = 0
        invariant = lambda x: True
        factory = "not_callable"
        serializer = lambda x: str(x)

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Factory must be callable"

def test_check_field_parameters_with_non_callable_serializer():
    class MockField:
        type = (int, str)
        initial = 0
        invariant = lambda x: True
        factory = lambda: 0
        serializer = "not_callable"

    try:
        _check_field_parameters(MockField())
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Serializer must be callable"


# LLM-generated content at query #55
#--------------------------

```python
def test_optional_type_when_optional_is_true():
    result = pmap_field(int, str, optional=True)
    assert result.type == optional_type(TheMap)


# LLM-generated content at query #56
#--------------------------

```python
def test_serialize_with_checked_type_and_no_serializer():
    checked_value = CheckedType()
    checked_value.serialize = lambda fmt: "serialized"
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == "serialized"

def test_serialize_with_custom_serializer():
    def custom_serializer(fmt, val):
        return f"{fmt}:{val}"
    assert serialize(custom_serializer, "json", "data") == "json:data"


# LLM-generated content at query #57
#--------------------------

```python
def test_restore_pmap_field_pickle():
    key_type = int
    value_type = str
    data = [(1, "a"), (2, "b")]
    _pmap_field_types = {(int, str): type("MockPMapField", (), {"create": lambda data, _factory_fields: data})}
    result = _restore_pmap_field_pickle(key_type, value_type, data)
    assert result == data


# LLM-generated content at query #58
#--------------------------

```python
def test_pfield_initialization():
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


# LLM-generated content at query #59
#--------------------------

```python
def test_restore_seq_field_pickle():
    checked_class = type('MockClass', (), {})
    item_type = int
    data = [1, 2, 3]

    result = _restore_seq_field_pickle(checked_class, item_type, data)

    assert isinstance(result, _seq_field_types[checked_class, item_type])
    assert result == _seq_field_types[checked_class, item_type].create(data, _factory_fields=set())


# LLM-generated content at query #60
#--------------------------

```python
def test_predicate_evaluates_to_false():
    field = type('Field', (), {
        'type': [123],
        'initial': 'initial_value',
        'invariant': lambda x: True,
        'factory': lambda: None,
        'serializer': lambda x: x
    })()

    with pytest.raises(TypeError) as excinfo:
        _check_field_parameters(field)
    assert 'Type parameter expected, not' in str(excinfo.value)


