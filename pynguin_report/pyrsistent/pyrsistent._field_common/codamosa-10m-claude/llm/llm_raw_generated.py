####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_set_fields():
    # Test 1: Basic field inheritance from a single base class
    class Base:
        fields = {'field1': 'value1', 'field2': 'value2'}
    
    dct = {}
    bases = (Base,)
    name = 'fields'
    
    set_fields(dct, bases, name)
    
    assert 'fields' in dct
    assert dct['fields'] == {'field1': 'value1', 'field2': 'value2'}
    
    # Test 2: Field inheritance from multiple base classes
    class Base1:
        fields = {'field1': 'value1'}
    
    class Base2:
        fields = {'field2': 'value2', 'field3': 'value3'}
    
    dct = {}
    bases = (Base1, Base2)
    name = 'fields'
    
    set_fields(dct, bases, name)
    
    assert 'fields' in dct
    assert dct['fields'] == {'field1': 'value1', 'field2': 'value2', 'field3': 'value3'}
    
    # Test 3: _PField instances are moved from dct to the fields dictionary
    pfield1 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                      mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    pfield2 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                      mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    dct = {'pf1': pfield1, 'pf2': pfield2, 'other': 'value'}
    bases = ()
    name = 'fields'
    
    set_fields(dct, bases, name)
    
    assert 'fields' in dct
    assert 'pf1' in dct['fields']
    assert 'pf2' in dct['fields']
    assert dct['fields']['pf1'] is pfield1
    assert dct['fields']['pf2'] is pfield2
    assert 'pf1' not in dct
    assert 'pf2' not in dct
    assert 'other' in dct
    
    # Test 4: Mixed inheritance and _PField instances
    class Base:
        fields = {'inherited_field': 'inherited_value'}
    
    pfield = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    dct = {'new_field': pfield}
    bases = (Base,)
    name = 'fields'
    
    set_fields(dct, bases, name)
    
    assert 'fields' in dct
    assert 'inherited_field' in dct['fields']
    assert 'new_field' in dct['fields']
    assert dct['fields']['new_field'] is pfield
    assert 'new_field' not in dct
    
    # Test 5: Base class with no fields attribute
    class BaseNoFields:
        pass
    
    dct = {}
    bases = (BaseNoFields,)
    name = 'fields'
    
    set_fields(dct, bases, name)
    
    assert 'fields' in dct
    assert dct['fields'] == {}
    
    # Test 6: Overriding inherited fields
    class Base:
        fields = {'field1': 'old_value'}
    
    pfield = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    dct = {'field1': pfield}
    bases = (Base,)
    name = 'fields'
    
    set_fields(dct, bases, name)
    
    assert 'fields' in dct
    assert dct['fields']['field1'] is pfield
    assert 'field1' not in dct


# LLM-generated content at query #2
#--------------------------

```python
def test_pmap_field():
    """Test pmap_field function for creating checked PMap fields."""
    
    # Test basic pmap_field creation with required types
    int_str_field = pmap_field(int, str)
    assert int_str_field.mandatory is True
    assert int_str_field.factory is not None
    
    # Test that the field has correct type
    assert int_str_field.type is not None
    
    # Test optional pmap_field
    optional_field = pmap_field(int, str, optional=True)
    assert optional_field.mandatory is True
    assert optional_field.type is not None
    
    # Test that factory works with optional=True and None value
    result = optional_field.factory(None)
    assert result is None
    
    # Test that factory works with optional=True and valid dict
    result = optional_field.factory({1: "a", 2: "b"})
    assert result is not None
    
    # Test that factory works with optional=False
    non_optional_field = pmap_field(int, str, optional=False)
    result = non_optional_field.factory({1: "x"})
    assert result is not None
    
    # Test with invariant parameter
    def custom_invariant(pmap):
        return (len(pmap) > 0, "PMap must not be empty")
    
    field_with_invariant = pmap_field(int, str, invariant=custom_invariant)
    assert field_with_invariant.invariant is not None
    
    # Test initial value is set to empty TheMap
    assert field_with_invariant.initial is not None
    
    # Test multiple pmap_field calls with same types reuse cached type
    field1 = pmap_field(int, str)
    field2 = pmap_field(int, str)
    # Both should use the same cached type from _pmap_field_types
    assert field1.factory.__self__.__class__ == field2.factory.__self__.__class__
    
    # Test with different key/value types
    str_int_field = pmap_field(str, int)
    assert str_int_field.type is not None
    
    # Test optional with custom invariant
    optional_with_inv = pmap_field(str, float, optional=True, invariant=custom_invariant)
    assert optional_with_inv.mandatory is True
    assert optional_with_inv.invariant is not None


# LLM-generated content at query #3
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return {"serialized": True, "format": format}
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == {"serialized": True, "format": "json"}
    
    # Test with CheckedType and custom serializer (should use custom serializer)
    def custom_serializer(format, value):
        return {"custom": True, "format": format}
    
    result = serialize(custom_serializer, "xml", mock_obj)
    assert result == {"custom": True, "format": "xml"}
    
    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", "test_value")
    assert result == "test_value"
    
    # Test with non-CheckedType and custom serializer
    def custom_serializer2(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer2, "csv", "data")
    assert result == "csv:data"
    
    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test with primitive types and custom serializer
    def primitive_serializer(format, value):
        return {"value": value, "type": type(value).__name__}
    
    result = serialize(primitive_serializer, "json", 42)
    assert result == {"value": 42, "type": "int"}
    
    result = serialize(primitive_serializer, "json", 3.14)
    assert result == {"value": 3.14, "type": "float"}
    
    result = serialize(primitive_serializer, "json", True)
    assert result == {"value": True, "type": "bool"}


# LLM-generated content at query #4
#--------------------------

```python
def test_pmap_field():
    """Test pmap_field function for creating checked PMap fields."""
    from pyrsistent import pmap
    
    # Test basic pmap_field creation
    map_field = pmap_field(str, int)
    assert map_field is not None
    assert map_field.mandatory is True
    assert map_field.factory is not None
    
    # Test that initial value is an empty CheckedPMap
    assert map_field.initial == {}
    
    # Test pmap_field with optional=True
    optional_map_field = pmap_field(str, int, optional=True)
    assert optional_map_field is not None
    assert optional_map_field.mandatory is True
    
    # Test factory with valid data
    test_data = {'a': 1, 'b': 2}
    result = map_field.factory(test_data)
    assert result['a'] == 1
    assert result['b'] == 2
    
    # Test optional factory with None
    optional_result = optional_map_field.factory(None)
    assert optional_result is None
    
    # Test optional factory with valid data
    optional_result = optional_map_field.factory(test_data)
    assert optional_result['a'] == 1
    assert optional_result['b'] == 2
    
    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) <= 5, "Map too large")
    
    constrained_field = pmap_field(str, int, invariant=custom_invariant)
    assert constrained_field.invariant is not None
    
    # Test that the type is set correctly
    assert map_field.type is not None
    
    # Test multiple calls with same types return cached type
    map_field2 = pmap_field(str, int)
    assert map_field.factory(test_data).__class__ == map_field2.factory(test_data).__class__
    
    # Test pmap_field with different types
    float_int_field = pmap_field(float, str)
    assert float_int_field is not None
    
    # Test that factory preserves data
    test_data_float = {1.5: 'a', 2.5: 'b'}
    result_float = float_int_field.factory(test_data_float)
    assert result_float[1.5] == 'a'
    assert result_float[2.5] == 'b'


# LLM-generated content at query #5
#--------------------------

```python
def test_serialize():
    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    value = 42
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == 42

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: val * 2
    result = serialize(custom_serializer, "json", 21)
    assert result == 42

    # Test with CheckedType and PFIELD_NO_SERIALIZER (should call serialize method)
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return "serialized_" + format
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_json"

    # Test with CheckedType and custom serializer (should use custom serializer)
    custom_serializer = lambda fmt, val: "custom_" + fmt
    result = serialize(custom_serializer, "xml", mock_obj)
    assert result == "custom_xml"

    # Test with different format strings
    custom_serializer = lambda fmt, val: fmt.upper()
    result = serialize(custom_serializer, "yaml", 100)
    assert result == "YAML"

    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None

    # Test serializer that returns different type
    custom_serializer = lambda fmt, val: str(val)
    result = serialize(custom_serializer, "json", 123)
    assert result == "123"
    assert isinstance(result, str)


# LLM-generated content at query #6
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation with required types
    field_obj = pmap_field(str, int)
    assert field_obj.mandatory is True
    assert field_obj.factory is not None
    assert callable(field_obj.factory)
    
    # Test that initial value is an empty CheckedPMap
    assert field_obj.initial is not None
    assert isinstance(field_obj.initial, CheckedPMap)
    
    # Test pmap_field with optional=True
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.mandatory is True
    assert callable(optional_field.factory)
    
    # Test factory with optional=False creates CheckedPMap
    factory = pmap_field(str, int, optional=False).factory
    result = factory({'key': 1})
    assert isinstance(result, CheckedPMap)
    
    # Test factory with optional=True handles None
    optional_factory = pmap_field(str, int, optional=True).factory
    assert optional_factory(None) is None
    result = optional_factory({'key': 1})
    assert isinstance(result, CheckedPMap)
    
    # Test with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")
    
    field_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_invariant.invariant is not None
    assert callable(field_with_invariant.invariant)
    
    # Test type is set correctly
    field_obj = pmap_field(str, int)
    assert len(field_obj.type) > 0
    
    # Test optional field type includes optional wrapper
    optional_field = pmap_field(str, int, optional=True)
    assert len(optional_field.type) > 0
    
    # Test factory creates correct type with data
    factory = pmap_field(int, str).factory
    result = factory({1: 'a', 2: 'b'})
    assert dict(result) == {1: 'a', 2: 'b'}
    
    # Test different key and value types
    field1 = pmap_field(int, str)
    field2 = pmap_field(str, float)
    assert field1 is not field2
    
    # Test caching of types
    field_a = pmap_field(int, str)
    field_b = pmap_field(int, str)
    # Both should use the same underlying type from cache
    assert field_a.factory is not None
    assert field_b.factory is not None


# LLM-generated content at query #7
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}"
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_json"
    
    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test with custom serializer
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "xml:test_value"
    
    # Test with CheckedType and custom serializer
    result = serialize(custom_serializer, "json", mock_obj)
    assert result == f"json:{mock_obj}"
    
    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test with different formats
    result = serialize(PFIELD_NO_SERIALIZER, "csv", 100)
    assert result == 100
    
    # Test with complex value and custom serializer
    def complex_serializer(format, value):
        return {"format": format, "value": value, "type": type(value).__name__}
    
    result = serialize(complex_serializer, "yaml", [1, 2, 3])
    assert result == {"format": "yaml", "value": [1, 2, 3], "type": "list"}


# LLM-generated content at query #8
#--------------------------

```python
def test_serialize():
    # Test with non-CheckedType value and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42

    # Test with non-CheckedType value and custom serializer
    def custom_serializer(fmt, val):
        return val * 2
    
    result = serialize(custom_serializer, "json", 21)
    assert result == 42

    # Test with CheckedType value and PFIELD_NO_SERIALIZER
    class TestRecord(CheckedType):
        def serialize(self, format):
            return "serialized_" + format
    
    test_obj = TestRecord()
    result = serialize(PFIELD_NO_SERIALIZER, "json", test_obj)
    assert result == "serialized_json"

    # Test with CheckedType value and custom serializer (should use custom serializer)
    def custom_serializer2(fmt, val):
        return "custom_" + fmt
    
    result = serialize(custom_serializer2, "xml", test_obj)
    assert result == "custom_xml"

    # Test with string value and custom serializer
    result = serialize(custom_serializer, "json", "hello")
    assert result == "hellohello"

    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None

    # Test with dict value and custom serializer
    def dict_serializer(fmt, val):
        return {k: v for k, v in val.items()}
    
    test_dict = {"key": "value"}
    result = serialize(dict_serializer, "json", test_dict)
    assert result == test_dict

    # Test with list value
    result = serialize(PFIELD_NO_SERIALIZER, "json", [1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #9
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    passing_invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    subject = object()
    # Should not raise
    check_global_invariants(subject, passing_invariants)

    # Test case 2: Single invariant fails
    failing_invariants = [
        lambda x: (False, "error_code_1"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, failing_invariants)
    assert exc_info.value.error_codes == ("error_code_1",)

    # Test case 3: Multiple invariants fail
    failing_invariants = [
        lambda x: (False, "error_1"),
        lambda x: (True, None),
        lambda x: (False, "error_2"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, failing_invariants)
    assert exc_info.value.error_codes == ("error_1", "error_2")

    # Test case 4: Empty invariants list
    check_global_invariants(subject, [])

    # Test case 5: Mixed pass/fail with multiple failures
    failing_invariants = [
        lambda x: (True, "ignored_1"),
        lambda x: (False, "code_a"),
        lambda x: (True, "ignored_2"),
        lambda x: (False, "code_b"),
        lambda x: (False, "code_c"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, failing_invariants)
    assert exc_info.value.error_codes == ("code_a", "code_b", "code_c")
    assert exc_info.value.invariant_errors == ()
    assert "Global invariant failed" in str(exc_info.value)


# LLM-generated content at query #10
#--------------------------

```python
def test_check_type():
    """Unit tests for check_type function."""
    
    # Test 1: Valid type - should not raise
    class TestClass:
        pass
    
    test_field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, 
                         initial=PFIELD_NO_INITIAL, mandatory=False, 
                         factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    check_type(TestClass, test_field, "test_field", 42)
    
    # Test 2: Invalid type - should raise PTypeError
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, test_field, "test_field", "string")
    
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == {int}
    assert exc_info.value.actual_type == str
    
    # Test 3: Multiple allowed types - valid type should not raise
    test_field_multi = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT,
                               initial=PFIELD_NO_INITIAL, mandatory=False,
                               factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    check_type(TestClass, test_field_multi, "test_field", "string")
    check_type(TestClass, test_field_multi, "test_field", 42)
    
    # Test 4: Multiple allowed types - invalid type should raise
    with pytest.raises(PTypeError):
        check_type(TestClass, test_field_multi, "test_field", [])
    
    # Test 5: No type constraint (empty type set) - should not raise
    test_field_no_type = _PField(type=set(), invariant=PFIELD_NO_INVARIANT,
                                 initial=PFIELD_NO_INITIAL, mandatory=False,
                                 factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    check_type(TestClass, test_field_no_type, "test_field", "anything")
    check_type(TestClass, test_field_no_type, "test_field", 123)
    check_type(TestClass, test_field_no_type, "test_field", None)
    
    # Test 6: Subclass of allowed type - should not raise
    class SubInt(int):
        pass
    
    check_type(TestClass, test_field, "test_field", SubInt(42))
    
    # Test 7: Error message format
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, test_field, "my_field", 3.14)
    
    error_msg = str(exc_info.value)
    assert "Invalid type for field TestClass.my_field" in error_msg
    assert "float" in error_msg


# LLM-generated content at query #11
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return "serialized_value"
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_value"
    
    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", "plain_value")
    assert result == "plain_value"
    
    # Test with custom serializer
    custom_serializer = lambda fmt, val: f"custom_{val}"
    result = serialize(custom_serializer, "json", "test")
    assert result == "custom_test"
    
    # Test with CheckedType and custom serializer (custom serializer takes precedence)
    mock_obj = MockCheckedType()
    result = serialize(custom_serializer, "json", mock_obj)
    assert result == "custom_<MockCheckedType object"
    
    # Test with different formats
    format_serializer = lambda fmt, val: f"{fmt}:{val}"
    result = serialize(format_serializer, "xml", "data")
    assert result == "xml:data"
    
    result = serialize(format_serializer, "yaml", "data")
    assert result == "yaml:data"
    
    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test with numeric value
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test with dict value
    result = serialize(PFIELD_NO_SERIALIZER, "json", {"key": "value"})
    assert result == {"key": "value"}


# LLM-generated content at query #12
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    subject = object()
    # Should not raise
    check_global_invariants(subject, invariants)

    # Test case 2: Single invariant fails
    invariants = [
        lambda x: (False, "error_code_1"),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("error_code_1",)

    # Test case 3: Multiple invariants, one fails
    invariants = [
        lambda x: (True, None),
        lambda x: (False, "error_code_2"),
        lambda x: (True, None),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("error_code_2",)

    # Test case 4: Multiple invariants fail
    invariants = [
        lambda x: (False, "error_1"),
        lambda x: (False, "error_2"),
        lambda x: (True, None),
        lambda x: (False, "error_3"),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("error_1", "error_2", "error_3")

    # Test case 5: Empty invariants list
    invariants = []
    subject = object()
    # Should not raise
    check_global_invariants(subject, invariants)

    # Test case 6: Invariants accessing subject properties
    class TestSubject:
        def __init__(self, value):
            self.value = value

    invariants = [
        lambda x: (x.value > 0, "negative_value"),
    ]
    subject = TestSubject(5)
    # Should not raise
    check_global_invariants(subject, invariants)

    subject = TestSubject(-1)
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("negative_value",)

    # Test case 7: Check exception has correct attributes
    invariants = [
        lambda x: (False, "test_error"),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("test_error",)
    assert exc_info.value.invariant_errors == ()
    assert "Global invariant failed" in str(exc_info.value)


# LLM-generated content at query #13
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test when ignore_extra is False (default)
    test_field = field(type=CheckedPVector)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, False)
    assert result is False

    # Test when ignore_extra is True but field type doesn't match
    test_field = field(type=int)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is False

    # Test when ignore_extra is True, field type matches, but factory doesn't have ignore_extra param
    test_field = field(type=CheckedPVector, factory=lambda x: x)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is False

    # Test when ignore_extra is True, field type matches, and factory has ignore_extra param
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    
    test_field = field(type=CheckedPVector, factory=factory_with_ignore_extra)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is True

    # Test with CheckedPSet type
    def factory_with_ignore_extra_pset(x, ignore_extra=False):
        return x
    
    test_field = field(type=CheckedPSet, factory=factory_with_ignore_extra_pset)
    result = is_field_ignore_extra_complaint(CheckedPSet, test_field, True)
    assert result is True

    # Test with set of types
    def factory_with_ignore_extra_multi(x, ignore_extra=False):
        return x
    
    test_field = field(type={CheckedPVector}, factory=factory_with_ignore_extra_multi)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is True

    # Test with multiple types in tuple where first is CheckedPVector
    def factory_with_ignore_extra_tuple(x, ignore_extra=False):
        return x
    
    test_field = field(type=(CheckedPVector, int), factory=factory_with_ignore_extra_tuple)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is True

    # Test with PFIELD_NO_FACTORY
    test_field = field(type=CheckedPVector)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    # PFIELD_NO_FACTORY is a lambda that doesn't have ignore_extra param
    assert result is False


# LLM-generated content at query #14
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    passing_invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    subject = object()
    # Should not raise
    check_global_invariants(subject, passing_invariants)

    # Test case 2: Single invariant fails
    failing_invariants = [
        lambda x: (False, "error_code_1"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, failing_invariants)
    assert exc_info.value.error_codes == ("error_code_1",)

    # Test case 3: Multiple invariants, some fail
    mixed_invariants = [
        lambda x: (True, None),
        lambda x: (False, "error_code_2"),
        lambda x: (False, "error_code_3"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, mixed_invariants)
    assert set(exc_info.value.error_codes) == {"error_code_2", "error_code_3"}

    # Test case 4: Empty invariants list
    check_global_invariants(subject, [])

    # Test case 5: All invariants fail
    all_failing_invariants = [
        lambda x: (False, "error_1"),
        lambda x: (False, "error_2"),
        lambda x: (False, "error_3"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, all_failing_invariants)
    assert len(exc_info.value.error_codes) == 3
    assert "Global invariant failed" in str(exc_info.value)

    # Test case 6: Verify InvariantException contains expected attributes
    try:
        check_global_invariants(subject, [lambda x: (False, "test_error")])
    except InvariantException as e:
        assert e.error_codes == ("test_error",)
        assert e.invariant_errors == ()


# LLM-generated content at query #15
#--------------------------

```python
def test_serialize():
    # Test 1: serialize with CheckedType and PFIELD_NO_SERIALIZER
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return "serialized_value"
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_value"
    
    # Test 2: serialize with custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "custom_xml_test_value"
    
    # Test 3: serialize with PFIELD_NO_SERIALIZER and non-CheckedType value
    result = serialize(PFIELD_NO_SERIALIZER, "json", "plain_string")
    assert result == "plain_string"
    
    # Test 4: serialize with custom serializer on non-CheckedType
    def another_serializer(format, value):
        return {"format": format, "value": value}
    
    result = serialize(another_serializer, "csv", 42)
    assert result == {"format": "csv", "value": 42}
    
    # Test 5: serialize with PFIELD_NO_SERIALIZER and numeric value
    result = serialize(PFIELD_NO_SERIALIZER, "binary", 123)
    assert result == 123
    
    # Test 6: serialize with custom serializer and None value
    def none_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(none_serializer, "json", None)
    assert result == "json:None"


# LLM-generated content at query #16
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return "serialized_value"
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_value"
    
    # Test with custom serializer
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "xml:test_value"
    
    # Test with PFIELD_NO_SERIALIZER and non-CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "json", "plain_value")
    assert result == "plain_value"
    
    # Test with custom serializer on non-CheckedType
    def format_serializer(format, value):
        return f"[{format}] {value}"
    
    result = serialize(format_serializer, "csv", 42)
    assert result == "[csv] 42"
    
    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test with custom serializer returning different types
    def type_changing_serializer(format, value):
        return {"format": format, "value": value}
    
    result = serialize(type_changing_serializer, "binary", "data")
    assert result == {"format": "binary", "value": "data"}


# LLM-generated content at query #17
#--------------------------

```python
def test_check_type():
    """Test check_type function for type validation."""
    
    # Test 1: Valid type - should not raise
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(dict, field_obj, 'test_field', 42)  # Should not raise
    
    # Test 2: Invalid type - should raise PTypeError
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(dict, field_obj, 'test_field', "string")
    assert exc_info.value.source_class == dict
    assert exc_info.value.field == 'test_field'
    assert exc_info.value.actual_type == str
    assert exc_info.value.expected_types == {int}
    
    # Test 3: Multiple valid types - should not raise
    field_obj = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(dict, field_obj, 'test_field', 42)  # Should not raise
    check_type(dict, field_obj, 'test_field', "string")  # Should not raise
    
    # Test 4: Multiple types, invalid - should raise
    field_obj = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError):
        check_type(dict, field_obj, 'test_field', 3.14)
    
    # Test 5: No type constraint - should not raise
    field_obj = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(dict, field_obj, 'test_field', "anything")  # Should not raise
    check_type(dict, field_obj, 'test_field', 42)  # Should not raise
    
    # Test 6: Error message contains correct field and class names
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(dict, field_obj, 'my_field', [])
    assert 'dict.my_field' in str(exc_info.value)
    assert 'list' in str(exc_info.value)
    
    # Test 7: Subclass instances should be accepted
    class MyInt(int):
        pass
    
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(dict, field_obj, 'test_field', MyInt(5))  # Should not raise


# LLM-generated content at query #18
#--------------------------

def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return "serialized_value"
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_value"
    
    # Test with custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "custom_xml_test_value"
    
    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test with custom serializer on non-CheckedType
    def format_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(format_serializer, "csv", "data")
    assert result == "csv:data"
    
    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test with complex custom serializer
    def json_serializer(format, value):
        if format == "json":
            return {"type": "json", "value": value}
        return value
    
    result = serialize(json_serializer, "json", "test")
    assert result == {"type": "json", "value": "test"}


# LLM-generated content at query #19
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return "serialized_value"
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_value"
    
    # Test with custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "json", "test_value")
    assert result == "custom_json_test_value"
    
    # Test with CheckedType and custom serializer
    result = serialize(custom_serializer, "xml", mock_obj)
    assert result == f"custom_xml_{mock_obj}"
    
    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", "plain_value")
    assert result == "plain_value"
    
    # Test with non-CheckedType and custom serializer
    result = serialize(custom_serializer, "csv", 123)
    assert result == "custom_csv_123"


# LLM-generated content at query #20
#--------------------------

```python
def test_pmap_field():
    """Test pmap_field function for creating checked PMap fields."""
    
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj is not None
    assert field_obj.mandatory is True
    assert callable(field_obj.factory)
    
    # Test that the field type is a CheckedPMap subclass
    assert len(field_obj.type) == 1
    field_type = get_type(tuple(field_obj.type)[0])
    assert issubclass(field_type, CheckedPMap)
    
    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.mandatory is True
    assert len(optional_field.type) == 2  # Should accept the type or None
    
    # Test factory with non-optional field
    factory = field_obj.factory
    result = factory({'key1': 1, 'key2': 2})
    assert result is not None
    assert isinstance(result, CheckedPMap)
    
    # Test factory with optional field and None value
    optional_factory = optional_field.factory
    result_none = optional_factory(None)
    assert result_none is None
    
    # Test factory with optional field and actual value
    result_value = optional_factory({'key1': 1})
    assert result_value is not None
    assert isinstance(result_value, CheckedPMap)
    
    # Test initial value
    assert field_obj.initial is not None
    assert isinstance(field_obj.initial, CheckedPMap)
    
    # Test with custom invariant
    custom_invariant = lambda x: (len(x) > 0, "Map must not be empty")
    field_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_invariant.invariant is not None
    assert callable(field_with_invariant.invariant)
    
    # Test caching of pmap field types
    field1 = pmap_field(str, int)
    field2 = pmap_field(str, int)
    type1 = get_type(tuple(field1.type)[0])
    type2 = get_type(tuple(field2.type)[0])
    assert type1 is type2  # Should be the same cached type
    
    # Test different key/value types create different field types
    field_int_str = pmap_field(int, str)
    type_int_str = get_type(tuple(field_int_str.type)[0])
    type_str_int = get_type(tuple(field1.type)[0])
    assert type_int_str is not type_str_int
    
    # Test that pmap_field returns a _PField instance
    assert isinstance(field_obj, _PField)


# LLM-generated content at query #21
#--------------------------

```python
def test_pmap_field():
    """Test pmap_field function for creating checked PMap fields."""
    
    # Test basic pmap_field creation
    pmap_fld = pmap_field(str, int)
    assert pmap_fld.mandatory is True
    assert pmap_fld.factory is not None
    assert callable(pmap_fld.factory)
    
    # Test that initial value is an empty TheMap instance
    assert pmap_fld.initial is not None
    
    # Test optional=False (default)
    pmap_fld_required = pmap_field(str, int, optional=False)
    assert pmap_fld_required.mandatory is True
    
    # Test optional=True
    pmap_fld_optional = pmap_field(str, int, optional=True)
    assert pmap_fld_optional.mandatory is True
    # When optional, the type should allow None
    assert pmap_fld_optional.type is not None
    
    # Test factory with non-optional field
    pmap_fld_non_opt = pmap_field(str, int, optional=False)
    test_dict = {"a": 1, "b": 2}
    result = pmap_fld_non_opt.factory(test_dict)
    assert result is not None
    
    # Test factory with optional field and None value
    pmap_fld_opt = pmap_field(str, int, optional=True)
    result_none = pmap_fld_opt.factory(None)
    assert result_none is None
    
    # Test factory with optional field and dict value
    result_dict = pmap_fld_opt.factory({"x": 10, "y": 20})
    assert result_dict is not None
    
    # Test with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")
    
    pmap_fld_inv = pmap_field(str, int, invariant=custom_invariant)
    assert pmap_fld_inv.invariant is not None
    assert callable(pmap_fld_inv.invariant)
    
    # Test with different types
    pmap_fld_int_str = pmap_field(int, str)
    assert pmap_fld_int_str.mandatory is True
    
    # Test that the field is a _PField instance
    assert isinstance(pmap_fld, _PField)
    
    # Test serializer is set to default
    assert pmap_fld.serializer is not None
    assert callable(pmap_fld.serializer)
    
    # Test caching: calling pmap_field with same types should use cached type
    pmap_fld_1 = pmap_field(str, int)
    pmap_fld_2 = pmap_field(str, int)
    # Both should have factories that create the same type class
    assert pmap_fld_1.factory is not None
    assert pmap_fld_2.factory is not None


# LLM-generated content at query #22
#--------------------------

```python
def test_pmap_field():
    """Test pmap_field function creates proper checked PMap fields."""
    import pytest
    from pyrsistent import pmap
    
    # Test basic pmap_field creation with required types
    str_to_int_field = pmap_field(str, int)
    assert str_to_int_field.type is not None
    assert str_to_int_field.mandatory is True
    assert str_to_int_field.initial == pmap()
    
    # Test that factory works correctly
    result = str_to_int_field.factory({"a": 1, "b": 2})
    assert result["a"] == 1
    assert result["b"] == 2
    
    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type is not None
    
    # Test optional factory with None
    none_result = optional_field.factory(None)
    assert none_result is None
    
    # Test optional factory with dict
    dict_result = optional_field.factory({"x": 10})
    assert dict_result["x"] == 10
    
    # Test with custom invariant
    def positive_values(pmap_obj):
        return (True, "")
    
    field_with_invariant = pmap_field(str, int, invariant=positive_values)
    assert field_with_invariant.invariant is not None
    
    # Test that the same key_type and value_type returns cached type
    field1 = pmap_field(str, int)
    field2 = pmap_field(str, int)
    assert field1.type == field2.type
    
    # Test different key/value types create different field types
    field3 = pmap_field(int, str)
    assert field1.type != field3.type
    
    # Test initial value is empty pmap
    assert len(str_to_int_field.initial) == 0
    
    # Test with multiple type combinations
    int_to_str_field = pmap_field(int, str)
    result = int_to_str_field.factory({1: "one", 2: "two"})
    assert result[1] == "one"
    assert result[2] == "two"


# LLM-generated content at query #23
#--------------------------

def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False (default)
    test_field = _PField(type={CheckedPVector}, invariant=PFIELD_NO_INVARIANT,
                         initial=PFIELD_NO_INITIAL, mandatory=False,
                         factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, False) is False

    # Test with ignore_extra=True but field type is not CheckedPVector/CheckedPSet
    test_field_str = _PField(type={str}, invariant=PFIELD_NO_INVARIANT,
                             initial=PFIELD_NO_INITIAL, mandatory=False,
                             factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field_str, True) is False

    # Test with ignore_extra=True, correct type, but factory doesn't have ignore_extra param
    test_field_no_param = _PField(type={CheckedPVector}, invariant=PFIELD_NO_INVARIANT,
                                  initial=PFIELD_NO_INITIAL, mandatory=False,
                                  factory=lambda x: x, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field_no_param, True) is False

    # Test with ignore_extra=True, correct type, and factory has ignore_extra param
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x

    test_field_with_param = _PField(type={CheckedPVector}, invariant=PFIELD_NO_INVARIANT,
                                    initial=PFIELD_NO_INITIAL, mandatory=False,
                                    factory=factory_with_ignore_extra,
                                    serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field_with_param, True) is True

    # Test with ignore_extra=True, CheckedPSet type
    test_field_pset = _PField(type={CheckedPSet}, invariant=PFIELD_NO_INVARIANT,
                              initial=PFIELD_NO_INITIAL, mandatory=False,
                              factory=factory_with_ignore_extra,
                              serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPSet, test_field_pset, True) is True

    # Test with set type (field.type is a set)
    test_field_set_type = _PField(type={CheckedPVector, str}, invariant=PFIELD_NO_INVARIANT,
                                  initial=PFIELD_NO_INITIAL, mandatory=False,
                                  factory=factory_with_ignore_extra,
                                  serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field_set_type, True) is True


# LLM-generated content at query #24
#--------------------------

```python
def test_serialize():
    # Test 1: serialize with CheckedType and PFIELD_NO_SERIALIZER
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}"
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_json"
    
    # Test 2: serialize with CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"custom_{fmt}_{val}"
    mock_obj = MockCheckedType()
    result = serialize(custom_serializer, "xml", mock_obj)
    assert result == "custom_xml_<MockCheckedType object at"
    
    # Test 3: serialize non-CheckedType with PFIELD_NO_SERIALIZER
    value = 42
    result = serialize(PFIELD_NO_SERIALIZER, "json", value)
    assert result == 42
    
    # Test 4: serialize with custom serializer
    custom_serializer = lambda fmt, val: f"{fmt}:{val}"
    result = serialize(custom_serializer, "json", "test_value")
    assert result == "json:test_value"
    
    # Test 5: serialize with PFIELD_NO_SERIALIZER on primitive type
    result = serialize(PFIELD_NO_SERIALIZER, "xml", "string_value")
    assert result == "string_value"
    
    # Test 6: serialize with complex serializer and format
    complex_serializer = lambda fmt, val: {fmt: val}
    result = serialize(complex_serializer, "protobuf", [1, 2, 3])
    assert result == {"protobuf": [1, 2, 3]}
    
    # Test 7: serialize None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 8: serialize dict with custom serializer
    dict_serializer = lambda fmt, val: f"dict_{fmt}"
    result = serialize(dict_serializer, "yaml", {"key": "value"})
    assert result == "dict_yaml"


# LLM-generated content at query #25
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    pmap_f = pmap_field(str, int)
    assert pmap_f.mandatory is True
    assert pmap_f.initial == {}
    assert callable(pmap_f.factory)
    
    # Test with optional=True
    pmap_f_optional = pmap_field(str, int, optional=True)
    assert pmap_f_optional.mandatory is True
    assert callable(pmap_f_optional.factory)
    
    # Test factory creates CheckedPMap
    result = pmap_f.factory({'a': 1, 'b': 2})
    assert isinstance(result, CheckedPMap)
    assert result == {'a': 1, 'b': 2}
    
    # Test optional factory with None
    result_none = pmap_f_optional.factory(None)
    assert result_none is None
    
    # Test optional factory with dict
    result_dict = pmap_f_optional.factory({'x': 10})
    assert isinstance(result_dict, CheckedPMap)
    assert result_dict == {'x': 10}
    
    # Test with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) > 0, "PMap must not be empty")
    
    pmap_f_inv = pmap_field(str, int, invariant=custom_invariant)
    assert pmap_f_inv.invariant is not None
    assert callable(pmap_f_inv.invariant)
    
    # Test type is set correctly
    pmap_f_types = pmap_field(str, int)
    assert len(pmap_f_types.type) > 0
    
    # Test optional type is wrapped
    pmap_f_opt_types = pmap_field(str, int, optional=True)
    assert len(pmap_f_opt_types.type) > 0
    
    # Test that generated type is cached
    pmap_f1 = pmap_field(str, int)
    pmap_f2 = pmap_field(str, int)
    # Both should use the same cached type
    assert pmap_f1.factory({}).__class__ == pmap_f2.factory({}).__class__
    
    # Test with different key/value types
    pmap_f_int_str = pmap_field(int, str)
    result_int_str = pmap_f_int_str.factory({1: 'a', 2: 'b'})
    assert isinstance(result_int_str, CheckedPMap)
    assert result_int_str == {1: 'a', 2: 'b'}
    
    # Test initial value is empty map
    assert pmap_f.initial == {}
    assert isinstance(pmap_f.initial, CheckedPMap)


# LLM-generated content at query #26
#--------------------------

```python
def test_serialize():
    # Test 1: serialize with CheckedType and PFIELD_NO_SERIALIZER uses CheckedType.serialize
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}"
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_json"
    
    # Test 2: serialize with CheckedType and custom serializer uses custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{type(value).__name__}"
    
    mock_obj = MockCheckedType()
    result = serialize(custom_serializer, "xml", mock_obj)
    assert result == "custom_xml_MockCheckedType"
    
    # Test 3: serialize with non-CheckedType and PFIELD_NO_SERIALIZER uses PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", "test_value")
    assert result == "test_value"
    
    # Test 4: serialize with non-CheckedType and custom serializer uses custom serializer
    def custom_serializer2(format, value):
        return f"formatted_{format}_{value}"
    
    result = serialize(custom_serializer2, "yaml", 42)
    assert result == "formatted_yaml_42"
    
    # Test 5: serialize with integer and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 123)
    assert result == 123
    
    # Test 6: serialize with dict and custom serializer
    def dict_serializer(format, value):
        return {f"format_{format}": value}
    
    test_dict = {"key": "value"}
    result = serialize(dict_serializer, "proto", test_dict)
    assert result == {"format_proto": test_dict}
    
    # Test 7: serialize with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 8: serialize with list and custom serializer
    def list_serializer(format, value):
        return [format] + value
    
    result = serialize(list_serializer, "csv", [1, 2, 3])
    assert result == ["csv", 1, 2, 3]


# LLM-generated content at query #27
#--------------------------

```python
def test_serialize():
    # Test with non-CheckedType value and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test with non-CheckedType value and custom serializer
    def custom_serializer(format, value):
        return value * 2
    result = serialize(custom_serializer, "json", 21)
    assert result == 42
    
    # Test with CheckedType value and PFIELD_NO_SERIALIZER
    class TestRecord(CheckedType):
        x = field(type=int)
    
    record = TestRecord(x=10)
    result = serialize(PFIELD_NO_SERIALIZER, "json", record)
    # Should call the CheckedType's serialize method
    assert result is not None
    
    # Test with CheckedType value and custom serializer
    result = serialize(custom_serializer, "json", record)
    # Should use the custom serializer, not the CheckedType's serialize method
    assert result is not None
    
    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test with string value and custom serializer
    def string_serializer(format, value):
        return value.upper()
    result = serialize(string_serializer, "json", "hello")
    assert result == "HELLO"
    
    # Test with dict value and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", {"key": "value"})
    assert result == {"key": "value"}
    
    # Test with list value and custom serializer
    def list_serializer(format, value):
        return len(value)
    result = serialize(list_serializer, "json", [1, 2, 3])
    assert result == 3


# LLM-generated content at query #28
#--------------------------

```python
def test_serialize():
    # Test with non-CheckedType value and default serializer
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42

    # Test with non-CheckedType value and custom serializer
    def custom_serializer(format, value):
        return value * 2
    
    result = serialize(custom_serializer, "json", 21)
    assert result == 42

    # Test with CheckedType value and default serializer (PFIELD_NO_SERIALIZER)
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return {"serialized": True, "format": format}
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == {"serialized": True, "format": "json"}

    # Test with CheckedType value and custom serializer (uses custom serializer, not CheckedType.serialize)
    def another_serializer(format, value):
        return {"custom": True}
    
    result = serialize(another_serializer, "xml", mock_obj)
    assert result == {"custom": True}

    # Test with string format parameter
    result = serialize(PFIELD_NO_SERIALIZER, "xml", 100)
    assert result == 100

    # Test with different format types
    result = serialize(PFIELD_NO_SERIALIZER, None, "test_string")
    assert result == "test_string"

    # Test with list value and default serializer
    result = serialize(PFIELD_NO_SERIALIZER, "json", [1, 2, 3])
    assert result == [1, 2, 3]

    # Test with dict value and custom serializer
    def dict_serializer(format, value):
        return {**value, "modified": True}
    
    result = serialize(dict_serializer, "json", {"key": "value"})
    assert result == {"key": "value", "modified": True}


# LLM-generated content at query #29
#--------------------------

def test_pmap_field():
    # Test basic pmap_field creation
    str_to_int_field = pmap_field(str, int)
    assert str_to_int_field.mandatory is True
    assert str_to_int_field.initial == {}
    
    # Test that the type is set correctly
    assert CheckedPMap in [get_type(t) for t in str_to_int_field.type]
    
    # Test factory creates correct type
    result = str_to_int_field.factory({'a': 1, 'b': 2})
    assert isinstance(result, CheckedPMap)
    assert result == {'a': 1, 'b': 2}
    
    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.mandatory is True
    assert optional_field.factory(None) is None
    assert isinstance(optional_field.factory({'x': 10}), CheckedPMap)
    
    # Test with invariant
    def positive_values(m):
        return (all(v > 0 for v in m.values()), "Values must be positive")
    
    inv_field = pmap_field(str, int, invariant=positive_values)
    assert inv_field.invariant is not None
    
    # Test factory with different types
    int_to_str_field = pmap_field(int, str)
    result2 = int_to_str_field.factory({1: 'a', 2: 'b'})
    assert isinstance(result2, CheckedPMap)
    assert result2 == {1: 'a', 2: 'b'}
    
    # Test that _pmap_field_types caches the type
    type1 = _make_pmap_field_type(str, int)
    type2 = _make_pmap_field_type(str, int)
    assert type1 is type2
    
    # Test optional with None factory
    opt_result = optional_field.factory(None)
    assert opt_result is None
    
    # Test non-optional with empty dict
    non_opt_field = pmap_field(str, float, optional=False)
    result3 = non_opt_field.factory({})
    assert isinstance(result3, CheckedPMap)
    assert len(result3) == 0


# LLM-generated content at query #30
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False (default), should return False
    test_field = _PField(
        type={CheckedPVector},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, False) is False

    # Test with ignore_extra=True but field type is not CheckedPVector/CheckedPSet
    test_field_str = _PField(
        type={str},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field_str, True) is False

    # Test with ignore_extra=True and matching type, but factory without ignore_extra param
    test_field_no_param = _PField(
        type={CheckedPVector},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=lambda x: x,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field_no_param, True) is False

    # Test with ignore_extra=True, matching type, and factory with ignore_extra param
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    
    test_field_with_param = _PField(
        type={CheckedPVector},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=factory_with_ignore_extra,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field_with_param, True) is True

    # Test with CheckedPSet type
    test_field_pset = _PField(
        type={CheckedPSet},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=factory_with_ignore_extra,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPSet, test_field_pset, True) is True

    # Test with set type (multiple types)
    test_field_set_type = _PField(
        type={CheckedPVector, str},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=factory_with_ignore_extra,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field_set_type, True) is True


# LLM-generated content at query #31
#--------------------------

```python
def test_serialize():
    # Test with PFIELD_NO_SERIALIZER and CheckedType instance
    class TestCheckedType(CheckedType):
        x = field(type=int)
    
    instance = TestCheckedType(x=42)
    result = serialize(PFIELD_NO_SERIALIZER, "json", instance)
    assert result == instance.serialize("json")
    
    # Test with PFIELD_NO_SERIALIZER and non-CheckedType value
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test with custom serializer and CheckedType instance
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    result = serialize(custom_serializer, "json", instance)
    assert result == f"serialized_{instance}"
    
    # Test with custom serializer and non-CheckedType value
    result = serialize(custom_serializer, "json", 42)
    assert result == "serialized_42"
    
    # Test with different format
    result = serialize(PFIELD_NO_SERIALIZER, "xml", 100)
    assert result == 100
    
    # Test with identity serializer
    identity_serializer = lambda fmt, val: val
    result = serialize(identity_serializer, "json", "test_value")
    assert result == "test_value"


# LLM-generated content at query #32
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False (default)
    test_field = field(type=CheckedPVector)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, False) is False

    # Test with ignore_extra=True but field type doesn't match
    test_field = field(type=int)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is False

    # Test with ignore_extra=True and matching type but factory doesn't have ignore_extra param
    test_field = field(type=CheckedPVector, factory=lambda x: x)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is False

    # Test with ignore_extra=True, matching type, and factory with ignore_extra param
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    test_field = field(type=CheckedPVector, factory=factory_with_ignore_extra)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is True

    # Test with ignore_extra=True, matching type, and default factory (CheckedType.create)
    test_field = field(type=CheckedPVector)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is True

    # Test with ignore_extra=True, matching type set, and factory with ignore_extra param
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    test_field = field(type={CheckedPVector}, factory=factory_with_ignore_extra)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is True

    # Test with ignore_extra=True, matching type tuple, and factory with ignore_extra param
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    test_field = field(type=(CheckedPVector,), factory=factory_with_ignore_extra)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is True

    # Test with CheckedPSet type
    test_field = field(type=CheckedPSet)
    assert is_field_ignore_extra_complaint(CheckedPSet, test_field, True) is True

    # Test with multiple types where first is CheckedPVector
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    test_field = field(type=[CheckedPVector, int], factory=factory_with_ignore_extra)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is True


# LLM-generated content at query #33
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False (default)
    f = field(type=CheckedPVector)
    assert is_field_ignore_extra_complaint(CheckedPVector, f, False) is False
    
    # Test with ignore_extra=True but field type doesn't match
    f = field(type=int)
    assert is_field_ignore_extra_complaint(CheckedPVector, f, True) is False
    
    # Test with ignore_extra=True and matching type, but factory doesn't have ignore_extra param
    f = field(type=CheckedPVector, factory=lambda x: x)
    assert is_field_ignore_extra_complaint(CheckedPVector, f, True) is False
    
    # Test with ignore_extra=True, matching type, and factory with ignore_extra param
    f = field(type=CheckedPVector)
    result = is_field_ignore_extra_complaint(CheckedPVector, f, True)
    assert result is True
    
    # Test with ignore_extra=True and CheckedPSet type
    f = field(type=CheckedPSet)
    result = is_field_ignore_extra_complaint(CheckedPSet, f, True)
    assert result is True
    
    # Test with ignore_extra=True and set of types including CheckedPVector
    f = field(type={CheckedPVector})
    result = is_field_ignore_extra_complaint(CheckedPVector, f, True)
    assert result is True
    
    # Test with ignore_extra=True and tuple of types
    f = field(type=(CheckedPVector,))
    result = is_field_ignore_extra_complaint(CheckedPVector, f, True)
    assert result is True
    
    # Test with ignore_extra=True but type is not a CheckedType subclass
    f = field(type=str)
    assert is_field_ignore_extra_complaint(CheckedPVector, f, True) is False
    
    # Test with custom factory that has ignore_extra parameter
    def custom_factory(arg, ignore_extra=False):
        return arg
    
    f = field(type=CheckedPVector, factory=custom_factory)
    result = is_field_ignore_extra_complaint(CheckedPVector, f, True)
    assert result is True
    
    # Test with custom factory that doesn't have ignore_extra parameter
    def custom_factory_no_param(arg):
        return arg
    
    f = field(type=CheckedPVector, factory=custom_factory_no_param)
    result = is_field_ignore_extra_complaint(CheckedPVector, f, True)
    assert result is False


# LLM-generated content at query #34
#--------------------------

```python
def test_serialize():
    # Test with PFIELD_NO_SERIALIZER and CheckedType value
    class TestRecord(CheckedType):
        def serialize(self, format):
            return {"serialized": True, "format": format}
    
    test_value = TestRecord()
    result = serialize(PFIELD_NO_SERIALIZER, "json", test_value)
    assert result == {"serialized": True, "format": "json"}
    
    # Test with PFIELD_NO_SERIALIZER and non-CheckedType value
    result = serialize(PFIELD_NO_SERIALIZER, "json", "plain_string")
    assert result == "plain_string"
    
    # Test with custom serializer
    custom_serializer = lambda fmt, val: f"custom_{fmt}_{val}"
    result = serialize(custom_serializer, "xml", "test_value")
    assert result == "custom_xml_test_value"
    
    # Test with custom serializer and CheckedType value
    test_value = TestRecord()
    result = serialize(custom_serializer, "json", test_value)
    assert result == f"custom_json_{test_value}"
    
    # Test with PFIELD_NO_SERIALIZER and None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test with PFIELD_NO_SERIALIZER and numeric value
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test with custom serializer and various formats
    serializer = lambda fmt, val: f"{fmt}:{val}"
    assert serialize(serializer, "csv", "data") == "csv:data"
    assert serialize(serializer, "xml", 123) == "xml:123"
    assert serialize(serializer, "yaml", [1, 2, 3]) == "yaml:[1, 2, 3]"


# LLM-generated content at query #35
#--------------------------

def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False (default), should return False
    test_field = _PField(type={CheckedPVector}, invariant=PFIELD_NO_INVARIANT, 
                         initial=PFIELD_NO_INITIAL, mandatory=False, 
                         factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, False) is False

    # Test with ignore_extra=True but field type doesn't match, should return False
    test_field = _PField(type={str}, invariant=PFIELD_NO_INVARIANT,
                         initial=PFIELD_NO_INITIAL, mandatory=False,
                         factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is False

    # Test with ignore_extra=True, matching type, but factory doesn't have ignore_extra param
    test_field = _PField(type={CheckedPVector}, invariant=PFIELD_NO_INVARIANT,
                         initial=PFIELD_NO_INITIAL, mandatory=False,
                         factory=lambda x: x, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is False

    # Test with ignore_extra=True, matching type, and factory has ignore_extra param
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    
    test_field = _PField(type={CheckedPVector}, invariant=PFIELD_NO_INVARIANT,
                         initial=PFIELD_NO_INITIAL, mandatory=False,
                         factory=factory_with_ignore_extra, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is True

    # Test with ignore_extra=True, matching type set, and factory has ignore_extra param
    def factory_with_ignore_extra_set(x, ignore_extra=False):
        return x
    
    test_field = _PField(type={CheckedPSet}, invariant=PFIELD_NO_INVARIANT,
                         initial=PFIELD_NO_INITIAL, mandatory=False,
                         factory=factory_with_ignore_extra_set, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPSet, test_field, True) is True

    # Test with set type field (should return False since set type doesn't match subclass check)
    test_field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT,
                         initial=PFIELD_NO_INITIAL, mandatory=False,
                         factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is False


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serialize():
    # Test with PFIELD_NO_SERIALIZER and CheckedType value
    class TestCheckedType(CheckedType):
        x = field(type=int)
    
    test_obj = TestCheckedType(x=5)
    result = serialize(PFIELD_NO_SERIALIZER, "json", test_obj)
    assert result == test_obj.serialize("json")
    
    # Test with PFIELD_NO_SERIALIZER and non-CheckedType value
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test with custom serializer and CheckedType value
    def custom_serializer(fmt, val):
        return f"custom_{fmt}_{val}"
    
    test_obj = TestCheckedType(x=5)
    result = serialize(custom_serializer, "json", test_obj)
    assert result == f"custom_json_{test_obj}"
    
    # Test with custom serializer and non-CheckedType value
    result = serialize(custom_serializer, "xml", "test")
    assert result == "custom_xml_test"
    
    # Test with default serializer (PFIELD_NO_SERIALIZER identity)
    result = serialize(PFIELD_NO_SERIALIZER, "any_format", [1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test with lambda serializer
    result = serialize(lambda fmt, val: val * 2, "json", 5)
    assert result == 10


# LLM-generated content at query #2
#--------------------------

```python
def test_set_fields():
    """Test set_fields function for proper field extraction and organization."""
    
    # Test 1: Basic field extraction from a single base class
    class BaseClass:
        __fields__ = {}
    
    dct = {}
    bases = (BaseClass,)
    set_fields(dct, bases, '__fields__')
    
    assert '__fields__' in dct
    assert isinstance(dct['__fields__'], dict)
    assert dct['__fields__'] == {}
    
    # Test 2: Field extraction with _PField instances
    field1 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    field2 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    dct = {'name': field1, 'age': field2}
    
    class Base:
        __fields__ = {}
    
    bases = (Base,)
    set_fields(dct, bases, '__fields__')
    
    assert '__fields__' in dct
    assert 'name' in dct['__fields__']
    assert 'age' in dct['__fields__']
    assert dct['__fields__']['name'] is field1
    assert dct['__fields__']['age'] is field2
    assert 'name' not in dct
    assert 'age' not in dct
    
    # Test 3: Multiple inheritance - fields from multiple base classes
    field3 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    field4 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    class Base1:
        __fields__ = {'field1': field3}
    
    class Base2:
        __fields__ = {'field2': field4}
    
    dct = {}
    bases = (Base1, Base2)
    set_fields(dct, bases, '__fields__')
    
    assert '__fields__' in dct
    assert 'field1' in dct['__fields__']
    assert 'field2' in dct['__fields__']
    assert dct['__fields__']['field1'] is field3
    assert dct['__fields__']['field2'] is field4
    
    # Test 4: Mixed _PField and non-_PField attributes
    field5 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    dct = {'field': field5, 'other_attr': 'value', 'another': 123}
    
    class Base:
        __fields__ = {}
    
    bases = (Base,)
    set_fields(dct, bases, '__fields__')
    
    assert '__fields__' in dct
    assert 'field' in dct['__fields__']
    assert 'other_attr' in dct
    assert 'another' in dct
    assert dct['other_attr'] == 'value'
    assert dct['another'] == 123
    
    # Test 5: Base class without __fields__ attribute
    dct = {}
    
    class BaseNoFields:
        pass
    
    bases = (BaseNoFields,)
    set_fields(dct, bases, '__fields__')
    
    assert '__fields__' in dct
    assert dct['__fields__'] == {}
    
    # Test 6: Field override from child over parent
    field6 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    field7 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    class BaseWithField:
        __fields__ = {'shared': field6}
    
    dct = {'shared': field7}
    bases = (BaseWithField,)
    set_fields(dct, bases, '__fields__')
    
    assert '__fields__' in dct
    assert 'shared' in dct['__fields__']
    assert dct['__fields__']['shared'] is field7  # Child overrides parent
    assert 'shared' not in dct  # Removed from main dict


# LLM-generated content at query #3
#--------------------------

```python
def test_set_fields():
    # Test with empty dictionary and no bases
    dct = {}
    bases = ()
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct[name] == {}
    assert 'fields' not in dct or isinstance(dct.get('fields'), dict)

    # Test with bases that have fields
    class Base1:
        __dict__ = {'fields': {'field1': _PField(type=set(), invariant=PFIELD_NO_INVARIANT, 
                                                    initial=PFIELD_NO_INITIAL, mandatory=False, 
                                                    factory=PFIELD_NO_FACTORY, 
                                                    serializer=PFIELD_NO_SERIALIZER)}}

    dct = {}
    bases = (Base1,)
    name = 'fields'
    set_fields(dct, bases, name)
    assert 'field1' in dct[name]

    # Test with _PField instances in dictionary
    field1 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, 
                     initial=PFIELD_NO_INITIAL, mandatory=False, 
                     factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    dct = {'my_field': field1}
    bases = ()
    name = 'fields'
    set_fields(dct, bases, name)
    assert 'my_field' in dct[name]
    assert dct[name]['my_field'] is field1
    assert 'my_field' not in dct

    # Test with multiple bases
    class Base2:
        __dict__ = {'fields': {'field2': _PField(type=set(), invariant=PFIELD_NO_INVARIANT, 
                                                    initial=PFIELD_NO_INITIAL, mandatory=False, 
                                                    factory=PFIELD_NO_FACTORY, 
                                                    serializer=PFIELD_NO_SERIALIZER)}}

    field3 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, 
                     initial=PFIELD_NO_INITIAL, mandatory=False, 
                     factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    dct = {'field3': field3}
    bases = (Base1, Base2)
    name = 'fields'
    set_fields(dct, bases, name)
    assert 'field1' in dct[name]
    assert 'field2' in dct[name]
    assert 'field3' in dct[name]
    assert 'field3' not in dct

    # Test with non-_PField items in dictionary (should not be moved)
    dct = {'regular_attr': 'value', 'another': 123}
    bases = ()
    name = 'fields'
    set_fields(dct, bases, name)
    assert dct[name] == {}
    assert 'regular_attr' in dct
    assert 'another' in dct

    # Test mixed case
    field_obj = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, 
                        initial=PFIELD_NO_INITIAL, mandatory=False, 
                        factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    dct = {'field_obj': field_obj, 'regular': 'attr'}
    bases = ()
    name = 'fields'
    set_fields(dct, bases, name)
    assert 'field_obj' in dct[name]
    assert 'regular' in dct
    assert 'field_obj' not in dct


# LLM-generated content at query #4
#--------------------------

```python
import pytest


def test_check_global_invariants():
    # Test case 1: All invariants pass
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    subject = object()
    # Should not raise
    check_global_invariants(subject, invariants)

    # Test case 2: Single invariant fails
    invariants = [
        lambda x: (False, "error_1"),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("error_1",)

    # Test case 3: Multiple invariants, one fails
    invariants = [
        lambda x: (True, None),
        lambda x: (False, "error_2"),
        lambda x: (True, None),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("error_2",)

    # Test case 4: Multiple invariants fail
    invariants = [
        lambda x: (False, "error_1"),
        lambda x: (False, "error_2"),
        lambda x: (False, "error_3"),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("error_1", "error_2", "error_3")

    # Test case 5: Empty invariants list
    invariants = []
    subject = object()
    # Should not raise
    check_global_invariants(subject, invariants)

    # Test case 6: Invariants receive the subject correctly
    received_subjects = []
    invariants = [
        lambda x: (received_subjects.append(x), (True, None))[1],
    ]
    subject = {"test": "value"}
    check_global_invariants(subject, invariants)
    assert received_subjects[0] is subject

    # Test case 7: Exception message contains 'Global invariant failed'
    invariants = [
        lambda x: (False, "custom_error"),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert "Global invariant failed" in str(exc_info.value)


# LLM-generated content at query #5
#--------------------------

```python
def test_set_fields():
    # Test with empty dict and no bases
    dct = {}
    bases = ()
    set_fields(dct, bases, 'test_name')
    assert dct['test_name'] == {}

    # Test with bases that have fields
    field1 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    field2 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    class Base1:
        __dict__ = {'_pfields': {'field1': field1}}
    
    class Base2:
        __dict__ = {'_pfields': {'field2': field2}}
    
    dct = {}
    bases = (Base1, Base2)
    set_fields(dct, bases, '_pfields')
    assert 'field1' in dct['_pfields']
    assert 'field2' in dct['_pfields']

    # Test that _PField instances are moved from dct to dct[name]
    field3 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    dct = {'my_field': field3, 'other_value': 'not_a_field'}
    bases = ()
    set_fields(dct, bases, '_pfields')
    
    assert 'my_field' in dct['_pfields']
    assert dct['_pfields']['my_field'] is field3
    assert 'my_field' not in dct
    assert 'other_value' in dct
    assert dct['other_value'] == 'not_a_field'

    # Test with multiple bases and fields in dct
    field4 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    field5 = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                     mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    
    class Base3:
        __dict__ = {'_pfields': {'inherited_field': field4}}
    
    dct = {'own_field': field5}
    bases = (Base3,)
    set_fields(dct, bases, '_pfields')
    
    assert 'inherited_field' in dct['_pfields']
    assert 'own_field' in dct['_pfields']
    assert dct['_pfields']['inherited_field'] is field4
    assert dct['_pfields']['own_field'] is field5
    assert 'own_field' not in dct


# LLM-generated content at query #6
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    subject = object()
    # Should not raise
    check_global_invariants(subject, invariants)

    # Test case 2: Single invariant fails
    invariants = [
        lambda x: (False, "error_code_1"),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("error_code_1",)

    # Test case 3: Multiple invariants fail
    invariants = [
        lambda x: (False, "error_1"),
        lambda x: (False, "error_2"),
        lambda x: (True, None),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert set(exc_info.value.error_codes) == {"error_1", "error_2"}

    # Test case 4: Empty invariants list
    invariants = []
    subject = object()
    # Should not raise
    check_global_invariants(subject, invariants)

    # Test case 5: Invariant with different error code types
    invariants = [
        lambda x: (False, 42),
        lambda x: (False, "string_error"),
    ]
    subject = object()
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert 42 in exc_info.value.error_codes
    assert "string_error" in exc_info.value.error_codes

    # Test case 6: Invariant receives the subject correctly
    received_subjects = []
    def capture_subject(x):
        received_subjects.append(x)
        return (True, None)
    
    invariants = [capture_subject]
    subject = {"test": "object"}
    check_global_invariants(subject, invariants)
    assert received_subjects[0] == subject


# LLM-generated content at query #7
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    def invariant_pass(subject):
        return (True, None)
    
    subject = object()
    invariants = [invariant_pass, invariant_pass]
    # Should not raise
    check_global_invariants(subject, invariants)
    
    # Test case 2: Single invariant fails
    def invariant_fail(subject):
        return (False, "error_code_1")
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant_fail])
    
    assert exc_info.value.error_codes == ("error_code_1",)
    assert exc_info.value.path == ()
    assert "Global invariant failed" in str(exc_info.value)
    
    # Test case 3: Multiple invariants, some fail
    def invariant_fail_2(subject):
        return (False, "error_code_2")
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant_pass, invariant_fail, invariant_fail_2])
    
    assert set(exc_info.value.error_codes) == {"error_code_1", "error_code_2"}
    assert exc_info.value.path == ()
    
    # Test case 4: Empty invariants list
    check_global_invariants(subject, [])
    
    # Test case 5: Multiple passes and multiple fails
    def invariant_fail_3(subject):
        return (False, "error_code_3")
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant_pass, invariant_fail, invariant_pass, invariant_fail_2, invariant_fail_3])
    
    assert set(exc_info.value.error_codes) == {"error_code_1", "error_code_2", "error_code_3"}


# LLM-generated content at query #8
#--------------------------

```python
def test_check_type():
    """Unit tests for check_type function."""
    
    # Test 1: Valid type should not raise exception
    class TestClass:
        pass
    
    f = field(type=int)
    check_type(TestClass, f, "test_field", 42)  # Should not raise
    
    # Test 2: Invalid type should raise PTypeError
    f = field(type=int)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, f, "test_field", "not an int")
    
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.actual_type == str
    assert "Invalid type for field TestClass.test_field" in str(exc_info.value)
    
    # Test 3: Multiple allowed types - first type valid
    f = field(type=[int, str])
    check_type(TestClass, f, "test_field", 42)  # Should not raise
    
    # Test 4: Multiple allowed types - second type valid
    f = field(type=[int, str])
    check_type(TestClass, f, "test_field", "hello")  # Should not raise
    
    # Test 5: Multiple allowed types - invalid type
    f = field(type=[int, str])
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, f, "test_field", 3.14)
    
    assert exc_info.value.actual_type == float
    
    # Test 6: Field with no type constraint should not raise
    f = field(type=PFIELD_NO_TYPE)
    check_type(TestClass, f, "test_field", "anything")  # Should not raise
    
    # Test 7: None value with no type constraint
    f = field(type=PFIELD_NO_TYPE)
    check_type(TestClass, f, "test_field", None)  # Should not raise
    
    # Test 8: Custom class type
    class CustomClass:
        pass
    
    instance = CustomClass()
    f = field(type=CustomClass)
    check_type(TestClass, f, "custom_field", instance)  # Should not raise
    
    # Test 9: Custom class type with wrong type
    f = field(type=CustomClass)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, f, "custom_field", 42)
    
    assert exc_info.value.actual_type == int
    
    # Test 10: Subclass should be accepted
    class ParentClass:
        pass
    
    class ChildClass(ParentClass):
        pass
    
    instance = ChildClass()
    f = field(type=ParentClass)
    check_type(TestClass, f, "parent_field", instance)  # Should not raise
    
    # Test 11: Type set with valid type
    f = field(type={int, str})
    check_type(TestClass, f, "test_field", 42)  # Should not raise
    
    # Test 12: Type set with invalid type
    f = field(type={int, str})
    with pytest.raises(PTypeError):
        check_type(TestClass, f, "test_field", [])


# LLM-generated content at query #9
#--------------------------

```python
def test_pmap_field():
    """Test pmap_field function."""
    
    # Test basic pmap_field creation
    pmap_f = pmap_field(str, int)
    assert pmap_f.mandatory is True
    assert pmap_f.serializer == PFIELD_NO_SERIALIZER
    
    # Test with optional=True
    pmap_f_optional = pmap_field(str, int, optional=True)
    assert pmap_f_optional.mandatory is True
    assert pmap_f_optional.factory(None) is None
    
    # Test with optional=False
    pmap_f_required = pmap_field(str, int, optional=False)
    assert pmap_f_required.mandatory is True
    
    # Test factory creates CheckedPMap
    test_data = {'key1': 1, 'key2': 2}
    result = pmap_f.factory(test_data)
    assert isinstance(result, CheckedPMap)
    
    # Test optional factory with None
    result_none = pmap_f_optional.factory(None)
    assert result_none is None
    
    # Test optional factory with data
    result_data = pmap_f_optional.factory(test_data)
    assert isinstance(result_data, CheckedPMap)
    
    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, "valid")
    
    pmap_f_inv = pmap_field(str, int, invariant=custom_invariant)
    assert pmap_f_inv.invariant is not None
    
    # Test initial value is empty CheckedPMap
    assert len(pmap_f.initial) == 0
    
    # Test type is set correctly
    assert pmap_f.type is not None
    
    # Test factory is callable
    assert callable(pmap_f.factory)
    
    # Test multiple calls create different type instances
    pmap_f1 = pmap_field(str, int)
    pmap_f2 = pmap_field(str, int)
    # Both should use the same cached type
    assert pmap_f1.factory.__self__.__class__ == pmap_f2.factory.__self__.__class__
    
    # Test different key/value types create different types
    pmap_f_str_str = pmap_field(str, str)
    pmap_f_int_int = pmap_field(int, int)
    assert pmap_f_str_str.factory.__self__.__class__ != pmap_f_int_int.factory.__self__.__class__


# LLM-generated content at query #10
#--------------------------

```python
def test_check_type():
    # Test case 1: Valid type - should not raise
    class TestClass:
        pass
    
    field_obj = _PField(type={str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", "string_value")
    
    # Test case 2: Invalid type - should raise PTypeError
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, "test_field", "string_value")
    
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == {int}
    assert exc_info.value.actual_type == str
    
    # Test case 3: Multiple valid types - should not raise
    field_obj = _PField(type={str, int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", 42)
    check_type(TestClass, field_obj, "test_field", "string")
    
    # Test case 4: No type constraint - should not raise
    field_obj = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", "any_value")
    check_type(TestClass, field_obj, "test_field", 123)
    check_type(TestClass, field_obj, "test_field", [1, 2, 3])
    
    # Test case 5: None value with type constraint - should raise
    field_obj = _PField(type={str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, "test_field", None)
    
    assert exc_info.value.actual_type == type(None)
    
    # Test case 6: Subclass should be accepted
    class Parent:
        pass
    
    class Child(Parent):
        pass
    
    field_obj = _PField(type={Parent}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", Child())


# LLM-generated content at query #11
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    passing_invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    subject = object()
    # Should not raise
    check_global_invariants(subject, passing_invariants)

    # Test case 2: Single invariant fails
    failing_invariants = [
        lambda x: (False, "error_code_1"),
    ]
    try:
        check_global_invariants(subject, failing_invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.error_codes == ("error_code_1",)
        assert e.path == ()
        assert "Global invariant failed" in str(e)

    # Test case 3: Multiple invariants, some fail
    mixed_invariants = [
        lambda x: (True, None),
        lambda x: (False, "error_code_2"),
        lambda x: (False, "error_code_3"),
    ]
    try:
        check_global_invariants(subject, mixed_invariants)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert set(e.error_codes) == {"error_code_2", "error_code_3"}
        assert e.path == ()

    # Test case 4: Empty invariants list
    check_global_invariants(subject, [])

    # Test case 5: Invariant with None error code (passing)
    invariants_with_none = [
        lambda x: (True, None),
        lambda x: (True, "some_code"),  # This one passes despite having code
    ]
    check_global_invariants(subject, invariants_with_none)

    # Test case 6: Multiple failing invariants with different error codes
    failing_multiple = [
        lambda x: (False, "code_a"),
        lambda x: (False, "code_b"),
        lambda x: (False, "code_c"),
    ]
    try:
        check_global_invariants(subject, failing_multiple)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert set(e.error_codes) == {"code_a", "code_b", "code_c"}


# LLM-generated content at query #12
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    passing_invariants = [
        lambda x: (True, None),
        lambda x: (True, "success"),
    ]
    # Should not raise any exception
    check_global_invariants("test_subject", passing_invariants)
    
    # Test case 2: Single failing invariant
    failing_invariants = [
        lambda x: (False, "error_code_1"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", failing_invariants)
    assert exc_info.value.error_codes == ("error_code_1",)
    
    # Test case 3: Multiple failing invariants
    multiple_failing_invariants = [
        lambda x: (False, "error_1"),
        lambda x: (False, "error_2"),
        lambda x: (True, None),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", multiple_failing_invariants)
    assert set(exc_info.value.error_codes) == {"error_1", "error_2"}
    
    # Test case 4: Empty invariants list
    check_global_invariants("test_subject", [])
    
    # Test case 5: Invariant with different subject types
    int_invariants = [
        lambda x: (x > 0, "not_positive") if isinstance(x, int) else (True, None),
    ]
    check_global_invariants(42, int_invariants)
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(-5, int_invariants)
    assert exc_info.value.error_codes == ("not_positive",)
    
    # Test case 6: Verify exception attributes
    test_invariants = [
        lambda x: (False, "test_error"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("subject", test_invariants)
    assert exc_info.value.error_codes == ("test_error",)
    assert exc_info.value.message == "Global invariant failed"


# LLM-generated content at query #13
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    """Test is_field_ignore_extra_complaint function."""
    
    # Test case 1: ignore_extra is False (default) - should return False
    test_field = _PField(
        type={str},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPMap, test_field, False) is False
    
    # Test case 2: ignore_extra is True but field type is not CheckedPMap - should return False
    test_field = _PField(
        type={str},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPMap, test_field, True) is False
    
    # Test case 3: ignore_extra is True, type_cls matches, but factory has no ignore_extra param
    test_field = _PField(
        type={CheckedPMap},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=lambda x: x,  # factory without ignore_extra parameter
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPMap, test_field, True) is False
    
    # Test case 4: ignore_extra is True, type_cls matches, factory has ignore_extra param - should return True
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    
    test_field = _PField(
        type={CheckedPMap},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=factory_with_ignore_extra,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPMap, test_field, True) is True
    
    # Test case 5: ignore_extra is True, type is a set (not a subclass check), factory has ignore_extra
    def factory_with_ignore_extra_2(x, ignore_extra=False):
        return x
    
    test_field = _PField(
        type={CheckedPMap},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=factory_with_ignore_extra_2,
        serializer=PFIELD_NO_SERIALIZER
    )
    # When field.type is a set, is_type_cls returns True immediately
    assert is_field_ignore_extra_complaint(CheckedPMap, test_field, True) is True
    
    # Test case 6: ignore_extra is True with CheckedPVector type
    def factory_with_ignore_extra_3(x, ignore_extra=False):
        return x
    
    test_field = _PField(
        type={CheckedPVector},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=factory_with_ignore_extra_3,
        serializer=PFIELD_NO_SERIALIZER
    )
    assert is_field_ignore_extra_complaint(CheckedPVector, test_field, True) is True


# LLM-generated content at query #14
#--------------------------

```python
def test_field():
    # Test basic field creation with no parameters
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory is False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with single type
    f = field(type=int)
    assert int in f.type
    assert len(f.type) == 1

    # Test field with multiple types as list
    f = field(type=[int, str])
    assert int in f.type
    assert str in f.type
    assert len(f.type) == 2

    # Test field with multiple types as tuple
    f = field(type=(int, str))
    assert int in f.type
    assert str in f.type
    assert len(f.type) == 2

    # Test field with multiple types as set
    f = field(type={int, str})
    assert int in f.type
    assert str in f.type
    assert len(f.type) == 2

    # Test field with mandatory flag
    f = field(type=int, mandatory=True)
    assert f.mandatory is True

    # Test field with initial value
    f = field(type=int, initial=42)
    assert f.initial == 42

    # Test field with custom factory
    custom_factory = lambda x: x * 2
    f = field(type=int, factory=custom_factory)
    assert f.factory == custom_factory

    # Test field with custom serializer
    custom_serializer = lambda fmt, val: str(val)
    f = field(type=int, serializer=custom_serializer)
    assert f.serializer == custom_serializer

    # Test field with custom invariant
    custom_invariant = lambda x: (x > 0, "Must be positive")
    f = field(type=int, invariant=custom_invariant)
    assert f.invariant is not PFIELD_NO_INVARIANT
    assert callable(f.invariant)

    # Test field with callable initial value
    f = field(type=int, initial=lambda: 10)
    assert callable(f.initial)

    # Test field type validation - invalid type parameter
    with pytest.raises(TypeError, match="Type parameter expected"):
        field(type=[123])

    # Test field with invalid initial type
    with pytest.raises(TypeError, match="Initial has invalid type"):
        field(type=int, initial="not an int")

    # Test field with non-callable invariant
    with pytest.raises(TypeError, match="Invariant must be callable"):
        field(invariant="not callable")

    # Test field with non-callable factory
    with pytest.raises(TypeError, match="Factory must be callable"):
        field(factory="not callable")

    # Test field with non-callable serializer
    with pytest.raises(TypeError, match="Serializer must be callable"):
        field(serializer="not callable")

    # Test field returns _PField instance
    f = field(type=str, mandatory=True)
    assert isinstance(f, _PField)

    # Test field with string type annotation
    f = field(type="SomeType")
    assert len(f.type) > 0

    # Test field with initial value matching type
    f = field(type=str, initial="hello")
    assert f.initial == "hello"

    # Test field with multiple types and initial value matching one of them
    f = field(type=[int, str], initial=42)
    assert f.initial == 42

    # Test field factory property returns PFIELD_NO_FACTORY when no factory specified
    f = field(type=int)
    assert f.factory == PFIELD_NO_FACTORY

    # Test field with empty type set
    f = field(type=set())
    assert f.type == set()


# LLM-generated content at query #15
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj is not None
    assert field_obj.mandatory is True
    assert callable(field_obj.factory)
    
    # Test that the field type is set correctly
    assert len(field_obj.type) > 0
    
    # Test pmap_field with optional=True
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.mandatory is True
    assert callable(optional_field.factory)
    
    # Test factory creates CheckedPMap with valid data
    field_obj = pmap_field(str, int)
    result = field_obj.factory({})
    assert isinstance(result, CheckedPMap)
    
    # Test factory with non-empty dict
    result = field_obj.factory({"key1": 1, "key2": 2})
    assert isinstance(result, CheckedPMap)
    assert result["key1"] == 1
    assert result["key2"] == 2
    
    # Test optional field factory with None
    optional_field = pmap_field(str, int, optional=True)
    result = optional_field.factory(None)
    assert result is None
    
    # Test optional field factory with valid data
    result = optional_field.factory({"a": 1})
    assert isinstance(result, CheckedPMap)
    assert result["a"] == 1
    
    # Test initial value is an empty CheckedPMap
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj.initial, CheckedPMap)
    assert len(field_obj.initial) == 0
    
    # Test with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) <= 10, "Map too large")
    
    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant is not None
    assert callable(field_obj.invariant)
    
    # Test that different key/value type combinations create different types
    field1 = pmap_field(str, int)
    field2 = pmap_field(int, str)
    assert field1.type != field2.type
    
    # Test caching of pmap field types
    field1 = pmap_field(str, int)
    field2 = pmap_field(str, int)
    assert field1.type == field2.type
    
    # Test factory is callable and returns CheckedPMap
    field_obj = pmap_field(str, float)
    test_data = {"x": 1.5, "y": 2.5}
    result = field_obj.factory(test_data)
    assert isinstance(result, CheckedPMap)
    assert result["x"] == 1.5


# LLM-generated content at query #16
#--------------------------

def test_check_global_invariants():
    # Test case 1: All invariants pass
    invariants = [
        lambda subject: (True, None),
        lambda subject: (True, None),
    ]
    # Should not raise
    check_global_invariants("test_subject", invariants)

    # Test case 2: Single invariant fails
    invariants = [
        lambda subject: (False, "error_1"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", invariants)
    assert exc_info.value.error_codes == ("error_1",)
    assert "Global invariant failed" in str(exc_info.value)

    # Test case 3: Multiple invariants, some fail
    invariants = [
        lambda subject: (True, None),
        lambda subject: (False, "error_2"),
        lambda subject: (False, "error_3"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", invariants)
    assert set(exc_info.value.error_codes) == {"error_2", "error_3"}
    assert "Global invariant failed" in str(exc_info.value)

    # Test case 4: Empty invariants list
    invariants = []
    # Should not raise
    check_global_invariants("test_subject", invariants)

    # Test case 5: Invariants with different subject types
    invariants = [
        lambda subject: (subject > 5, "value_too_small") if isinstance(subject, int) else (True, None),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(3, invariants)
    assert exc_info.value.error_codes == ("value_too_small",)

    # Test case 6: Invariants pass for different subject
    invariants = [
        lambda subject: (subject > 5, "value_too_small") if isinstance(subject, int) else (True, None),
    ]
    # Should not raise
    check_global_invariants(10, invariants)

    # Test case 7: Complex subject object
    class MockSubject:
        def __init__(self, value):
            self.value = value

    subject = MockSubject(42)
    invariants = [
        lambda s: (s.value > 0, "negative_value"),
        lambda s: (s.value < 100, "value_too_large"),
    ]
    # Should not raise
    check_global_invariants(subject, invariants)

    # Test case 8: Complex subject fails invariant
    subject = MockSubject(-5)
    invariants = [
        lambda s: (s.value > 0, "negative_value"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("negative_value",)

    # Test case 9: Multiple failures with various error codes
    invariants = [
        lambda subject: (False, "error_a"),
        lambda subject: (False, "error_b"),
        lambda subject: (False, "error_c"),
        lambda subject: (True, None),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("subject", invariants)
    assert set(exc_info.value.error_codes) == {"error_a", "error_b", "error_c"}


# LLM-generated content at query #17
#--------------------------

def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.mandatory is True
    assert field_obj.factory is not None
    assert callable(field_obj.factory)
    
    # Test that the field type is a CheckedPMap subclass
    assert isinstance(field_obj.type, set)
    type_obj = list(field_obj.type)[0]
    assert issubclass(type_obj, CheckedPMap)
    
    # Test with optional=True
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.mandatory is True
    
    # Test that optional factory handles None
    optional_factory = optional_field.factory
    result = optional_factory(None)
    assert result is None
    
    # Test that optional factory handles dict input
    result = optional_factory({'key': 1})
    assert result is not None
    
    # Test non-optional factory with dict input
    non_optional_field = pmap_field(str, int, optional=False)
    result = non_optional_field.factory({'key': 2})
    assert result is not None
    
    # Test initial value is an empty CheckedPMap
    assert non_optional_field.initial is not None
    
    # Test with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) <= 10, "Map too large")
    
    field_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_invariant.invariant is not None
    
    # Test that different key/value types create different cached types
    field1 = pmap_field(str, int)
    field2 = pmap_field(str, str)
    type1 = list(field1.type)[0]
    type2 = list(field2.type)[0]
    assert type1 is not type2
    
    # Test that same key/value types return cached type
    field3 = pmap_field(str, int)
    type3 = list(field3.type)[0]
    assert type1 is type3
    
    # Test optional field type includes optional wrapper
    optional_field = pmap_field(int, str, optional=True)
    assert len(optional_field.type) == 1 or len(optional_field.type) == 2


# LLM-generated content at query #18
#--------------------------

```python
def test_is_type_cls():
    from pyrsistent import PVector, PSet, PMap
    
    # Test with set type_cls - should always return True
    assert is_type_cls(CheckedPVector, {int, str}) is True
    assert is_type_cls(CheckedPSet, {int}) is True
    assert is_type_cls(CheckedPMap, set()) is True
    
    # Test with tuple of single matching type
    assert is_type_cls(CheckedPVector, (CheckedPVector,)) is True
    assert is_type_cls(CheckedPSet, (CheckedPSet,)) is True
    
    # Test with tuple of single non-matching type
    assert is_type_cls(CheckedPVector, (CheckedPSet,)) is False
    assert is_type_cls(CheckedPSet, (CheckedPVector,)) is False
    
    # Test with empty tuple - should return False
    assert is_type_cls(CheckedPVector, ()) is False
    assert is_type_cls(CheckedPSet, ()) is False
    
    # Test with list of types
    assert is_type_cls(CheckedPVector, [CheckedPVector, int]) is True
    assert is_type_cls(CheckedPSet, [CheckedPSet, str]) is True
    assert is_type_cls(CheckedPVector, [CheckedPSet, int]) is False
    
    # Test with string type (forward reference)
    assert is_type_cls(CheckedPVector, ("CheckedPVector",)) is False
    
    # Test with subclass of CheckedPVector
    class MyVector(CheckedPVector):
        __type__ = int
    
    assert is_type_cls(CheckedPVector, (MyVector,)) is True
    
    # Test with subclass of CheckedPSet
    class MySet(CheckedPSet):
        __type__ = str
    
    assert is_type_cls(CheckedPSet, (MySet,)) is True
    assert is_type_cls(CheckedPVector, (MySet,)) is False


# LLM-generated content at query #19
#--------------------------

```python
def test_field():
    # Test basic field creation with no parameters
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory == False
    assert f._factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with single type
    f = field(type=int)
    assert int in f.type
    assert len(f.type) == 1

    # Test field with multiple types as list
    f = field(type=[int, str])
    assert int in f.type
    assert str in f.type
    assert len(f.type) == 2

    # Test field with multiple types as tuple
    f = field(type=(int, str, float))
    assert int in f.type
    assert str in f.type
    assert float in f.type
    assert len(f.type) == 3

    # Test field with multiple types as set
    f = field(type={int, str})
    assert int in f.type
    assert str in f.type

    # Test field with initial value
    f = field(type=int, initial=42)
    assert f.initial == 42

    # Test field with mandatory flag
    f = field(mandatory=True)
    assert f.mandatory == True

    # Test field with custom factory
    custom_factory = lambda x: x * 2
    f = field(factory=custom_factory)
    assert f._factory == custom_factory

    # Test field with custom serializer
    custom_serializer = lambda fmt, val: str(val)
    f = field(serializer=custom_serializer)
    assert f.serializer == custom_serializer

    # Test field with custom invariant
    custom_invariant = lambda x: (x > 0, "must be positive")
    f = field(invariant=custom_invariant)
    assert f.invariant is not None

    # Test that invariant is wrapped when callable
    f = field(invariant=custom_invariant)
    assert callable(f.invariant)

    # Test field with all parameters
    f = field(
        type=[int, str],
        invariant=lambda x: (True, ""),
        initial=10,
        mandatory=True,
        factory=lambda x: int(x),
        serializer=lambda fmt, val: str(val)
    )
    assert int in f.type
    assert str in f.type
    assert f.initial == 10
    assert f.mandatory == True
    assert callable(f.invariant)

    # Test that invalid type parameter raises TypeError
    with pytest.raises(TypeError):
        field(type=[123, "invalid"])

    # Test that invalid initial type raises TypeError
    with pytest.raises(TypeError):
        field(type=int, initial="not an int")

    # Test that non-callable invariant raises TypeError
    with pytest.raises(TypeError):
        field(invariant="not callable")

    # Test that non-callable factory raises TypeError
    with pytest.raises(TypeError):
        field(factory="not callable")

    # Test that non-callable serializer raises TypeError
    with pytest.raises(TypeError):
        field(serializer="not callable")

    # Test field with callable initial value
    f = field(type=int, initial=lambda: 42)
    assert callable(f.initial)

    # Test field with string type (forward reference)
    f = field(type="SomeClass")
    assert "SomeClass" in f.type

    # Test empty type list
    f = field(type=[])
    assert f.type == set()

    # Test field factory property returns PFIELD_NO_FACTORY by default
    f = field()
    assert f.factory == PFIELD_NO_FACTORY

    # Test field with no invariant uses default
    f = field(invariant=PFIELD_NO_INVARIANT)
    assert f.invariant == PFIELD_NO_INVARIANT


# LLM-generated content at query #20
#--------------------------

```python
def test_is_type_cls():
    # Test with set type
    assert is_type_cls(CheckedPVector, {CheckedPVector}) is True
    assert is_type_cls(CheckedPSet, {CheckedPSet}) is True
    assert is_type_cls(CheckedPMap, {CheckedPMap}) is True
    
    # Test with tuple type
    assert is_type_cls(CheckedPVector, (CheckedPVector,)) is True
    assert is_type_cls(CheckedPSet, (CheckedPSet,)) is True
    assert is_type_cls(CheckedPMap, (CheckedPMap,)) is True
    
    # Test with list type
    assert is_type_cls(CheckedPVector, [CheckedPVector]) is True
    assert is_type_cls(CheckedPSet, [CheckedPSet]) is True
    
    # Test with empty tuple/list
    assert is_type_cls(CheckedPVector, ()) is False
    assert is_type_cls(CheckedPSet, []) is False
    
    # Test with non-matching type
    assert is_type_cls(CheckedPVector, (CheckedPSet,)) is False
    assert is_type_cls(CheckedPSet, (CheckedPVector,)) is False
    assert is_type_cls(CheckedPMap, (CheckedPVector,)) is False
    
    # Test with multiple types where first matches
    assert is_type_cls(CheckedPVector, (CheckedPVector, int)) is True
    
    # Test with multiple types where first doesn't match
    assert is_type_cls(CheckedPVector, (int, CheckedPVector)) is False
    
    # Test with string type reference
    assert is_type_cls(CheckedPVector, ('SomeType',)) is False
    
    # Test with built-in types
    assert is_type_cls(CheckedPVector, (int,)) is False
    assert is_type_cls(CheckedPVector, (str,)) is False


# LLM-generated content at query #21
#--------------------------

```python
import pytest


def test_check_type():
    """Test check_type function for type validation."""
    
    # Test 1: Valid type - should not raise
    class TestClass:
        pass
    
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", 42)  # Should pass
    
    # Test 2: Invalid type - should raise PTypeError
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, "test_field", "string")
    
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == {int}
    assert exc_info.value.actual_type == str
    
    # Test 3: Multiple allowed types - valid
    field_obj = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", "string")  # Should pass
    check_type(TestClass, field_obj, "test_field", 42)  # Should pass
    
    # Test 4: Multiple allowed types - invalid
    field_obj = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, "test_field", 3.14)
    
    assert exc_info.value.actual_type == float
    
    # Test 5: No type constraint - should not raise
    field_obj = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", "anything")  # Should pass
    check_type(TestClass, field_obj, "test_field", 42)  # Should pass
    check_type(TestClass, field_obj, "test_field", None)  # Should pass
    
    # Test 6: None value with int type - should raise
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, "test_field", None)
    
    assert exc_info.value.actual_type == type(None)
    
    # Test 7: Subclass should be valid
    class Parent:
        pass
    
    class Child(Parent):
        pass
    
    field_obj = _PField(type={Parent}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", Child())  # Should pass


# LLM-generated content at query #22
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    invariants = [
        lambda subject: (True, None),
        lambda subject: (True, None),
    ]
    # Should not raise any exception
    check_global_invariants("test_subject", invariants)

    # Test case 2: Single invariant fails
    invariants = [
        lambda subject: (False, "error_1"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", invariants)
    assert exc_info.value.error_codes == ("error_1",)

    # Test case 3: Multiple invariants, one fails
    invariants = [
        lambda subject: (True, None),
        lambda subject: (False, "error_2"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", invariants)
    assert exc_info.value.error_codes == ("error_2",)

    # Test case 4: Multiple invariants fail
    invariants = [
        lambda subject: (False, "error_1"),
        lambda subject: (False, "error_2"),
        lambda subject: (True, None),
        lambda subject: (False, "error_3"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", invariants)
    assert exc_info.value.error_codes == ("error_1", "error_2", "error_3")

    # Test case 5: Empty invariants list
    invariants = []
    # Should not raise any exception
    check_global_invariants("test_subject", invariants)

    # Test case 6: Invariants with different subject types
    invariants = [
        lambda subject: (subject == 42, "not_42"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(100, invariants)
    assert exc_info.value.error_codes == ("not_42",)

    # Test case 7: Invariant passes with correct subject
    invariants = [
        lambda subject: (subject == 42, "not_42"),
    ]
    check_global_invariants(42, invariants)

    # Test case 8: InvariantException has correct structure
    invariants = [
        lambda subject: (False, "test_error"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("subject", invariants)
    exception = exc_info.value
    assert exception.error_codes == ("test_error",)
    assert exception.invariant_errors == ()


# LLM-generated content at query #23
#--------------------------

```python
def test_check_type():
    """Test check_type function for type validation."""
    
    # Test 1: Valid type - should not raise
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(object, field_obj, 'test_field', 42)  # Should pass
    
    # Test 2: Invalid type - should raise PTypeError
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(object, field_obj, 'test_field', "string")
    assert exc_info.value.field == 'test_field'
    assert exc_info.value.expected_types == {int}
    assert exc_info.value.actual_type == str
    
    # Test 3: Multiple allowed types - valid
    field_obj = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(object, field_obj, 'test_field', "string")  # Should pass
    check_type(object, field_obj, 'test_field', 42)  # Should pass
    
    # Test 4: Multiple allowed types - invalid
    field_obj = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError):
        check_type(object, field_obj, 'test_field', [])
    
    # Test 5: No type constraint - should not raise
    field_obj = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(object, field_obj, 'test_field', "anything")  # Should pass
    
    # Test 6: Error message format
    class TestClass:
        pass
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, 'my_field', 3.14)
    assert 'TestClass' in str(exc_info.value)
    assert 'my_field' in str(exc_info.value)
    assert 'float' in str(exc_info.value)
    
    # Test 7: Subclass instances should be accepted
    class Parent:
        pass
    class Child(Parent):
        pass
    field_obj = _PField(type={Parent}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(object, field_obj, 'test_field', Child())  # Should pass


# LLM-generated content at query #24
#--------------------------

```python
def test_pmap_field():
    """Test pmap_field function creates proper checked PMap fields."""
    from pyrsistent import pmap
    
    # Test basic pmap_field creation
    int_to_str_field = pmap_field(int, str)
    assert int_to_str_field is not None
    assert int_to_str_field.mandatory is True
    assert callable(int_to_str_field.factory)
    
    # Test that factory creates empty pmap by default
    empty_map = int_to_str_field.initial
    assert len(empty_map) == 0
    
    # Test that factory accepts and converts data
    test_data = {1: "one", 2: "two"}
    result = int_to_str_field.factory(test_data)
    assert result[1] == "one"
    assert result[2] == "two"
    
    # Test optional pmap_field with None
    optional_field = pmap_field(int, str, optional=True)
    assert optional_field.factory(None) is None
    assert optional_field.factory({1: "one"})[1] == "one"
    
    # Test non-optional pmap_field rejects None
    non_optional_field = pmap_field(int, str, optional=False)
    result = non_optional_field.factory({1: "one"})
    assert result[1] == "one"
    
    # Test with custom invariant
    def custom_invariant(pmap_val):
        return (len(pmap_val) > 0, "PMap must not be empty")
    
    field_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_invariant.invariant is not None
    assert callable(field_with_invariant.invariant)
    
    # Test type is set correctly
    int_to_str_field = pmap_field(int, str)
    assert int_to_str_field.type is not None
    
    # Test optional field type includes optional wrapper
    optional_int_to_str = pmap_field(int, str, optional=True)
    assert optional_int_to_str.type is not None
    
    # Test caching of generated types
    field1 = pmap_field(int, str)
    field2 = pmap_field(int, str)
    # Both should create from the same cached type
    assert type(field1.factory({})).__name__ == type(field2.factory({})).__name__
    
    # Test with different key/value types
    str_to_float_field = pmap_field(str, float)
    result = str_to_float_field.factory({"pi": 3.14, "e": 2.71})
    assert result["pi"] == 3.14
    assert result["e"] == 2.71


# LLM-generated content at query #25
#--------------------------

```python
def test_check_type():
    # Test 1: Valid type - should not raise
    class TestClass:
        pass
    
    test_field = _PField(
        type={int},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    check_type(TestClass, test_field, "test_field", 42)
    
    # Test 2: Invalid type - should raise PTypeError
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, test_field, "test_field", "invalid")
    
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == {int}
    assert exc_info.value.actual_type == str
    assert "Invalid type for field" in str(exc_info.value)
    
    # Test 3: Multiple allowed types - valid
    test_field_multi = _PField(
        type={int, str},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    check_type(TestClass, test_field_multi, "multi_field", 42)
    check_type(TestClass, test_field_multi, "multi_field", "valid")
    
    # Test 4: Multiple allowed types - invalid
    with pytest.raises(PTypeError):
        check_type(TestClass, test_field_multi, "multi_field", [1, 2, 3])
    
    # Test 5: No type constraint - should not raise
    test_field_no_type = _PField(
        type=PFIELD_NO_TYPE,
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    check_type(TestClass, test_field_no_type, "no_type_field", "anything")
    check_type(TestClass, test_field_no_type, "no_type_field", 123)
    check_type(TestClass, test_field_no_type, "no_type_field", None)
    
    # Test 6: Subclass should be valid
    class Parent:
        pass
    
    class Child(Parent):
        pass
    
    test_field_parent = _PField(
        type={Parent},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    check_type(TestClass, test_field_parent, "parent_field", Child())


# LLM-generated content at query #26
#--------------------------

```python
def test_check_type():
    """Unit tests for check_type function."""
    
    # Test 1: Valid type - should not raise
    class TestClass:
        pass
    
    test_field = _PField(
        type={int},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    # Should not raise for valid type
    check_type(TestClass, test_field, 'test_field', 42)
    
    # Test 2: Invalid type - should raise PTypeError
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, test_field, 'test_field', "invalid_string")
    
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == 'test_field'
    assert exc_info.value.expected_types == {int}
    assert exc_info.value.actual_type == str
    assert "Invalid type for field TestClass.test_field" in str(exc_info.value)
    
    # Test 3: Multiple valid types
    test_field_multi = _PField(
        type={int, str},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    # Should not raise for either valid type
    check_type(TestClass, test_field_multi, 'multi_field', 42)
    check_type(TestClass, test_field_multi, 'multi_field', "string")
    
    # Test 4: Invalid with multiple allowed types
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, test_field_multi, 'multi_field', 3.14)
    
    assert exc_info.value.actual_type == float
    assert exc_info.value.expected_types == {int, str}
    
    # Test 5: No type constraint - should not raise
    test_field_no_type = _PField(
        type=PFIELD_NO_TYPE,
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    check_type(TestClass, test_field_no_type, 'any_field', "anything")
    check_type(TestClass, test_field_no_type, 'any_field', 12345)
    check_type(TestClass, test_field_no_type, 'any_field', None)
    
    # Test 6: Subclass should be valid
    class Parent:
        pass
    
    class Child(Parent):
        pass
    
    test_field_parent = _PField(
        type={Parent},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    check_type(TestClass, test_field_parent, 'parent_field', Child())


# LLM-generated content at query #27
#--------------------------

```python
def test_check_type():
    # Test with no type constraint - should pass
    class MockField:
        type = ()
    
    field_obj = MockField()
    check_type(str, field_obj, "test_field", "any_value")
    
    # Test with matching type - should pass
    class MockField2:
        type = (int,)
    
    field_obj2 = MockField2()
    check_type(str, field_obj2, "test_field", 42)
    
    # Test with non-matching type - should raise PTypeError
    class MockField3:
        type = (int,)
    
    field_obj3 = MockField3()
    with pytest.raises(PTypeError) as exc_info:
        check_type(str, field_obj3, "test_field", "not_an_int")
    
    assert exc_info.value.source_class == str
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == (int,)
    assert exc_info.value.actual_type == str
    assert "Invalid type for field" in str(exc_info.value)
    
    # Test with multiple allowed types - should pass
    class MockField4:
        type = (int, str)
    
    field_obj4 = MockField4()
    check_type(list, field_obj4, "multi_field", "string_value")
    check_type(list, field_obj4, "multi_field", 100)
    
    # Test with multiple allowed types - should fail
    class MockField5:
        type = (int, str)
    
    field_obj5 = MockField5()
    with pytest.raises(PTypeError) as exc_info:
        check_type(list, field_obj5, "multi_field", [1, 2, 3])
    
    assert exc_info.value.actual_type == list
    assert exc_info.value.expected_types == (int, str)
    
    # Test with None value and no type constraint
    class MockField6:
        type = ()
    
    field_obj6 = MockField6()
    check_type(dict, field_obj6, "nullable_field", None)
    
    # Test with custom class type
    class CustomClass:
        pass
    
    class MockField7:
        type = (CustomClass,)
    
    field_obj7 = MockField7()
    obj = CustomClass()
    check_type(str, field_obj7, "custom_field", obj)
    
    # Test error message format
    class MockField8:
        type = (float,)
    
    field_obj8 = MockField8()
    with pytest.raises(PTypeError) as exc_info:
        check_type(MyClass, field_obj8, "my_field", True)
    
    error_msg = str(exc_info.value)
    assert "MyClass" in error_msg
    assert "my_field" in error_msg
    assert "bool" in error_msg


# LLM-generated content at query #28
#--------------------------

```python
def test_check_type():
    # Test case 1: Valid type - should not raise
    class TestClass:
        __name__ = "TestClass"
    
    class TestField:
        type = (int,)
    
    field_obj = TestField()
    try:
        check_type(TestClass, field_obj, "test_field", 42)
    except PTypeError:
        pytest.fail("check_type raised PTypeError unexpectedly for valid type")
    
    # Test case 2: Invalid type - should raise PTypeError
    field_obj = TestField()
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, "test_field", "invalid")
    
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == (int,)
    assert exc_info.value.actual_type == str
    assert "Invalid type for field TestClass.test_field" in str(exc_info.value)
    
    # Test case 3: Multiple allowed types - valid
    class MultiTypeField:
        type = (int, str)
    
    field_obj = MultiTypeField()
    try:
        check_type(TestClass, field_obj, "multi_field", "string_value")
    except PTypeError:
        pytest.fail("check_type raised PTypeError unexpectedly for valid type in multi-type field")
    
    try:
        check_type(TestClass, field_obj, "multi_field", 123)
    except PTypeError:
        pytest.fail("check_type raised PTypeError unexpectedly for valid type in multi-type field")
    
    # Test case 4: Multiple allowed types - invalid
    field_obj = MultiTypeField()
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, "multi_field", [])
    
    assert exc_info.value.actual_type == list
    
    # Test case 5: No type constraint - should not raise
    class NoTypeField:
        type = ()
    
    field_obj = NoTypeField()
    try:
        check_type(TestClass, field_obj, "no_type_field", "anything")
        check_type(TestClass, field_obj, "no_type_field", 42)
        check_type(TestClass, field_obj, "no_type_field", [])
    except PTypeError:
        pytest.fail("check_type raised PTypeError for field with no type constraint")
    
    # Test case 6: None type should fail when not allowed
    field_obj = TestField()
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, "test_field", None)
    
    assert exc_info.value.actual_type == type(None)


# LLM-generated content at query #29
#--------------------------

```python
def test_pmap_field():
    """Test pmap_field function for creating checked PMap fields."""
    
    # Test basic pmap_field creation
    pmap_fld = pmap_field(str, int)
    assert pmap_fld.mandatory is True
    assert pmap_fld.factory is not None
    assert callable(pmap_fld.factory)
    
    # Test that the field type is a CheckedPMap subclass
    assert len(pmap_fld.type) == 1
    field_type = get_type(tuple(pmap_fld.type)[0])
    assert issubclass(field_type, CheckedPMap)
    
    # Test optional pmap_field
    optional_pmap_fld = pmap_field(str, int, optional=True)
    assert optional_pmap_fld.mandatory is True
    assert len(optional_pmap_fld.type) == 2  # Should include None type
    
    # Test factory with non-optional field
    pmap_fld = pmap_field(str, int)
    test_data = {'key1': 1, 'key2': 2}
    result = pmap_fld.factory(test_data)
    assert result is not None
    assert len(result) == 2
    
    # Test factory with optional field and None value
    optional_pmap_fld = pmap_field(str, int, optional=True)
    result = optional_pmap_fld.factory(None)
    assert result is None
    
    # Test factory with optional field and actual data
    result = optional_pmap_fld.factory(test_data)
    assert result is not None
    assert len(result) == 2
    
    # Test initial value
    pmap_fld = pmap_field(str, int)
    assert pmap_fld.initial is not None
    assert len(pmap_fld.initial) == 0
    
    # Test with custom invariant
    custom_invariant = lambda x: (True, "")
    pmap_fld = pmap_field(str, int, invariant=custom_invariant)
    assert pmap_fld.invariant is not None
    
    # Test that different key/value type combinations create different types
    pmap_fld1 = pmap_field(str, int)
    pmap_fld2 = pmap_field(str, float)
    pmap_fld3 = pmap_field(int, str)
    
    type1 = get_type(tuple(pmap_fld1.type)[0])
    type2 = get_type(tuple(pmap_fld2.type)[0])
    type3 = get_type(tuple(pmap_fld3.type)[0])
    
    assert type1 is not type2
    assert type2 is not type3
    assert type1 is not type3
    
    # Test caching of pmap field types
    pmap_fld_a = pmap_field(str, int)
    pmap_fld_b = pmap_field(str, int)
    type_a = get_type(tuple(pmap_fld_a.type)[0])
    type_b = get_type(tuple(pmap_fld_b.type)[0])
    assert type_a is type_b


# LLM-generated content at query #30
#--------------------------

```python
def test_serialize():
    # Test 1: serialize with CheckedType and PFIELD_NO_SERIALIZER
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return "serialized_value"
    
    mock_obj = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_obj)
    assert result == "serialized_value"
    
    # Test 2: serialize with custom serializer and CheckedType
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    class AnotherCheckedType(CheckedType):
        def serialize(self, format):
            return "should_not_use_this"
    
    obj = AnotherCheckedType()
    result = serialize(custom_serializer, "xml", obj)
    assert result == "custom_xml_" + str(obj)
    
    # Test 3: serialize with PFIELD_NO_SERIALIZER and non-CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "json", "plain_string")
    assert result == "plain_string"
    
    # Test 4: serialize with custom serializer and non-CheckedType
    def another_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(another_serializer, "csv", 42)
    assert result == "csv:42"
    
    # Test 5: serialize with default PFIELD_NO_SERIALIZER and dict
    result = serialize(PFIELD_NO_SERIALIZER, "json", {"key": "value"})
    assert result == {"key": "value"}
    
    # Test 6: serialize with custom serializer and None value
    def none_serializer(format, value):
        return "none_handled"
    
    result = serialize(none_serializer, "json", None)
    assert result == "none_handled"
    
    # Test 7: serialize with PFIELD_NO_SERIALIZER and list
    test_list = [1, 2, 3]
    result = serialize(PFIELD_NO_SERIALIZER, "json", test_list)
    assert result == test_list
    
    # Test 8: serialize with custom serializer that uses format
    def format_aware_serializer(format, value):
        if format == "json":
            return f"json_{value}"
        elif format == "xml":
            return f"xml_{value}"
        return value
    
    result = serialize(format_aware_serializer, "json", "test")
    assert result == "json_test"
    result = serialize(format_aware_serializer, "xml", "test")
    assert result == "xml_test"


# LLM-generated content at query #31
#--------------------------

```python
def test_check_global_invariants():
    # Test 1: All invariants pass
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None),
    ]
    subject = "test_subject"
    # Should not raise
    check_global_invariants(subject, invariants)

    # Test 2: Single invariant fails
    invariants = [
        lambda x: (False, "error_code_1"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("error_code_1",)

    # Test 3: Multiple invariants fail
    invariants = [
        lambda x: (False, "error_code_1"),
        lambda x: (False, "error_code_2"),
        lambda x: (True, None),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert set(exc_info.value.error_codes) == {"error_code_1", "error_code_2"}

    # Test 4: Empty invariants list
    invariants = []
    # Should not raise
    check_global_invariants(subject, invariants)

    # Test 5: Mixed pass/fail with multiple failures
    invariants = [
        lambda x: (True, "ok_1"),
        lambda x: (False, "error_1"),
        lambda x: (False, "error_2"),
        lambda x: (True, "ok_2"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert set(exc_info.value.error_codes) == {"error_1", "error_2"}

    # Test 6: Verify exception has correct attributes
    invariants = [
        lambda x: (False, "test_error"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("test_error",)
    assert exc_info.value.message == "Global invariant failed"


# LLM-generated content at query #32
#--------------------------

```python
def test_check_type():
    # Test case 1: Valid type - should not raise
    class TestClass:
        pass
    
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", 42)
    
    # Test case 2: Valid type with multiple allowed types
    field_obj = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", "hello")
    check_type(TestClass, field_obj, "test_field", 123)
    
    # Test case 3: Invalid type - should raise PTypeError
    field_obj = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field_obj, "test_field", "not an int")
    
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == {int}
    assert exc_info.value.actual_type == str
    assert "Invalid type for field TestClass.test_field" in str(exc_info.value)
    
    # Test case 4: No type constraint - should not raise
    field_obj = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", "any value")
    check_type(TestClass, field_obj, "test_field", 123)
    check_type(TestClass, field_obj, "test_field", None)
    
    # Test case 5: None value with type constraint - should raise
    field_obj = _PField(type={str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError):
        check_type(TestClass, field_obj, "test_field", None)
    
    # Test case 6: Subclass should be accepted
    class Parent:
        pass
    
    class Child(Parent):
        pass
    
    field_obj = _PField(type={Parent}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                        mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field_obj, "test_field", Child())


# LLM-generated content at query #33
#--------------------------

def test_check_global_invariants():
    # Test case 1: All invariants pass
    def invariant_pass(subject):
        return (True, None)
    
    subject = object()
    check_global_invariants(subject, [invariant_pass])
    # Should not raise
    
    # Test case 2: Single failing invariant
    def invariant_fail(subject):
        return (False, "error_code_1")
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant_fail])
    
    assert exc_info.value.error_codes == ("error_code_1",)
    assert exc_info.value.message == 'Global invariant failed'
    
    # Test case 3: Multiple invariants, some passing, some failing
    def invariant_pass_2(subject):
        return (True, None)
    
    def invariant_fail_2(subject):
        return (False, "error_code_2")
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant_pass, invariant_fail, invariant_pass_2, invariant_fail_2])
    
    assert set(exc_info.value.error_codes) == {"error_code_1", "error_code_2"}
    assert exc_info.value.message == 'Global invariant failed'
    
    # Test case 4: Multiple failing invariants
    def invariant_fail_3(subject):
        return (False, "error_code_3")
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant_fail, invariant_fail_2, invariant_fail_3])
    
    assert set(exc_info.value.error_codes) == {"error_code_1", "error_code_2", "error_code_3"}
    
    # Test case 5: Empty invariants list
    check_global_invariants(subject, [])
    # Should not raise
    
    # Test case 6: All invariants pass with multiple invariants
    check_global_invariants(subject, [invariant_pass, invariant_pass_2])
    # Should not raise


# LLM-generated content at query #34
#--------------------------

```python
def test_check_type():
    """Test check_type function for type validation."""
    
    # Test 1: Valid type - should not raise
    class TestClass:
        pass
    
    field_obj = _PField(
        type={int},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    # Should not raise for valid type
    check_type(TestClass, field_obj, 'test_field', 42)
    
    # Test 2: Invalid type - should raise PTypeError
    field_obj_str = _PField(
        type={str},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    try:
        check_type(TestClass, field_obj_str, 'test_field', 42)
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.source_class == TestClass
        assert e.field == 'test_field'
        assert e.expected_types == {str}
        assert e.actual_type == int
    
    # Test 3: Multiple valid types
    field_obj_multi = _PField(
        type={int, str},
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    check_type(TestClass, field_obj_multi, 'test_field', 42)
    check_type(TestClass, field_obj_multi, 'test_field', "hello")
    
    # Test 4: Multiple types with invalid value
    try:
        check_type(TestClass, field_obj_multi, 'test_field', 3.14)
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.actual_type == float
    
    # Test 5: Field with no type constraint - should not raise
    field_obj_no_type = _PField(
        type=PFIELD_NO_TYPE,
        invariant=PFIELD_NO_INVARIANT,
        initial=PFIELD_NO_INITIAL,
        mandatory=False,
        factory=PFIELD_NO_FACTORY,
        serializer=PFIELD_NO_SERIALIZER
    )
    
    check_type(TestClass, field_obj_no_type, 'test_field', "any value")
    check_type(TestClass, field_obj_no_type, 'test_field', 42)
    check_type(TestClass, field_obj_no_type, 'test_field', None)
    
    # Test 6: Check error message format
    try:
        check_type(TestClass, field_obj_str, 'my_field', 123)
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert "Invalid type for field" in str(e)
        assert "TestClass.my_field" in str(e)
        assert "int" in str(e)


# LLM-generated content at query #35
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test case 1: ignore_extra is False, should return False
    test_field = _PField(type={CheckedPVector}, invariant=PFIELD_NO_INVARIANT, 
                         initial=PFIELD_NO_INITIAL, mandatory=False, 
                         factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, False)
    assert result is False

    # Test case 2: ignore_extra is True, but field type is not CheckedPVector/CheckedPSet
    test_field = _PField(type={str}, invariant=PFIELD_NO_INVARIANT, 
                         initial=PFIELD_NO_INITIAL, mandatory=False, 
                         factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is False

    # Test case 3: ignore_extra is True, field type is CheckedPVector, but factory doesn't have ignore_extra parameter
    test_field = _PField(type={CheckedPVector}, invariant=PFIELD_NO_INVARIANT, 
                         initial=PFIELD_NO_INITIAL, mandatory=False, 
                         factory=lambda x: x, serializer=PFIELD_NO_SERIALIZER)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is False

    # Test case 4: ignore_extra is True, field type is CheckedPVector, factory has ignore_extra parameter
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    
    test_field = _PField(type={CheckedPVector}, invariant=PFIELD_NO_INVARIANT, 
                         initial=PFIELD_NO_INITIAL, mandatory=False, 
                         factory=factory_with_ignore_extra, serializer=PFIELD_NO_SERIALIZER)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is True

    # Test case 5: ignore_extra is True, field type is CheckedPSet, factory has ignore_extra parameter
    def factory_with_ignore_extra2(x, ignore_extra=False):
        return x
    
    test_field = _PField(type={CheckedPSet}, invariant=PFIELD_NO_INVARIANT, 
                         initial=PFIELD_NO_INITIAL, mandatory=False, 
                         factory=factory_with_ignore_extra2, serializer=PFIELD_NO_SERIALIZER)
    result = is_field_ignore_extra_complaint(CheckedPSet, test_field, True)
    assert result is True

    # Test case 6: ignore_extra is True, field type is a tuple containing CheckedPVector, factory has ignore_extra
    def factory_with_ignore_extra3(x, ignore_extra=False):
        return x
    
    test_field = _PField(type=(CheckedPVector,), invariant=PFIELD_NO_INVARIANT, 
                         initial=PFIELD_NO_INITIAL, mandatory=False, 
                         factory=factory_with_ignore_extra3, serializer=PFIELD_NO_SERIALIZER)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is True

    # Test case 7: ignore_extra is True, field type is set, should return False (is_type_cls returns False for set type)
    test_field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, 
                         initial=PFIELD_NO_INITIAL, mandatory=False, 
                         factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    result = is_field_ignore_extra_complaint(CheckedPVector, test_field, True)
    assert result is False


# LLM-generated content at query #36
#--------------------------

```python
def test_serialize():
    # Test 1: serialize with CheckedType and PFIELD_NO_SERIALIZER
    class TestCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}"
    
    test_obj = TestCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", test_obj)
    assert result == "serialized_json"
    
    # Test 2: serialize with CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "xml", test_obj)
    assert result == "custom_xml_" + str(test_obj)
    
    # Test 3: serialize non-CheckedType with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", "test_value")
    assert result == "test_value"
    
    # Test 4: serialize non-CheckedType with custom serializer
    def format_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(format_serializer, "csv", 42)
    assert result == "csv:42"
    
    # Test 5: serialize with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 6: serialize with dict value and custom serializer
    def dict_serializer(format, value):
        return str(sorted(value.items()))
    
    test_dict = {"a": 1, "b": 2}
    result = serialize(dict_serializer, "json", test_dict)
    assert "[('a', 1), ('b', 2)]" == result
    
    # Test 7: serialize with list value
    def list_serializer(format, value):
        return f"{format}:{len(value)}"
    
    result = serialize(list_serializer, "xml", [1, 2, 3])
    assert result == "xml:3"


# LLM-generated content at query #37
#--------------------------

```python
def test_check_global_invariants():
    # Test with all invariants passing
    passing_invariants = [
        lambda x: (True, None),
        lambda x: (True, "ok"),
    ]
    check_global_invariants("test_subject", passing_invariants)  # Should not raise
    
    # Test with single failing invariant
    failing_invariants = [
        lambda x: (False, "error_code_1"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", failing_invariants)
    assert exc_info.value.error_codes == ("error_code_1",)
    assert exc_info.value.path == ()
    
    # Test with multiple failing invariants
    multiple_failing_invariants = [
        lambda x: (False, "error_1"),
        lambda x: (True, "ok"),
        lambda x: (False, "error_2"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", multiple_failing_invariants)
    assert exc_info.value.error_codes == ("error_1", "error_2")
    
    # Test with empty invariants list
    check_global_invariants("test_subject", [])  # Should not raise
    
    # Test with mixed invariants
    mixed_invariants = [
        lambda x: (True, "pass1"),
        lambda x: (False, "fail1"),
        lambda x: (True, "pass2"),
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants("test_subject", mixed_invariants)
    assert exc_info.value.error_codes == ("fail1",)
    
    # Test that subject is passed to invariants
    subjects_received = []
    def capturing_invariant(x):
        subjects_received.append(x)
        return (True, None)
    
    test_subject = {"key": "value"}
    check_global_invariants(test_subject, [capturing_invariant])
    assert subjects_received == [test_subject]


# LLM-generated content at query #38
#--------------------------

```python
def test_check_global_invariants():
    # Test 1: All invariants pass
    def invariant_pass(subject):
        return (True, None)
    
    invariants = [invariant_pass, invariant_pass]
    subject = object()
    
    # Should not raise
    check_global_invariants(subject, invariants)
    
    # Test 2: Single invariant fails
    def invariant_fail(subject):
        return (False, "error_code_1")
    
    invariants = [invariant_fail]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("error_code_1",)
        assert e.path == ()
        assert "Global invariant failed" in str(e)
    
    # Test 3: Multiple invariants fail
    def invariant_fail_2(subject):
        return (False, "error_code_2")
    
    invariants = [invariant_fail, invariant_fail_2]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert set(e.error_codes) == {"error_code_1", "error_code_2"}
        assert e.path == ()
    
    # Test 4: Mixed passing and failing invariants
    invariants = [invariant_pass, invariant_fail, invariant_pass, invariant_fail_2]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert set(e.error_codes) == {"error_code_1", "error_code_2"}
    
    # Test 5: Empty invariants list
    invariants = []
    check_global_invariants(subject, invariants)
    
    # Test 6: Invariant with different error codes
    def invariant_fail_custom(subject):
        return (False, "custom_error")
    
    invariants = [invariant_fail_custom]
    
    try:
        check_global_invariants(subject, invariants)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("custom_error",)


# LLM-generated content at query #39
#--------------------------

```python
def test_pmap_field():
    """Test pmap_field function for creating checked PMap fields."""
    
    # Test basic pmap_field creation
    pmap_fld = pmap_field(str, int)
    assert pmap_fld.mandatory is True
    assert pmap_fld.factory is not None
    assert callable(pmap_fld.factory)
    
    # Test that initial value is an empty TheMap instance
    assert pmap_fld.initial is not None
    
    # Test optional=False
    pmap_fld_required = pmap_field(str, int, optional=False)
    assert pmap_fld_required.mandatory is True
    
    # Test optional=True
    pmap_fld_optional = pmap_field(str, int, optional=True)
    assert pmap_fld_optional.mandatory is True
    
    # Test factory with optional=False
    TheMap = _make_pmap_field_type(str, int)
    factory_required = pmap_field(str, int, optional=False).factory
    result = factory_required({"a": 1, "b": 2})
    assert result is not None
    assert isinstance(result, CheckedPMap)
    
    # Test factory with optional=True returns None for None input
    factory_optional = pmap_field(str, int, optional=True).factory
    result_none = factory_optional(None)
    assert result_none is None
    
    # Test factory with optional=True creates map for non-None input
    result_map = factory_optional({"x": 10})
    assert result_map is not None
    assert isinstance(result_map, CheckedPMap)
    
    # Test with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")
    
    pmap_fld_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert pmap_fld_with_invariant.invariant is not None
    assert callable(pmap_fld_with_invariant.invariant)
    
    # Test that different key/value types create different field types
    pmap_fld_str_int = pmap_field(str, int)
    pmap_fld_int_str = pmap_field(int, str)
    assert pmap_fld_str_int.type != pmap_fld_int_str.type
    
    # Test that same key/value types reuse the cached type
    pmap_fld_1 = pmap_field(str, int)
    pmap_fld_2 = pmap_field(str, int)
    assert pmap_fld_1.type == pmap_fld_2.type
    
    # Test field type is set correctly
    pmap_fld = pmap_field(str, int, optional=False)
    assert pmap_fld.type is not None
    
    # Test optional field type wrapping
    pmap_fld_opt = pmap_field(str, int, optional=True)
    assert pmap_fld_opt.type is not None


# LLM-generated content at query #40
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    def invariant_pass_1(subject):
        return (True, None)
    
    def invariant_pass_2(subject):
        return (True, None)
    
    invariants = [invariant_pass_1, invariant_pass_2]
    subject = object()
    
    # Should not raise any exception
    check_global_invariants(subject, invariants)
    
    # Test case 2: Single invariant fails
    def invariant_fail(subject):
        return (False, "error_code_1")
    
    invariants = [invariant_fail]
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    
    assert exc_info.value.error_codes == ("error_code_1",)
    assert exc_info.value.invariant_errors == ()
    assert "Global invariant failed" in str(exc_info.value)
    
    # Test case 3: Multiple invariants, some fail
    def invariant_pass(subject):
        return (True, None)
    
    def invariant_fail_1(subject):
        return (False, "error_1")
    
    def invariant_fail_2(subject):
        return (False, "error_2")
    
    invariants = [invariant_pass, invariant_fail_1, invariant_fail_2]
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    
    assert set(exc_info.value.error_codes) == {"error_1", "error_2"}
    assert len(exc_info.value.error_codes) == 2
    
    # Test case 4: Empty invariants list
    invariants = []
    # Should not raise any exception
    check_global_invariants(subject, invariants)
    
    # Test case 5: Invariants with various error codes
    def inv_fail_a(subject):
        return (False, "code_a")
    
    def inv_fail_b(subject):
        return (False, "code_b")
    
    def inv_fail_c(subject):
        return (False, "code_c")
    
    invariants = [inv_fail_a, inv_fail_b, inv_fail_c]
    
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    
    assert set(exc_info.value.error_codes) == {"code_a", "code_b", "code_c"}


