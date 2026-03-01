####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_set_fields():
    # Test 1: Basic functionality with empty bases
    class Base1:
        pass
    
    class Base2:
        pass
    
    dct = {'field1': 'value1', 'field2': 'value2'}
    bases = [Base1, Base2]
    name = 'test_fields'
    
    set_fields(dct, bases, name)
    
    assert name in dct
    assert dct[name] == {}
    assert 'field1' not in dct
    assert 'field2' not in dct
    
    # Test 2: With existing fields in bases
    class Base1:
        _fields = {'base1_field': 'base1_value'}
    
    class Base2:
        _fields = {'base2_field': 'base2_value'}
    
    dct = {'new_field': 'new_value'}
    bases = [Base1, Base2]
    name = '_fields'
    
    set_fields(dct, bases, name)
    
    assert name in dct
    assert dct[name] == {'base1_field': 'base1_value', 'base2_field': 'base2_value'}
    assert 'new_field' not in dct
    
    # Test 3: With _PField instances in dct
    class Base1:
        _fields = {'base_field': 'base_value'}
    
    class Base2:
        pass
    
    pfield_instance = _PField(
        type=int,
        invariant=lambda x: (True, None),
        initial=0,
        mandatory=False,
        factory=lambda x: x,
        serializer=lambda f, v: v
    )
    
    dct = {'custom_field': pfield_instance}
    bases = [Base1, Base2]
    name = '_fields'
    
    set_fields(dct, bases, name)
    
    assert name in dct
    assert dct[name]['base_field'] == 'base_value'
    assert dct[name]['custom_field'] is pfield_instance
    assert 'custom_field' not in dct
    
    # Test 4: Multiple inheritance with overlapping field names
    class Base1:
        _fields = {'field1': 'value1', 'common_field': 'base1_value'}
    
    class Base2:
        _fields = {'field2': 'value2', 'common_field': 'base2_value'}
    
    dct = {}
    bases = [Base1, Base2]
    name = '_fields'
    
    set_fields(dct, bases, name)
    
    # Last base wins for overlapping fields
    assert dct[name]['common_field'] == 'base2_value'
    assert dct[name]['field1'] == 'value1'
    assert dct[name]['field2'] == 'value2'
    
    # Test 5: Empty dct and empty bases
    dct = {}
    bases = []
    name = 'fields'
    
    set_fields(dct, bases, name)
    
    assert name in dct
    assert dct[name] == {}


# LLM-generated content at query #2
#--------------------------

```python
def test_serialize():
    # Test with non-CheckedType value and default serializer
    result = serialize(lambda fmt, val: f"{fmt}:{val}", "json", "test_value")
    assert result == "json:test_value"
    
    # Test with CheckedType value and PFIELD_NO_SERIALIZER
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    checked_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    assert result == "serialized_json"
    
    # Test with custom serializer and CheckedType value
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "custom_xml_data"
    
    # Test with None value
    result = serialize(lambda fmt, val: f"{fmt}:{val}", "json", None)
    assert result == "json:None"
    
    # Test with integer value
    result = serialize(lambda fmt, val: f"{fmt}:{val}", "json", 42)
    assert result == "json:42"


# LLM-generated content at query #3
#--------------------------

```python
def test_set_fields():
    # Test 1: Basic functionality - fields are moved from dct to the specified name
    class Base1:
        _fields = {'base1_field': 'value1'}
    
    class Base2:
        _fields = {'base2_field': 'value2'}
    
    dct = {
        'field1': 'value1',
        'field2': 'value2',
        '_fields': {'existing_field': 'existing_value'}
    }
    
    # Create _PField instances to simulate field definitions
    pfield1 = _PField(type=set(), invariant=lambda x: (True, None), 
                      initial=object(), mandatory=False, 
                      factory=lambda x: x, serializer=lambda f, v: v)
    pfield2 = _PField(type=set(), invariant=lambda x: (True, None), 
                      initial=object(), mandatory=False, 
                      factory=lambda x: x, serializer=lambda f, v: v)
    
    dct['pfield1'] = pfield1
    dct['pfield2'] = pfield2
    
    bases = [Base1, Base2]
    
    set_fields(dct, bases, '_fields')
    
    # Check that _fields contains all expected entries
    assert '_fields' in dct
    assert dct['_fields']['base1_field'] == 'value1'
    assert dct['_fields']['base2_field'] == 'value2'
    assert dct['_fields']['existing_field'] == 'existing_value'
    assert dct['_fields']['pfield1'] is pfield1
    assert dct['_fields']['pfield2'] is pfield2
    
    # Check that _PField instances were removed from dct
    assert 'pfield1' not in dct
    assert 'pfield2' not in dct
    
    # Check that regular fields remain in dct
    assert dct['field1'] == 'value1'
    assert dct['field2'] == 'value2'
    
    # Test 2: Empty bases list
    dct2 = {'_fields': {'existing': 'value'}}
    pfield3 = _PField(type=set(), invariant=lambda x: (True, None), 
                      initial=object(), mandatory=False, 
                      factory=lambda x: x, serializer=lambda f, v: v)
    dct2['pfield3'] = pfield3
    
    set_fields(dct2, [], '_fields')
    
    assert '_fields' in dct2
    assert dct2['_fields']['existing'] == 'value'
    assert dct2['_fields']['pfield3'] is pfield3
    assert 'pfield3' not in dct2
    
    # Test 3: Bases without the specified attribute
    class BaseWithoutAttr:
        pass
    
    dct3 = {'_fields': {}}
    pfield4 = _PField(type=set(), invariant=lambda x: (True, None), 
                      initial=object(), mandatory=False, 
                      factory=lambda x: x, serializer=lambda f, v: v)
    dct3['pfield4'] = pfield4
    
    set_fields(dct3, [BaseWithoutAttr], '_fields')
    
    assert '_fields' in dct3
    assert dct3['_fields']['pfield4'] is pfield4
    assert 'pfield4' not in dct3
    
    # Test 4: Multiple inheritance with overlapping field names
    class BaseA:
        _fields = {'common_field': 'from_base_a', 'unique_a': 'a'}
    
    class BaseB:
        _fields = {'common_field': 'from_base_b', 'unique_b': 'b'}
    
    dct4 = {'_fields': {'existing': 'value'}}
    
    set_fields(dct4, [BaseA, BaseB], '_fields')
    
    # Later bases should override earlier ones
    assert dct4['_fields']['common_field'] == 'from_base_b'
    assert dct4['_fields']['unique_a'] == 'a'
    assert dct4['_fields']['unique_b'] == 'b'
    assert dct4['_fields']['existing'] == 'value'
    
    # Test 5: Non-_PField items in dct are not moved
    dct5 = {'_fields': {}, 'regular_attr': 'not_moved', 'another': 'also_not_moved'}
    
    set_fields(dct5, [], '_fields')
    
    assert 'regular_attr' in dct5
    assert 'another' in dct5
    assert dct5['regular_attr'] == 'not_moved'
    assert dct5['another'] == 'also_not_moved'


# LLM-generated content at query #4
#--------------------------

```python
def test_field():
    # Test basic field creation with no parameters
    f = field()
    assert isinstance(f, _PField)
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial is PFIELD_NO_INITIAL
    assert f.mandatory is False
    assert f._factory is PFIELD_NO_FACTORY
    assert f.serializer is PFIELD_NO_SERIALIZER

    # Test field with single type
    f = field(type=int)
    assert f.type == {int}
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial is PFIELD_NO_INITIAL

    # Test field with multiple types as list
    f = field(type=[int, str])
    assert f.type == {int, str}

    # Test field with multiple types as tuple
    f = field(type=(int, str))
    assert f.type == {int, str}

    # Test field with multiple types as set
    f = field(type={int, str})
    assert f.type == {int, str}

    # Test field with invariant
    def inv(x):
        return x > 0, "Must be positive"
    
    f = field(type=int, invariant=inv)
    assert f.invariant is not PFIELD_NO_INVARIANT
    assert f.invariant(5) == (True, None)
    assert f.invariant(-1) == (False, "Must be positive")

    # Test field with initial value
    f = field(type=int, initial=10)
    assert f.initial == 10

    # Test field with callable initial
    f = field(type=int, initial=lambda: 42)
    assert callable(f.initial)
    assert f.initial() == 42

    # Test mandatory field
    f = field(mandatory=True)
    assert f.mandatory is True

    # Test field with factory
    def custom_factory(x):
        return str(x)
    
    f = field(factory=custom_factory)
    assert f.factory is custom_factory
    assert f.factory(123) == "123"

    # Test field with serializer
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    f = field(serializer=custom_serializer)
    assert f.serializer is custom_serializer
    assert f.serializer("json", 123) == "json:123"

    # Test field with all parameters
    def inv(x):
        return len(x) > 0, "Must not be empty"
    
    def factory(x):
        return x.upper()
    
    def serializer(format, value):
        return value.lower()
    
    f = field(
        type=str,
        invariant=inv,
        initial="hello",
        mandatory=True,
        factory=factory,
        serializer=serializer
    )
    
    assert f.type == {str}
    assert f.invariant("test") == (True, None)
    assert f.invariant("") == (False, "Must not be empty")
    assert f.initial == "hello"
    assert f.mandatory is True
    assert f.factory("test") == "TEST"
    assert f.serializer("any", "TEST") == "test"

    # Test field with string type
    f = field(type="int")
    assert len(f.type) == 1
    # Note: The actual type resolution happens elsewhere

    # Test field with optional type wrapper
    f = field(type=optional_type(int))
    assert len(f.type) == 1
    # The optional_type creates a special wrapper

    # Test that field removes _PField instances from dct
    class TestClass:
        pass
    
    dct = {'field1': _PField(type={int}, invariant=PFIELD_NO_INVARIANT,
                             initial=PFIELD_NO_INITIAL, mandatory=False,
                             factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER),
           'regular_attr': 'value'}
    
    original_dct = dict(dct)
    set_fields(dct, [], '_precord_fields')
    
    assert 'field1' not in dct
    assert 'regular_attr' in dct
    assert dct['regular_attr'] == 'value'
    assert '_precord_fields' in dct
    assert 'field1' in dct['_precord_fields']
    assert dct['_precord_fields']['field1'] == original_dct['field1']

    # Test error cases
    # Invalid type parameter (not type, str, or container)
    try:
        field(type=123)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Invalid initial type
    try:
        field(type=int, initial="not an int")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Non-callable invariant
    try:
        field(invariant="not callable")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Non-callable factory
    try:
        field(factory="not callable")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Non-callable serializer
    try:
        field(serializer="not callable")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    f = pmap_field(int, str)
    assert isinstance(f, _PField)
    assert f.mandatory is True
    assert f.initial == {}
    assert f.type == {optional_type(f._factory({}).__class__)}
    
    # Test with optional=True
    f_optional = pmap_field(int, str, optional=True)
    assert f_optional.mandatory is True
    assert f_optional.initial == {}
    assert optional_type in [type(t) for t in f_optional.type]
    
    # Test factory function with non-None argument
    test_map = {1: "a", 2: "b"}
    result = f_optional.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map
    
    # Test factory function with None argument when optional=True
    assert f_optional.factory(None) is None
    
    # Test factory function with None argument when optional=False
    f_non_optional = pmap_field(int, str, optional=False)
    result = f_non_optional.factory({})
    assert isinstance(result, CheckedPMap)
    
    # Test with invariant
    def custom_invariant(value):
        return len(value) > 0, "Map must not be empty"
    
    f_with_invariant = pmap_field(int, str, invariant=custom_invariant)
    assert f_with_invariant.invariant is not PFIELD_NO_INVARIANT
    
    # Test type checking
    TheMap = f.factory({}).__class__
    assert issubclass(TheMap, CheckedPMap)
    assert TheMap.__key_type__ == int
    assert TheMap.__value_type__ == str
    
    # Test that different type combinations create different classes
    f2 = pmap_field(str, int)
    TheMap2 = f2.factory({}).__class__
    assert TheMap2 is not TheMap
    assert TheMap2.__key_type__ == str
    assert TheMap2.__value_type__ == int
    
    # Test pickle restoration
    import pickle
    original = f.factory({1: "test", 2: "data"})
    pickled = pickle.dumps(original)
    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, CheckedPMap)
    assert dict(unpickled) == {1: "test", 2: "data"}
    
    # Test that same type combination returns same class
    f3 = pmap_field(int, str)
    TheMap3 = f3.factory({}).__class__
    assert TheMap3 is TheMap


# LLM-generated content at query #6
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    f = pmap_field(int, str)
    assert isinstance(f, _PField)
    assert f.mandatory is True
    assert f.initial == {}
    assert len(f.type) == 1
    map_type = next(iter(f.type))
    assert issubclass(map_type, CheckedPMap)
    assert map_type.__key_type__ == int
    assert map_type.__value_type__ == str
    
    # Test optional pmap_field
    f_optional = pmap_field(int, str, optional=True)
    assert f_optional.mandatory is True
    assert len(f_optional.type) == 1
    optional_type = next(iter(f_optional.type))
    assert optional_type.__name__ == "OptionalType"
    
    # Test with invariant
    def custom_invariant(m):
        return (len(m) < 3, "Too many items")
    
    f_with_invariant = pmap_field(int, str, invariant=custom_invariant)
    assert f_with_invariant.invariant is not PFIELD_NO_INVARIANT
    
    # Test factory function for non-optional field
    map_instance = f.factory({1: "a", 2: "b"})
    assert isinstance(map_instance, CheckedPMap)
    assert dict(map_instance) == {1: "a", 2: "b"}
    
    # Test factory function for optional field
    assert f_optional.factory(None) is None
    map_instance_opt = f_optional.factory({3: "c"})
    assert isinstance(map_instance_opt, CheckedPMap)
    assert dict(map_instance_opt) == {3: "c"}
    
    # Test type name generation
    f2 = pmap_field(str, int)
    map_type2 = next(iter(f2.type))
    assert "StrToIntPMap" in map_type2.__name__
    
    # Test that same types produce same class (caching)
    f3 = pmap_field(int, str)
    map_type3 = next(iter(f3.type))
    assert map_type3 is map_type
    
    # Test with different types
    f4 = pmap_field(str, list)
    map_type4 = next(iter(f4.type))
    assert map_type4 is not map_type
    assert map_type4.__key_type__ == str
    assert map_type4.__value_type__ == list
    
    # Test initial value
    initial_map = f.initial
    assert isinstance(initial_map, CheckedPMap)
    assert len(initial_map) == 0
    assert initial_map.__key_type__ == int
    assert initial_map.__value_type__ == str
    
    # Test that field is mandatory
    assert f.mandatory is True
    assert f_optional.mandatory is True


# LLM-generated content at query #7
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = {"test": "data"}
    invariants = []
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with passing invariants
    invariants = [
        lambda x: (True, None),
        lambda x: (True, "OK"),
        lambda x: (True, 123)
    ]
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with single failing invariant
    invariants = [
        lambda x: (False, "error1")
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("error1",)
        assert e.field_errors == ()

    # Test with multiple failing invariants
    invariants = [
        lambda x: (True, "ok1"),
        lambda x: (False, "error1"),
        lambda x: (True, "ok2"),
        lambda x: (False, "error2"),
        lambda x: (False, "error3")
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("error1", "error2", "error3")
        assert e.field_errors == ()

    # Test with mixed return types in error codes
    invariants = [
        lambda x: (False, 123),
        lambda x: (False, "string_error"),
        lambda x: (False, {"code": "dict_error"})
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == (123, "string_error", {"code": "dict_error"})
        assert e.field_errors == ()

    # Test that subject is passed to invariants
    captured_subject = None
    def capturing_invariant(subj):
        nonlocal captured_subject
        captured_subject = subj
        return (True, None)
    
    test_subject = {"key": "value"}
    invariants = [capturing_invariant]
    check_global_invariants(test_subject, invariants)
    assert captured_subject == test_subject


# LLM-generated content at query #8
#--------------------------

```python
def test_serialize():
    # Test 1: Serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}_{self.__class__.__name__}"
    
    mock_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_value)
    assert result == "serialized_json_MockCheckedType"
    
    # Test 2: Serializer is PFIELD_NO_SERIALIZER and value is not CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "json", "plain_string")
    assert result == "plain_string"
    
    # Test 3: Custom serializer function
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "xml:data"
    
    # Test 4: Serializer with CheckedType value but custom serializer provided
    result = serialize(custom_serializer, "xml", mock_value)
    assert result == "xml:MockCheckedType"
    
    # Test 5: None value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 6: Integer value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test 7: List value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", [1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #9
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Mock classes to simulate CheckedType subclasses
    class MockCheckedType:
        pass
    
    class MockCheckedPVector(MockCheckedType):
        pass
    
    class MockCheckedPSet(MockCheckedType):
        pass
    
    # Test 1: ignore_extra is False should return False
    field = type('Field', (), {'type': (MockCheckedPVector,), 'factory': lambda x: x})
    assert not is_field_ignore_extra_complaint(MockCheckedPVector, field, False)
    
    # Test 2: ignore_extra is True but field type is not sequence type
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x})
    assert not is_field_ignore_extra_complaint(MockCheckedPVector, field, True)
    
    # Test 3: ignore_extra is True, field type is sequence type, but factory doesn't have ignore_extra parameter
    field = type('Field', (), {'type': (MockCheckedPVector,), 'factory': lambda x: x})
    assert not is_field_ignore_extra_complaint(MockCheckedPVector, field, True)
    
    # Test 4: ignore_extra is True, field type is sequence type, factory has ignore_extra parameter
    def factory_with_ignore_extra(argument, ignore_extra=False):
        return argument
    
    field = type('Field', (), {'type': (MockCheckedPVector,), 'factory': factory_with_ignore_extra})
    assert is_field_ignore_extra_complaint(MockCheckedPVector, field, True)
    
    # Test 5: field type as set instead of tuple
    field = type('Field', (), {'type': {MockCheckedPVector}, 'factory': factory_with_ignore_extra})
    assert is_field_ignore_extra_complaint(MockCheckedPVector, field, True)
    
    # Test 6: field type with multiple types including sequence type
    field = type('Field', (), {'type': (MockCheckedPVector, int), 'factory': factory_with_ignore_extra})
    assert is_field_ignore_extra_complaint(MockCheckedPVector, field, True)
    
    # Test 7: empty type tuple should return False
    field = type('Field', (), {'type': (), 'factory': factory_with_ignore_extra})
    assert not is_field_ignore_extra_complaint(MockCheckedPVector, field, True)
    
    # Test 8: factory with other parameters plus ignore_extra
    def factory_with_multiple_params(argument, some_param=None, ignore_extra=False, another_param=None):
        return argument
    
    field = type('Field', (), {'type': (MockCheckedPSet,), 'factory': factory_with_multiple_params})
    assert is_field_ignore_extra_complaint(MockCheckedPSet, field, True)
    
    # Test 9: factory with ignore_extra in different position
    def factory_ignore_extra_first(ignore_extra=False, argument=None):
        return argument
    
    field = type('Field', (), {'type': (MockCheckedPVector,), 'factory': factory_ignore_extra_first})
    assert is_field_ignore_extra_complaint(MockCheckedPVector, field, True)


# LLM-generated content at query #10
#--------------------------

```python
def test_check_type():
    # Mock classes for testing
    class MockDestinationClass:
        __name__ = "MockRecord"
    
    class ValidType1:
        pass
    
    class ValidType2:
        pass
    
    class InvalidType:
        pass
    
    # Test 1: Field with no type restriction should accept any value
    field_no_type = type('Field', (), {'type': None})()
    check_type(MockDestinationClass, field_no_type, "test_field", "any_value")
    check_type(MockDestinationClass, field_no_type, "test_field", 123)
    check_type(MockDestinationClass, field_no_type, "test_field", object())
    
    # Test 2: Field with single type should accept matching type
    field_single_type = type('Field', (), {'type': (ValidType1,)})()
    valid_instance = ValidType1()
    check_type(MockDestinationClass, field_single_type, "test_field", valid_instance)
    
    # Test 3: Field with multiple types should accept any matching type
    field_multi_type = type('Field', (), {'type': (ValidType1, ValidType2)})()
    valid_instance1 = ValidType1()
    valid_instance2 = ValidType2()
    check_type(MockDestinationClass, field_multi_type, "test_field", valid_instance1)
    check_type(MockDestinationClass, field_multi_type, "test_field", valid_instance2)
    
    # Test 4: Field with type restriction should reject non-matching type
    field_with_type = type('Field', (), {'type': (ValidType1,)})()
    invalid_instance = InvalidType()
    
    try:
        check_type(MockDestinationClass, field_with_type, "test_field", invalid_instance)
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.source_class == MockDestinationClass
        assert e.field == "test_field"
        assert e.expected_types == (ValidType1,)
        assert e.actual_type == InvalidType
        assert "Invalid type for field MockRecord.test_field, was InvalidType" in str(e)
    
    # Test 5: Field with empty type tuple should reject all values
    field_empty_type = type('Field', (), {'type': ()})()
    
    try:
        check_type(MockDestinationClass, field_empty_type, "test_field", "any_value")
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.source_class == MockDestinationClass
        assert e.field == "test_field"
        assert e.expected_types == ()
        assert e.actual_type == str
    
    # Test 6: Field with type that uses get_type resolution (string type names)
    # This tests the isinstance check with get_type
    class StringType:
        pass
    
    # Mock get_type to return StringType when called with "StringType"
    original_get_type = get_type
    try:
        # Temporarily replace get_type for this test
        import pyrsistent._checked_types
        pyrsistent._checked_types.get_type = lambda x: StringType if x == "StringType" else original_get_type(x)
        
        field_string_type = type('Field', (), {'type': ("StringType",)})()
        valid_instance = StringType()
        check_type(MockDestinationClass, field_string_type, "test_field", valid_instance)
        
        invalid_instance = InvalidType()
        try:
            check_type(MockDestinationClass, field_string_type, "test_field", invalid_instance)
            assert False, "Should have raised PTypeError"
        except PTypeError:
            pass
    finally:
        # Restore original get_type
        pyrsistent._checked_types.get_type = original_get_type
    
    # Test 7: Multiple inheritance type check
    class BaseType:
        pass
    
    class DerivedType(BaseType):
        pass
    
    field_base_type = type('Field', (), {'type': (BaseType,)})()
    derived_instance = DerivedType()
    check_type(MockDestinationClass, field_base_type, "test_field", derived_instance)


# LLM-generated content at query #11
#--------------------------

```python
def test_serialize():
    from pyrsistent import PRecord, field
    from pyrsistent._checked_types import CheckedType
    
    class TestRecord(PRecord):
        value = field()
    
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}_{self.value}"
        
        def __init__(self, value):
            self.value = value
    
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_obj = MockCheckedType("test")
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_obj)
    assert result == "serialized_json_test"
    
    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", "regular_string")
    assert result == "regular_string"
    
    # Test with custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "custom_xml_data"
    
    # Test with CheckedType and custom serializer (should use custom serializer)
    result = serialize(custom_serializer, "xml", checked_obj)
    assert result == "custom_xml_{}".format(checked_obj)


# LLM-generated content at query #12
#--------------------------

```python
def test_serialize():
    # Test 1: Serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}_{self.__class__.__name__}"
    
    mock_checked = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_checked)
    assert result == "serialized_json_MockCheckedType"
    
    # Test 2: Serializer is PFIELD_NO_SERIALIZER and value is not CheckedType
    regular_value = {"key": "value"}
    result = serialize(PFIELD_NO_SERIALIZER, "json", regular_value)
    assert result == regular_value
    
    # Test 3: Custom serializer function
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "xml:data"
    
    # Test 4: Serializer with different formats
    def format_sensitive_serializer(format, value):
        if format == "json":
            return f'{{"{value}": true}}'
        elif format == "xml":
            return f"<{value}/>"
        return str(value)
    
    result = serialize(format_sensitive_serializer, "json", "test")
    assert result == '{"test": true}'
    
    result = serialize(format_sensitive_serializer, "xml", "test")
    assert result == "<test/>"
    
    # Test 5: None value with custom serializer
    result = serialize(custom_serializer, "json", None)
    assert result == "json:None"
    
    # Test 6: Numeric value
    result = serialize(lambda fmt, val: val * 2, "any", 5)
    assert result == 10
    
    # Test 7: List value
    result = serialize(lambda fmt, val: len(val), "any", [1, 2, 3])
    assert result == 3


# LLM-generated content at query #13
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    from unittest.mock import Mock, MagicMock
    import inspect

    # Test 1: ignore_extra is False should return False
    field = Mock()
    result = is_field_ignore_extra_complaint(CheckedPVector, field, False)
    assert result is False

    # Test 2: ignore_extra is True but field type is not sequence type
    field = Mock()
    field.type = (int, str)  # Not a sequence type
    result = is_field_ignore_extra_complaint(CheckedPVector, field, True)
    assert result is False

    # Test 3: ignore_extra is True, field type is sequence type, but factory doesn't have ignore_extra parameter
    field = Mock()
    field.type = (CheckedPVector,)
    factory_mock = Mock()
    field.factory = factory_mock
    factory_mock.__signature__ = inspect.signature(lambda x: x)
    result = is_field_ignore_extra_complaint(CheckedPVector, field, True)
    assert result is False

    # Test 4: ignore_extra is True, field type is sequence type, factory has ignore_extra parameter
    field = Mock()
    field.type = (CheckedPVector,)
    factory_mock = Mock()
    field.factory = factory_mock
    factory_mock.__signature__ = inspect.signature(lambda x, ignore_extra=False: x)
    result = is_field_ignore_extra_complaint(CheckedPVector, field, True)
    assert result is True

    # Test 5: Test with CheckedPSet instead of CheckedPVector
    field = Mock()
    field.type = (CheckedPSet,)
    factory_mock = Mock()
    field.factory = factory_mock
    factory_mock.__signature__ = inspect.signature(lambda x, ignore_extra=False: x)
    result = is_field_ignore_extra_complaint(CheckedPSet, field, True)
    assert result is True

    # Test 6: Field type is set (should return True from is_type_cls)
    field = Mock()
    field.type = {CheckedPVector}
    factory_mock = Mock()
    field.factory = factory_mock
    factory_mock.__signature__ = inspect.signature(lambda x, ignore_extra=False: x)
    result = is_field_ignore_extra_complaint(CheckedPVector, field, True)
    assert result is True

    # Test 7: Field type is empty tuple (should return False from is_type_cls)
    field = Mock()
    field.type = ()
    result = is_field_ignore_extra_complaint(CheckedPVector, field, True)
    assert result is False

    # Test 8: Field type with multiple types including sequence type
    field = Mock()
    field.type = (CheckedPVector, CheckedPSet)
    factory_mock = Mock()
    field.factory = factory_mock
    factory_mock.__signature__ = inspect.signature(lambda x, ignore_extra=False: x)
    result = is_field_ignore_extra_complaint(CheckedPVector, field, True)
    assert result is True


# LLM-generated content at query #14
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Mock field with type that is a CheckedType subclass
    class MockCheckedType:
        pass
    
    class MockCheckedPVector(MockCheckedType):
        pass
    
    # Test 1: ignore_extra is False, should return False
    mock_field1 = type('Field', (), {
        'type': (MockCheckedPVector,),
        'factory': lambda x: x
    })()
    assert not is_field_ignore_extra_complaint(MockCheckedType, mock_field1, False)
    
    # Test 2: ignore_extra is True but field type is not a type_cls subclass
    mock_field2 = type('Field', (), {
        'type': (int,),
        'factory': lambda x: x
    })()
    assert not is_field_ignore_extra_complaint(MockCheckedType, mock_field2, True)
    
    # Test 3: ignore_extra is True, field type is type_cls subclass, 
    # but factory doesn't have ignore_extra parameter
    mock_field3 = type('Field', (), {
        'type': (MockCheckedPVector,),
        'factory': lambda x: x
    })()
    assert not is_field_ignore_extra_complaint(MockCheckedType, mock_field3, True)
    
    # Test 4: ignore_extra is True, field type is type_cls subclass,
    # and factory has ignore_extra parameter
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    
    mock_field4 = type('Field', (), {
        'type': (MockCheckedPVector,),
        'factory': factory_with_ignore_extra
    })()
    assert is_field_ignore_extra_complaint(MockCheckedType, mock_field4, True)
    
    # Test 5: field.type is a set instead of tuple
    mock_field5 = type('Field', (), {
        'type': {MockCheckedPVector},
        'factory': factory_with_ignore_extra
    })()
    assert is_field_ignore_extra_complaint(MockCheckedType, mock_field5, True)
    
    # Test 6: field.type is empty tuple
    mock_field6 = type('Field', (), {
        'type': (),
        'factory': factory_with_ignore_extra
    })()
    assert not is_field_ignore_extra_complaint(MockCheckedType, mock_field6, True)
    
    # Test 7: field.type is a list (should work like tuple)
    mock_field7 = type('Field', (), {
        'type': [MockCheckedPVector],
        'factory': factory_with_ignore_extra
    })()
    assert is_field_ignore_extra_complaint(MockCheckedType, mock_field7, True)


# LLM-generated content at query #15
#--------------------------

```python
def test_serialize():
    # Test 1: Serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"

    mock_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_value)
    assert result == "serialized_json"

    # Test 2: Serializer is PFIELD_NO_SERIALIZER and value is not CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "json", "plain_value")
    assert result == "plain_value"

    # Test 3: Custom serializer function
    def custom_serializer(format, value):
        return f"{format}:{value}"

    result = serialize(custom_serializer, "xml", "data")
    assert result == "xml:data"

    # Test 4: Serializer with CheckedType but custom serializer provided
    result = serialize(custom_serializer, "yaml", mock_value)
    assert result == "yaml:serialized_yaml"

    # Test 5: None value with custom serializer
    result = serialize(custom_serializer, "json", None)
    assert result == "json:None"


# LLM-generated content at query #16
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    f = pmap_field(int, str)
    assert isinstance(f, _PField)
    assert f.mandatory is True
    assert f.initial == {}
    assert len(f.type) == 1
    map_type = next(iter(f.type))
    assert issubclass(map_type, CheckedPMap)
    assert map_type.__key_type__ == int
    assert map_type.__value_type__ == str
    
    # Test optional pmap_field
    f_optional = pmap_field(int, str, optional=True)
    assert f_optional.mandatory is True
    assert f_optional.initial == {}
    assert len(f_optional.type) == 1
    optional_type = next(iter(f_optional.type))
    assert optional_type.__origin__ == optional_type
    assert optional_type.__args__[0].__key_type__ == int
    assert optional_type.__args__[0].__value_type__ == str
    
    # Test pmap_field with invariant
    def custom_invariant(m):
        return (len(m) <= 3, "Too many items")
    
    f_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert f_with_invariant.invariant is not PFIELD_NO_INVARIANT
    
    # Test factory function for non-optional field
    map_instance = f.factory({"a": 1, "b": 2})
    assert isinstance(map_instance, CheckedPMap)
    assert dict(map_instance) == {"a": 1, "b": 2}
    
    # Test factory function for optional field
    assert f_optional.factory(None) is None
    map_instance_opt = f_optional.factory({"x": 10})
    assert isinstance(map_instance_opt, CheckedPMap)
    assert dict(map_instance_opt) == {"x": 10}
    
    # Test type checking in factory
    try:
        f.factory("not a dict")
        assert False, "Should have raised TypeError"
    except (TypeError, InvariantException):
        pass
    
    # Test that same key/value types produce same class
    f1 = pmap_field(int, str)
    f2 = pmap_field(int, str)
    type1 = next(iter(f1.type))
    type2 = next(iter(f2.type))
    assert type1 is type2
    
    # Test different key/value types produce different classes
    f3 = pmap_field(str, int)
    type3 = next(iter(f3.type))
    assert type3 is not type1
    
    # Test pmap_field name generation
    int_str_type = next(iter(f.type))
    assert "IntToStrPMap" in int_str_type.__name__
    
    str_int_type = next(iter(f3.type))
    assert "StrToIntPMap" in str_int_type.__name__
    
    # Test with complex types
    f_complex = pmap_field(optional_type(int), list)
    complex_type = next(iter(f_complex.type))
    assert complex_type.__key_type__ == optional_type(int)
    assert complex_type.__value_type__ == list


# LLM-generated content at query #17
#--------------------------

```python
def test_serialize():
    # Test 1: Serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    mock_checked = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_checked)
    assert result == "serialized_json"
    
    # Test 2: Serializer is PFIELD_NO_SERIALIZER and value is not CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "json", "test_value")
    assert result == "test_value"
    
    # Test 3: Custom serializer function
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "xml:data"
    
    # Test 4: Serializer with CheckedType value but custom serializer
    result = serialize(custom_serializer, "yaml", mock_checked)
    assert result == "yaml:{}".format(mock_checked)
    
    # Test 5: None value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 6: Integer value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test 7: List value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", [1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test 8: Complex serializer that modifies format
    def format_modifying_serializer(format, value):
        return {"format": format, "value": value}
    
    result = serialize(format_modifying_serializer, "binary", "test")
    assert result == {"format": "binary", "value": "test"}


# LLM-generated content at query #18
#--------------------------

```python
def test_check_type():
    # Mock classes for testing
    class MockDestinationClass:
        pass
    
    class ValidType1:
        pass
    
    class ValidType2:
        pass
    
    class InvalidType:
        pass
    
    # Test 1: Field with single valid type, value matches
    field1 = _PField(
        type=(ValidType1,),
        invariant=lambda _: (True, None),
        initial=object(),
        mandatory=False,
        factory=lambda x: x,
        serializer=lambda _, value: value
    )
    
    value1 = ValidType1()
    # Should not raise exception
    check_type(MockDestinationClass, field1, "test_field", value1)
    
    # Test 2: Field with multiple valid types, value matches one
    field2 = _PField(
        type=(ValidType1, ValidType2),
        invariant=lambda _: (True, None),
        initial=object(),
        mandatory=False,
        factory=lambda x: x,
        serializer=lambda _, value: value
    )
    
    value2 = ValidType2()
    # Should not raise exception
    check_type(MockDestinationClass, field2, "test_field", value2)
    
    # Test 3: Field with type, value doesn't match
    field3 = _PField(
        type=(ValidType1,),
        invariant=lambda _: (True, None),
        initial=object(),
        mandatory=False,
        factory=lambda x: x,
        serializer=lambda _, value: value
    )
    
    value3 = InvalidType()
    try:
        check_type(MockDestinationClass, field3, "test_field", value3)
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.source_class == MockDestinationClass
        assert e.field == "test_field"
        assert e.expected_types == (ValidType1,)
        assert e.actual_type == InvalidType
        assert "Invalid type for field MockDestinationClass.test_field, was InvalidType" in str(e)
    
    # Test 4: Field with no type constraint (empty tuple)
    field4 = _PField(
        type=(),
        invariant=lambda _: (True, None),
        initial=object(),
        mandatory=False,
        factory=lambda x: x,
        serializer=lambda _, value: value
    )
    
    value4 = InvalidType()
    # Should not raise exception when no type constraint
    check_type(MockDestinationClass, field4, "test_field", value4)
    
    # Test 5: Field with type constraint, value is None
    field5 = _PField(
        type=(ValidType1,),
        invariant=lambda _: (True, None),
        initial=object(),
        mandatory=False,
        factory=lambda x: x,
        serializer=lambda _, value: value
    )
    
    value5 = None
    try:
        check_type(MockDestinationClass, field5, "test_field", value5)
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.actual_type == type(None)
    
    # Test 6: Field with string type names (simulating deferred type resolution)
    # This tests the get_type function behavior
    class MockType:
        pass
    
    # Mock get_type to return MockType for string "MockType"
    original_get_type = get_type
    try:
        # Temporarily replace get_type
        import pyrsistent._checked_types
        pyrsistent._checked_types.get_type = lambda t: MockType if t == "MockType" else original_get_type(t)
        
        field6 = _PField(
            type=("MockType",),
            invariant=lambda _: (True, None),
            initial=object(),
            mandatory=False,
            factory=lambda x: x,
            serializer=lambda _, value: value
        )
        
        value6 = MockType()
        # Should not raise exception
        check_type(MockDestinationClass, field6, "test_field", value6)
        
        value7 = InvalidType()
        try:
            check_type(MockDestinationClass, field6, "test_field", value7)
            assert False, "Should have raised PTypeError"
        except PTypeError:
            pass
    finally:
        # Restore original get_type
        pyrsistent._checked_types.get_type = original_get_type


# LLM-generated content at query #19
#--------------------------

```python
def test_check_type():
    # Mock classes for testing
    class MockDestinationClass:
        __name__ = "MockDestinationClass"
    
    class AllowedType1:
        pass
    
    class AllowedType2:
        pass
    
    class NotAllowedType:
        pass
    
    # Test 1: Valid type - single allowed type
    field1 = _PField(type={AllowedType1}, invariant=None, initial=None, 
                     mandatory=False, factory=None, serializer=None)
    value1 = AllowedType1()
    try:
        check_type(MockDestinationClass, field1, "field1", value1)
    except PTypeError:
        assert False, "Should not raise PTypeError for valid type"
    
    # Test 2: Valid type - multiple allowed types
    field2 = _PField(type={AllowedType1, AllowedType2}, invariant=None, initial=None,
                     mandatory=False, factory=None, serializer=None)
    value2 = AllowedType2()
    try:
        check_type(MockDestinationClass, field2, "field2", value2)
    except PTypeError:
        assert False, "Should not raise PTypeError for valid type among multiple"
    
    # Test 3: Invalid type - not in allowed types
    field3 = _PField(type={AllowedType1}, invariant=None, initial=None,
                     mandatory=False, factory=None, serializer=None)
    value3 = NotAllowedType()
    try:
        check_type(MockDestinationClass, field3, "field3", value3)
        assert False, "Should raise PTypeError for invalid type"
    except PTypeError as e:
        assert e.source_class == MockDestinationClass
        assert e.field == "field3"
        assert e.expected_types == {AllowedType1}
        assert e.actual_type == NotAllowedType
        assert "Invalid type for field MockDestinationClass.field3" in str(e)
    
    # Test 4: No type constraint - should always pass
    field4 = _PField(type=PFIELD_NO_TYPE, invariant=None, initial=None,
                     mandatory=False, factory=None, serializer=None)
    value4 = "any value"
    try:
        check_type(MockDestinationClass, field4, "field4", value4)
    except PTypeError:
        assert False, "Should not raise PTypeError when no type constraint"
    
    # Test 5: Empty type set - should always pass
    field5 = _PField(type=set(), invariant=None, initial=None,
                     mandatory=False, factory=None, serializer=None)
    value5 = object()
    try:
        check_type(MockDestinationClass, field5, "field5", value5)
    except PTypeError:
        assert False, "Should not raise PTypeError for empty type set"
    
    # Test 6: Type with string representation
    field6 = _PField(type={"AllowedType1"}, invariant=None, initial=None,
                     mandatory=False, factory=None, serializer=None)
    value6 = AllowedType1()
    try:
        check_type(MockDestinationClass, field6, "field6", value6)
    except PTypeError:
        assert False, "Should handle string type representations"


# LLM-generated content at query #20
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class MockClass:
        pass
    
    class MockField:
        def __init__(self, type_val):
            self.type = type_val
    
    # Test single matching type
    field = MockField((str,))
    value = "test"
    try:
        check_type(MockClass, field, "test_field", value)
    except PTypeError:
        assert False, "Should not raise PTypeError for matching type"
    
    # Test multiple possible types with match
    field = MockField((str, int))
    value = 123
    try:
        check_type(MockClass, field, "test_field", value)
    except PTypeError:
        assert False, "Should not raise PTypeError for matching type in multiple options"
    
    # Test with no type constraint
    field = MockField(None)
    value = "any_value"
    try:
        check_type(MockClass, field, "test_field", value)
    except PTypeError:
        assert False, "Should not raise PTypeError when no type constraint"
    
    # Test with empty type tuple
    field = MockField(())
    value = "any_value"
    try:
        check_type(MockClass, field, "test_field", value)
    except PTypeError:
        assert False, "Should not raise PTypeError when empty type constraint"
    
    # Test type mismatch - should raise PTypeError
    field = MockField((int,))
    value = "not_an_int"
    try:
        check_type(MockClass, field, "test_field", value)
        assert False, "Should have raised PTypeError for type mismatch"
    except PTypeError as e:
        assert e.source_class == MockClass
        assert e.field == "test_field"
        assert e.expected_types == (int,)
        assert e.actual_type == str
    
    # Test type mismatch with multiple types
    field = MockField((int, float))
    value = "not_a_number"
    try:
        check_type(MockClass, field, "test_field", value)
        assert False, "Should have raised PTypeError for type mismatch with multiple options"
    except PTypeError as e:
        assert e.source_class == MockClass
        assert e.field == "test_field"
        assert e.expected_types == (int, float)
        assert e.actual_type == str
    
    # Test with CheckedType subclass
    class MockCheckedType(CheckedType):
        pass
    
    field = MockField((MockCheckedType,))
    value = MockCheckedType()
    try:
        check_type(MockClass, field, "test_field", value)
    except PTypeError:
        assert False, "Should not raise PTypeError for CheckedType subclass"
    
    # Test with string type reference
    field = MockField(("builtins.str",))
    value = "test_string"
    try:
        check_type(MockClass, field, "test_field", value)
    except PTypeError:
        assert False, "Should not raise PTypeError for string type reference"


# LLM-generated content at query #21
#--------------------------

```python
def test_serialize():
    # Test with non-CheckedType value and default serializer
    result = serialize(lambda fmt, val: f"{fmt}:{val}", "json", "test_value")
    assert result == "json:test_value"
    
    # Test with CheckedType value and default serializer
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    checked_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_value)
    assert result == "serialized_json"
    
    # Test with custom serializer and CheckedType value
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "custom_xml_data"
    
    # Test with None value
    result = serialize(lambda fmt, val: f"{fmt}:{val}", "json", None)
    assert result == "json:None"
    
    # Test with integer value
    result = serialize(lambda fmt, val: f"{fmt}:{val}", "json", 42)
    assert result == "json:42"
    
    # Test with list value
    result = serialize(lambda fmt, val: f"{fmt}:{val}", "json", [1, 2, 3])
    assert result == "json:[1, 2, 3]"


# LLM-generated content at query #22
#--------------------------

```python
def test_serialize():
    # Test with PFIELD_NO_SERIALIZER and CheckedType
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}_{self.__class__.__name__}"
    
    checked_instance = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", checked_instance)
    assert result == "serialized_json_MockCheckedType"
    
    # Test with PFIELD_NO_SERIALIZER and non-CheckedType
    regular_value = "test_value"
    result = serialize(PFIELD_NO_SERIALIZER, "json", regular_value)
    assert result == regular_value
    
    # Test with custom serializer
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "xml:data"
    
    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test with integer value
    result = serialize(PFIELD_NO_SERIALIZER, "binary", 42)
    assert result == 42
    
    # Test with list value
    test_list = [1, 2, 3]
    result = serialize(PFIELD_NO_SERIALIZER, "yaml", test_list)
    assert result == test_list


# LLM-generated content at query #23
#--------------------------

```python
def test_check_type():
    # Test with valid type
    class MockClass:
        pass
    
    class MockField:
        def __init__(self, type_val):
            self.type = type_val
    
    field = MockField((int,))
    check_type(MockClass, field, "test_field", 42)
    
    # Test with invalid type
    try:
        check_type(MockClass, field, "test_field", "not_an_int")
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.source_class == MockClass
        assert e.field == "test_field"
        assert e.expected_types == (int,)
        assert e.actual_type == str
    
    # Test with multiple allowed types
    field_multi = MockField((int, str))
    check_type(MockClass, field_multi, "multi_field", 42)
    check_type(MockClass, field_multi, "multi_field", "string")
    
    # Test with no type constraint
    field_no_type = MockField(PFIELD_NO_TYPE)
    check_type(MockClass, field_no_type, "no_type_field", "any_value")
    check_type(MockClass, field_no_type, "no_type_field", 123)
    check_type(MockClass, field_no_type, "no_type_field", object())
    
    # Test with CheckedType subclass
    class MockCheckedType(CheckedType):
        pass
    
    field_checked = MockField((MockCheckedType,))
    instance = MockCheckedType()
    check_type(MockClass, field_checked, "checked_field", instance)
    
    # Test error message
    try:
        check_type(MockClass, field, "error_field", "wrong")
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert str(e) == "Invalid type for field MockClass.error_field, was str"
    
    # Test with empty type tuple
    field_empty = MockField(())
    check_type(MockClass, field_empty, "empty_field", "anything")


# LLM-generated content at query #24
#--------------------------

```python
def test_check_type():
    # Mock classes for testing
    class MockDestinationClass:
        __name__ = "MockDestinationClass"

    class AllowedType1:
        pass

    class AllowedType2:
        pass

    class NotAllowedType:
        pass

    # Test case 1: Value matches one of the allowed types
    field1 = _PField(type={AllowedType1, AllowedType2}, invariant=None, initial=None, mandatory=False, factory=None, serializer=None)
    value1 = AllowedType1()
    try:
        check_type(MockDestinationClass, field1, "test_field", value1)
        assert True  # No exception should be raised
    except PTypeError:
        assert False, "Should not raise PTypeError when type matches"

    # Test case 2: Value does not match any allowed types
    field2 = _PField(type={AllowedType1, AllowedType2}, invariant=None, initial=None, mandatory=False, factory=None, serializer=None)
    value2 = NotAllowedType()
    try:
        check_type(MockDestinationClass, field2, "test_field", value2)
        assert False, "Should raise PTypeError when type doesn't match"
    except PTypeError as e:
        assert e.source_class == MockDestinationClass
        assert e.field == "test_field"
        assert e.expected_types == {AllowedType1, AllowedType2}
        assert e.actual_type == type(value2)
        assert "Invalid type for field MockDestinationClass.test_field" in str(e)

    # Test case 3: Field with no type restriction (empty type set)
    field3 = _PField(type=set(), invariant=None, initial=None, mandatory=False, factory=None, serializer=None)
    value3 = "any_value"
    try:
        check_type(MockDestinationClass, field3, "test_field", value3)
        assert True  # No exception should be raised with empty type set
    except PTypeError:
        assert False, "Should not raise PTypeError when type set is empty"

    # Test case 4: Field with single allowed type
    field4 = _PField(type={AllowedType1}, invariant=None, initial=None, mandatory=False, factory=None, serializer=None)
    value4 = AllowedType1()
    try:
        check_type(MockDestinationClass, field4, "test_field", value4)
        assert True  # No exception should be raised
    except PTypeError:
        assert False, "Should not raise PTypeError when type matches single allowed type"

    # Test case 5: Value is instance of subclass of allowed type
    class SubclassOfAllowed1(AllowedType1):
        pass

    field5 = _PField(type={AllowedType1}, invariant=None, initial=None, mandatory=False, factory=None, serializer=None)
    value5 = SubclassOfAllowed1()
    try:
        check_type(MockDestinationClass, field5, "test_field", value5)
        assert True  # No exception should be raised for subclass
    except PTypeError:
        assert False, "Should not raise PTypeError for subclass of allowed type"


# LLM-generated content at query #25
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = {"key": "value"}
    invariants = []
    check_global_invariants(subject, invariants)
    
    # Test with passing invariants
    def invariant1(x):
        return True, None
    
    def invariant2(x):
        return True, "OK"
    
    invariants = [invariant1, invariant2]
    check_global_invariants(subject, invariants)
    
    # Test with single failing invariant
    def failing_invariant(x):
        return False, "ERROR_1"
    
    invariants = [failing_invariant]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("ERROR_1",)
        assert e.field_errors == ()
        assert "Global invariant failed" in str(e)
    
    # Test with multiple failing invariants
    def failing_invariant2(x):
        return False, "ERROR_2"
    
    def failing_invariant3(x):
        return False, "ERROR_3"
    
    invariants = [failing_invariant, failing_invariant2, failing_invariant3]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.invariant_errors) == {"ERROR_1", "ERROR_2", "ERROR_3"}
        assert e.field_errors == ()
    
    # Test with mixed passing and failing invariants
    def passing_invariant(x):
        return True, "PASS"
    
    invariants = [passing_invariant, failing_invariant, passing_invariant, failing_invariant2]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.invariant_errors) == {"ERROR_1", "ERROR_2"}
        assert e.field_errors == ()
    
    # Test with subject being a CheckedType instance
    class MockCheckedType:
        pass
    
    checked_subject = MockCheckedType()
    invariants = [failing_invariant]
    try:
        check_global_invariants(checked_subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("ERROR_1",)


# LLM-generated content at query #26
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = {"name": "test"}
    check_global_invariants(subject, [])
    
    # Test with passing invariants
    def invariant1(x):
        return True, None
    
    def invariant2(x):
        return True, "OK"
    
    check_global_invariants(subject, [invariant1, invariant2])
    
    # Test with single failing invariant
    def failing_invariant(x):
        return False, "ERROR_1"
    
    try:
        check_global_invariants(subject, [failing_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("ERROR_1",)
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with multiple failing invariants
    def failing_invariant2(x):
        return False, "ERROR_2"
    
    def failing_invariant3(x):
        return False, "ERROR_3"
    
    try:
        check_global_invariants(subject, [invariant1, failing_invariant, failing_invariant2, failing_invariant3])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.invariant_errors) == {"ERROR_1", "ERROR_2", "ERROR_3"}
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with mixed passing and failing invariants
    try:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2, failing_invariant2])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.invariant_errors) == {"ERROR_1", "ERROR_2"}
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with different subject types
    class TestSubject:
        def __init__(self, value):
            self.value = value
    
    obj = TestSubject(42)
    
    def obj_invariant(x):
        return x.value > 0, "POSITIVE_REQUIRED"
    
    check_global_invariants(obj, [obj_invariant])
    
    def failing_obj_invariant(x):
        return x.value < 0, "NEGATIVE_REQUIRED"
    
    try:
        check_global_invariants(obj, [failing_obj_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("NEGATIVE_REQUIRED",)
        assert e.field_errors == ()


# LLM-generated content at query #27
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    from unittest.mock import Mock
    
    # Test 1: ignore_extra is False
    assert not is_field_ignore_extra_complaint(CheckedPVector, Mock(type=(int,)), False)
    
    # Test 2: ignore_extra is True but field type is not sequence type
    mock_field = Mock(type=(int,))
    assert not is_field_ignore_extra_complaint(CheckedPVector, mock_field, True)
    
    # Test 3: ignore_extra is True, field type is sequence type, but factory doesn't have ignore_extra parameter
    mock_field = Mock(type=(CheckedPVector,))
    mock_field.factory = Mock()
    mock_field.factory.__signature__ = None
    mock_signature = Mock(parameters={'other_param': Mock()})
    mock_field.factory.__signature__ = mock_signature
    assert not is_field_ignore_extra_complaint(CheckedPVector, mock_field, True)
    
    # Test 4: ignore_extra is True, field type is sequence type, factory has ignore_extra parameter
    mock_field = Mock(type=(CheckedPVector,))
    mock_field.factory = Mock()
    mock_signature = Mock(parameters={'ignore_extra': Mock(), 'other_param': Mock()})
    mock_field.factory.__signature__ = mock_signature
    assert is_field_ignore_extra_complaint(CheckedPVector, mock_field, True)
    
    # Test 5: field type is set (special case)
    mock_field = Mock(type={int, str})
    assert is_field_ignore_extra_complaint(CheckedPVector, mock_field, True)
    
    # Test 6: field type is empty tuple
    mock_field = Mock(type=())
    assert not is_field_ignore_extra_complaint(CheckedPVector, mock_field, True)
    
    # Test 7: field type with multiple types including sequence type
    mock_field = Mock(type=(CheckedPVector, int))
    mock_field.factory = Mock()
    mock_signature = Mock(parameters={'ignore_extra': Mock()})
    mock_field.factory.__signature__ = mock_signature
    assert is_field_ignore_extra_complaint(CheckedPVector, mock_field, True)


# LLM-generated content at query #28
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = {"key": "value"}
    check_global_invariants(subject, [])
    
    # Test with passing invariants
    def invariant1(x):
        return True, None
    
    def invariant2(x):
        return True, "OK"
    
    check_global_invariants(subject, [invariant1, invariant2])
    
    # Test with failing invariants
    def failing_invariant1(x):
        return False, "ERROR_1"
    
    def failing_invariant2(x):
        return False, "ERROR_2"
    
    try:
        check_global_invariants(subject, [failing_invariant1, failing_invariant2])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("ERROR_1", "ERROR_2")
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with mixed passing and failing invariants
    def passing_invariant(x):
        return True, "PASS"
    
    try:
        check_global_invariants(subject, [passing_invariant, failing_invariant1])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("ERROR_1",)
        assert e.field_errors == ()
    
    # Test with invariants returning different error code types
    def failing_invariant_int(x):
        return False, 404
    
    def failing_invariant_tuple(x):
        return False, ("ERR", "CODE")
    
    try:
        check_global_invariants(subject, [failing_invariant_int, failing_invariant_tuple])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == (404, ("ERR", "CODE"))
    
    # Test with complex subject
    complex_subject = [1, 2, 3]
    def list_length_invariant(lst):
        return len(lst) == 3, "WRONG_LENGTH"
    
    check_global_invariants(complex_subject, [list_length_invariant])
    
    def wrong_length_invariant(lst):
        return len(lst) == 5, "NEEDS_5_ITEMS"
    
    try:
        check_global_invariants(complex_subject, [wrong_length_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("NEEDS_5_ITEMS",)


# LLM-generated content at query #29
#--------------------------

```python
def test_serialize():
    # Test 1: Serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    class MockCheckedType:
        def serialize(self, format):
            return f"serialized_{format}"
    
    mock_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_value)
    assert result == "serialized_json"
    
    # Test 2: Serializer is PFIELD_NO_SERIALIZER and value is not CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "json", "test_value")
    assert result == "test_value"
    
    # Test 3: Custom serializer function
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "xml:data"
    
    # Test 4: CheckedType value with custom serializer
    mock_value = MockCheckedType()
    result = serialize(custom_serializer, "yaml", mock_value)
    assert result == "yaml:<test_serialize.<locals>.MockCheckedType object"
    
    # Test 5: None value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 6: Integer value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test 7: List value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", [1, 2, 3])
    assert result == [1, 2, 3]


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_set_fields():
    # Test basic functionality with empty bases
    class Base1:
        pass
    
    class Base2:
        pass
    
    dct = {'field1': 'value1', 'field2': 'value2'}
    bases = (Base1, Base2)
    name = 'test_fields'
    
    set_fields(dct, bases, name)
    
    assert name in dct
    assert dct[name] == {}
    assert 'field1' not in dct
    assert 'field2' not in dct
    
    # Test with _PField instances in dct
    class MockPField:
        def __init__(self, name):
            self.name = name
    
    field1 = MockPField('field1')
    field2 = MockPField('field2')
    
    dct = {'field1': field1, 'field2': field2, 'regular': 'value'}
    bases = (Base1, Base2)
    name = 'pfields'
    
    set_fields(dct, bases, name)
    
    assert name in dct
    assert dct[name]['field1'] == field1
    assert dct[name]['field2'] == field2
    assert 'regular' in dct
    assert dct['regular'] == 'value'
    assert 'field1' not in dct
    assert 'field2' not in dct
    
    # Test inheritance from bases
    class BaseWithFields:
        inherited_field = MockPField('inherited')
    
    class AnotherBase:
        another_field = MockPField('another')
    
    dct = {'new_field': MockPField('new')}
    bases = (BaseWithFields, AnotherBase)
    name = 'all_fields'
    
    set_fields(dct, bases, name)
    
    assert name in dct
    assert dct[name]['inherited_field'] == BaseWithFields.inherited_field
    assert dct[name]['another_field'] == AnotherBase.another_field
    assert dct[name]['new_field'] == dct['new_field']
    assert 'new_field' not in dct
    
    # Test with empty dct
    dct = {}
    bases = (BaseWithFields,)
    name = 'empty_fields'
    
    set_fields(dct, bases, name)
    
    assert name in dct
    assert dct[name] == {'inherited_field': BaseWithFields.inherited_field}
    
    # Test with no bases
    dct = {'field': MockPField('test')}
    bases = ()
    name = 'no_base_fields'
    
    set_fields(dct, bases, name)
    
    assert name in dct
    assert dct[name] == {'field': dct['field']}
    assert 'field' not in dct


# LLM-generated content at query #2
#--------------------------

```python
def test_set_fields():
    # Test 1: Basic functionality - fields are moved to specified name
    class Base1:
        _fields = {'field1': 'value1'}
    
    class Base2:
        _fields = {'field2': 'value2'}
    
    dct = {'field3': 'value3', '_fields': {}}
    set_fields(dct, [Base1, Base2], '_fields')
    
    assert '_fields' in dct
    assert dct['_fields']['field1'] == 'value1'
    assert dct['_fields']['field2'] == 'value2'
    assert dct['_fields']['field3'] == 'value3'
    assert 'field3' not in dct
    
    # Test 2: _PField instances are properly handled
    class MockPField:
        def __init__(self, name):
            self.name = name
    
    field1 = MockPField('field1')
    field2 = MockPField('field2')
    
    class BaseWithPFields:
        _fields = {'existing': 'value'}
    
    dct = {'new_field': field1, 'another_field': field2}
    set_fields(dct, [BaseWithPFields], '_fields')
    
    assert '_fields' in dct
    assert dct['_fields']['new_field'] == field1
    assert dct['_fields']['another_field'] == field2
    assert dct['_fields']['existing'] == 'value'
    assert 'new_field' not in dct
    assert 'another_field' not in dct
    
    # Test 3: Empty bases list
    dct = {'field1': 'value1', 'field2': 'value2'}
    set_fields(dct, [], '_fields')
    
    assert '_fields' in dct
    assert dct['_fields']['field1'] == 'value1'
    assert dct['_fields']['field2'] == 'value2'
    assert 'field1' not in dct
    assert 'field2' not in dct
    
    # Test 4: Multiple inheritance with overlapping field names
    class BaseA:
        _fields = {'field1': 'from_A', 'common': 'A'}
    
    class BaseB:
        _fields = {'field2': 'from_B', 'common': 'B'}
    
    dct = {'field3': 'from_dct'}
    set_fields(dct, [BaseA, BaseB], '_fields')
    
    # Last base in list wins for overlapping keys
    assert dct['_fields']['common'] == 'B'
    assert dct['_fields']['field1'] == 'from_A'
    assert dct['_fields']['field2'] == 'from_B'
    assert dct['_fields']['field3'] == 'from_dct'
    
    # Test 5: Base without the specified attribute
    class BaseWithoutFields:
        pass
    
    dct = {'field1': 'value1'}
    set_fields(dct, [BaseWithoutFields], '_fields')
    
    assert '_fields' in dct
    assert dct['_fields']['field1'] == 'value1'
    
    # Test 6: Mixed bases with and without fields
    class BaseWith:
        _fields = {'base_field': 'base_value'}
    
    class BaseWithout:
        pass
    
    dct = {'dct_field': 'dct_value'}
    set_fields(dct, [BaseWith, BaseWithout], '_fields')
    
    assert dct['_fields']['base_field'] == 'base_value'
    assert dct['_fields']['dct_field'] == 'dct_value'


# LLM-generated content at query #3
#--------------------------

```python
def test_serialize():
    from pyrsistent import PRecord, field
    from pyrsistent._checked_types import CheckedType
    
    class MockCheckedType(CheckedType):
        def __init__(self, data):
            self.data = data
        
        def serialize(self, format):
            return f"serialized_{format}_{self.data}"
    
    # Test 1: CheckedType with PFIELD_NO_SERIALIZER
    mock_checked = MockCheckedType("test")
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_checked)
    assert result == "serialized_json_test"
    
    # Test 2: Non-CheckedType with PFIELD_NO_SERIALIZER
    regular_value = "regular_string"
    result = serialize(PFIELD_NO_SERIALIZER, "json", regular_value)
    assert result == regular_value
    
    # Test 3: Custom serializer function
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "custom_xml_data"
    
    # Test 4: CheckedType with custom serializer (should use custom serializer, not CheckedType.serialize)
    mock_checked = MockCheckedType("test")
    result = serialize(custom_serializer, "xml", mock_checked)
    assert result == "custom_xml_<__main__.test_serialize.<locals>.MockCheckedType object"
    
    # Test 5: None value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 6: Integer value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test 7: List value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", [1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #4
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = {"key": "value"}
    check_global_invariants(subject, [])
    
    # Test with passing invariants
    def invariant1(x):
        return True, None
    
    def invariant2(x):
        return True, "OK"
    
    check_global_invariants(subject, [invariant1, invariant2])
    
    # Test with single failing invariant
    def failing_invariant(x):
        return False, "ERROR_1"
    
    try:
        check_global_invariants(subject, [failing_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("ERROR_1",)
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with multiple failing invariants
    def failing_invariant2(x):
        return False, "ERROR_2"
    
    try:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.invariant_errors) == {"ERROR_1", "ERROR_2"}
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with mixed passing and failing invariants
    def passing_invariant(x):
        return True, "PASS"
    
    try:
        check_global_invariants(subject, [passing_invariant, failing_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("ERROR_1",)
        assert e.field_errors == ()
    
    # Test with different subject types
    class TestSubject:
        def __init__(self, value):
            self.value = value
    
    obj = TestSubject(42)
    
    def obj_invariant(x):
        return x.value > 50, "TOO_SMALL"
    
    try:
        check_global_invariants(obj, [obj_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("TOO_SMALL",)
    
    # Test with passing object invariant
    def passing_obj_invariant(x):
        return x.value == 42, "CORRECT"
    
    check_global_invariants(obj, [passing_obj_invariant])


# LLM-generated content at query #5
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    f = pmap_field(int, str)
    assert isinstance(f, _PField)
    assert f.mandatory is True
    assert f.initial == {}
    assert len(f.type) == 1
    map_type = tuple(f.type)[0]
    assert issubclass(map_type, CheckedPMap)
    assert map_type.__key_type__ == int
    assert map_type.__value_type__ == str
    
    # Test optional pmap_field
    f_optional = pmap_field(int, str, optional=True)
    assert f_optional.mandatory is True
    assert len(f_optional.type) == 1
    optional_type = tuple(f_optional.type)[0]
    assert optional_type.__name__ == "OptionalType"
    
    # Test factory function for non-optional field
    test_map = {"a": 1, "b": 2}
    result = f.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map
    
    # Test factory function for optional field with None
    assert f_optional.factory(None) is None
    
    # Test factory function for optional field with value
    result_optional = f_optional.factory(test_map)
    assert isinstance(result_optional, CheckedPMap)
    assert dict(result_optional) == test_map
    
    # Test with invariant
    def custom_invariant(value):
        return len(value) > 0, "Map must not be empty"
    
    f_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert f_with_invariant.invariant is not PFIELD_NO_INVARIANT
    
    # Test type name generation
    f2 = pmap_field(str, int)
    map_type2 = tuple(f2.type)[0]
    assert "StrToIntPMap" in map_type2.__name__
    
    # Test that same types produce same class (caching)
    f3 = pmap_field(int, str)
    map_type3 = tuple(f3.type)[0]
    assert map_type3 is map_type
    
    # Test different types produce different class
    f4 = pmap_field(str, str)
    map_type4 = tuple(f4.type)[0]
    assert map_type4 is not map_type
    
    # Test pickle restoration function
    test_data = {"key1": "value1", "key2": "value2"}
    restored = _restore_pmap_field_pickle(str, str, test_data)
    assert isinstance(restored, CheckedPMap)
    assert dict(restored) == test_data
    
    # Test that factory creates correct type
    created = f.factory({"test": 123})
    assert isinstance(created, CheckedPMap)
    assert created.__key_type__ == int
    assert created.__value_type__ == str


# LLM-generated content at query #6
#--------------------------

```python
def test_set_fields():
    # Test 1: Basic functionality - fields are moved from dct to specified name
    class Base1:
        _fields = {'field1': 'value1'}
    
    class Base2:
        _fields = {'field2': 'value2'}
    
    dct = {'field3': 'value3', '_fields': {}}
    bases = (Base1, Base2)
    
    set_fields(dct, bases, '_fields')
    
    assert '_fields' in dct
    assert dct['_fields']['field1'] == 'value1'
    assert dct['_fields']['field2'] == 'value2'
    assert 'field3' not in dct  # Should be removed
    assert 'field1' not in dct  # Should be removed
    assert 'field2' not in dct  # Should be removed
    
    # Test 2: _PField instances are properly handled
    class _PField:
        def __init__(self, name):
            self.name = name
    
    field1 = _PField('field1')
    field2 = _PField('field2')
    
    dct = {'field1': field1, 'field2': field2, '_fields': {}}
    bases = ()
    
    set_fields(dct, bases, '_fields')
    
    assert '_fields' in dct
    assert dct['_fields']['field1'] == field1
    assert dct['_fields']['field2'] == field2
    assert 'field1' not in dct
    assert 'field2' not in dct
    
    # Test 3: Inheritance chain - fields from multiple bases are combined
    class GrandBase:
        _fields = {'grand_field': 'grand_value'}
    
    class Parent(GrandBase):
        _fields = {'parent_field': 'parent_value'}
    
    class Child(Parent):
        pass
    
    dct = {'child_field': 'child_value', '_fields': {}}
    bases = (Child,)
    
    set_fields(dct, bases, '_fields')
    
    assert '_fields' in dct
    assert dct['_fields']['grand_field'] == 'grand_value'
    assert dct['_fields']['parent_field'] == 'parent_value'
    assert 'child_field' not in dct
    
    # Test 4: Empty bases list
    dct = {'field1': 'value1', '_fields': {}}
    bases = ()
    
    set_fields(dct, bases, '_fields')
    
    assert '_fields' in dct
    assert dct['_fields'] == {}
    assert 'field1' not in dct
    
    # Test 5: Non-_PField values in dct are not moved to _fields
    dct = {'regular_field': 'regular_value', '_fields': {}}
    bases = ()
    
    set_fields(dct, bases, '_fields')
    
    assert '_fields' in dct
    assert dct['_fields'] == {}
    assert 'regular_field' not in dct
    
    # Test 6: Bases without the specified attribute
    class BaseWithoutFields:
        pass
    
    dct = {'field1': 'value1', '_fields': {}}
    bases = (BaseWithoutFields,)
    
    set_fields(dct, bases, '_fields')
    
    assert '_fields' in dct
    assert dct['_fields'] == {}
    assert 'field1' not in dct
    
    # Test 7: Mix of bases with and without the attribute
    class BaseWithFields:
        _fields = {'base_field': 'base_value'}
    
    class BaseWithoutFields:
        pass
    
    dct = {'dct_field': 'dct_value', '_fields': {}}
    bases = (BaseWithFields, BaseWithoutFields)
    
    set_fields(dct, bases, '_fields')
    
    assert '_fields' in dct
    assert dct['_fields']['base_field'] == 'base_value'
    assert 'dct_field' not in dct


# LLM-generated content at query #7
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    f = pmap_field(int, str)
    assert isinstance(f, _PField)
    assert f.mandatory is True
    assert isinstance(f.initial, CheckedPMap)
    assert f.type == {CheckedPMap}
    
    # Test with optional=True
    f_optional = pmap_field(int, str, optional=True)
    assert f_optional.type == {optional_type(CheckedPMap)}
    
    # Test factory function for non-optional field
    non_optional_factory = f.factory
    result = non_optional_factory({1: "a", 2: "b"})
    assert isinstance(result, CheckedPMap)
    assert result[1] == "a"
    assert result[2] == "b"
    
    # Test factory function for optional field
    optional_factory = f_optional.factory
    assert optional_factory(None) is None
    result = optional_factory({3: "c"})
    assert isinstance(result, CheckedPMap)
    assert result[3] == "c"
    
    # Test invariant parameter
    def custom_invariant(m):
        return (len(m) > 0, "map must not be empty")
    
    f_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert f_with_invariant.invariant is not PFIELD_NO_INVARIANT
    
    # Test that same key/value type combination returns cached type
    TheMap1 = _make_pmap_field_type(int, str)
    TheMap2 = _make_pmap_field_type(int, str)
    assert TheMap1 is TheMap2
    
    # Test different key/value type combinations create different types
    TheMap3 = _make_pmap_field_type(str, int)
    assert TheMap3 is not TheMap1
    
    # Test type name generation
    assert "IntToStrPMap" in TheMap1.__name__
    assert "StrToIntPMap" in TheMap3.__name__
    
    # Test pickle restoration function
    data = {1: "one", 2: "two"}
    restored = _restore_pmap_field_pickle(int, str, data)
    assert isinstance(restored, CheckedPMap)
    assert restored[1] == "one"
    assert restored[2] == "two"


# LLM-generated content at query #8
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class MockClass:
        pass
    
    class MockField:
        def __init__(self, type_val):
            self.type = type_val
    
    # Test case 1: Single matching type
    field1 = MockField((int,))
    check_type(MockClass, field1, "test_field", 42)
    
    # Test case 2: Multiple types, one matches
    field2 = MockField((int, str, float))
    check_type(MockClass, field2, "test_field", "hello")
    
    # Test case 3: Type with no constraints (empty type tuple)
    field3 = MockField(())
    check_type(MockClass, field3, "test_field", "any_value")
    check_type(MockClass, field3, "test_field", 123)
    check_type(MockClass, field3, "test_field", None)
    
    # Test case 4: Type mismatch - should raise PTypeError
    field4 = MockField((int, float))
    try:
        check_type(MockClass, field4, "test_field", "string_value")
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.source_class == MockClass
        assert e.field == "test_field"
        assert e.expected_types == (int, float)
        assert e.actual_type == str
        assert "Invalid type for field MockClass.test_field, was str" in str(e)
    
    # Test case 5: CheckedType subclass matching
    class TestCheckedType(CheckedType):
        pass
    
    field5 = MockField((TestCheckedType,))
    instance = TestCheckedType()
    check_type(MockClass, field5, "test_field", instance)
    
    # Test case 6: Type mismatch with CheckedType
    class OtherType:
        pass
    
    try:
        check_type(MockClass, field5, "test_field", OtherType())
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.actual_type == OtherType
    
    # Test case 7: Multiple type mismatch
    field6 = MockField((int, list))
    try:
        check_type(MockClass, field6, "test_field", 3.14)
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert e.expected_types == (int, list)
        assert e.actual_type == float
    
    # Test case 8: Field with None type (empty tuple)
    field7 = MockField(())
    check_type(MockClass, field7, "test_field", object())


# LLM-generated content at query #9
#--------------------------

```python
def test_serialize():
    # Test 1: Serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}_{self.__class__.__name__}"
    
    mock_checked = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_checked)
    assert result == "serialized_json_MockCheckedType"
    
    # Test 2: Serializer is PFIELD_NO_SERIALIZER and value is not CheckedType
    regular_value = "test_value"
    result = serialize(PFIELD_NO_SERIALIZER, "json", regular_value)
    assert result == regular_value
    
    # Test 3: Custom serializer function
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    test_value = "data"
    result = serialize(custom_serializer, "xml", test_value)
    assert result == "xml:data"
    
    # Test 4: Serializer with different formats
    def format_sensitive_serializer(format, value):
        if format == "json":
            return f'{{"data": "{value}"}}'
        elif format == "xml":
            return f'<data>{value}</data>'
        return str(value)
    
    result = serialize(format_sensitive_serializer, "json", "test")
    assert result == '{"data": "test"}'
    
    result = serialize(format_sensitive_serializer, "xml", "test")
    assert result == '<data>test</data>'
    
    # Test 5: Serializer returns None
    def none_serializer(format, value):
        return None
    
    result = serialize(none_serializer, "json", "test")
    assert result is None
    
    # Test 6: Serializer with complex value
    complex_value = {"key": "value", "number": 42}
    result = serialize(PFIELD_NO_SERIALIZER, "json", complex_value)
    assert result == complex_value
    
    # Test 7: CheckedType with different serialize implementation
    class AnotherCheckedType(CheckedType):
        def serialize(self, format):
            return {"format": format, "type": self.__class__.__name__}
    
    another_checked = AnotherCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "yaml", another_checked)
    assert result == {"format": "yaml", "type": "AnotherCheckedType"}


# LLM-generated content at query #10
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    f = pmap_field(int, str)
    assert isinstance(f, _PField)
    assert f.mandatory is True
    assert f.initial == {}
    assert len(f.type) == 1
    map_type = next(iter(f.type))
    assert issubclass(map_type, CheckedPMap)
    assert map_type.__key_type__ == int
    assert map_type.__value_type__ == str
    
    # Test optional pmap_field
    f_optional = pmap_field(int, str, optional=True)
    assert f_optional.mandatory is True
    assert len(f_optional.type) == 1
    optional_type = next(iter(f_optional.type))
    assert optional_type.__name__ == "OptionalType"
    
    # Test with invariant
    def custom_invariant(m):
        return (len(m) < 3, "too_many_items")
    
    f_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert f_with_invariant.invariant is not PFIELD_NO_INVARIANT
    
    # Test factory function for non-optional field
    map_type = _make_pmap_field_type(int, str)
    test_map = {"a": 1, "b": 2}
    result = f.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map
    
    # Test factory function for optional field
    result_none = f_optional.factory(None)
    assert result_none is None
    
    result_map = f_optional.factory(test_map)
    assert isinstance(result_map, CheckedPMap)
    assert dict(result_map) == test_map
    
    # Test type caching
    map_type1 = _make_pmap_field_type(int, str)
    map_type2 = _make_pmap_field_type(int, str)
    assert map_type1 is map_type2
    
    # Test different types create different classes
    map_type_str_int = _make_pmap_field_type(str, int)
    assert map_type_str_int is not map_type1
    
    # Test pickle restoration
    import pickle
    map_instance = map_type1.create({"x": 1, "y": 2})
    pickled = pickle.dumps(map_instance)
    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, CheckedPMap)
    assert dict(unpickled) == {"x": 1, "y": 2}
    
    # Test field with custom invariant
    def length_invariant(m):
        return (len(m) <= 2, "max_length_exceeded")
    
    f_custom = pmap_field(str, int, invariant=length_invariant)
    valid_map = f_custom.factory({"a": 1})
    assert isinstance(valid_map, CheckedPMap)
    
    # Test that initial value is empty map
    f = pmap_field(str, int)
    assert f.initial == {}
    assert isinstance(f.initial, CheckedPMap)
    
    # Test type name generation
    map_type = _make_pmap_field_type(int, str)
    assert "IntToStrPMap" in map_type.__name__
    
    # Test with multiple types
    from pyrsistent import PRecord, field as pfield
    
    class TestRecord(PRecord):
        my_map = pmap_field(str, int)
    
    record = TestRecord.create({"my_map": {"a": 1, "b": 2}})
    assert isinstance(record.my_map, CheckedPMap)
    assert dict(record.my_map) == {"a": 1, "b": 2}


# LLM-generated content at query #11
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field1 = pmap_field(int, str)
    assert isinstance(field1, _PField)
    assert field1.mandatory is True
    assert isinstance(field1.initial, CheckedPMap)
    assert field1.type == {field1.factory({})}
    
    # Test optional pmap_field
    field2 = pmap_field(int, str, optional=True)
    assert field2.mandatory is True
    assert field2.initial == field2.factory({})
    assert None in field2.type
    
    # Test factory function for optional field
    assert field2.factory(None) is None
    assert isinstance(field2.factory({1: "a"}), CheckedPMap)
    
    # Test factory function for non-optional field
    assert isinstance(field1.factory({1: "a"}), CheckedPMap)
    
    # Test with invariant
    def custom_invariant(value):
        return (len(value) < 3, "Too many items")
    
    field3 = pmap_field(int, str, invariant=custom_invariant)
    assert field3.invariant is not PFIELD_NO_INVARIANT
    
    # Test type checking
    TheMap = field1.factory({})
    assert isinstance(TheMap, CheckedPMap)
    
    # Test that same key/value types produce same field type
    field4 = pmap_field(int, str)
    field5 = pmap_field(int, str)
    assert field4.factory({}) == field5.factory({})
    
    # Test with different types
    field6 = pmap_field(str, int)
    field7 = pmap_field(str, list)
    assert field6.factory({}) != field7.factory({})
    
    # Test initial value
    initial_map = field1.initial
    assert isinstance(initial_map, CheckedPMap)
    assert len(initial_map) == 0
    
    # Test that factory creates correct type
    test_map = field1.factory({1: "one", 2: "two"})
    assert isinstance(test_map, CheckedPMap)
    assert test_map[1] == "one"
    assert test_map[2] == "two"


# LLM-generated content at query #12
#--------------------------

```python
def test_serialize():
    # Test with non-CheckedType value and default serializer
    result = serialize(PFIELD_NO_SERIALIZER, 'json', 'test_value')
    assert result == 'test_value'
    
    # Test with custom serializer
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, 'json', 'test')
    assert result == 'json:test'
    
    # Test with CheckedType value and default serializer
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}"
    
    mock_checked = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, 'json', mock_checked)
    assert result == 'serialized_json'
    
    # Test with CheckedType value and custom serializer
    # Custom serializer should take precedence
    result = serialize(custom_serializer, 'xml', mock_checked)
    assert result == 'xml:mock_checked'
    
    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, 'json', None)
    assert result is None
    
    # Test with integer value
    result = serialize(PFIELD_NO_SERIALIZER, 'json', 42)
    assert result == 42
    
    # Test with list value
    result = serialize(PFIELD_NO_SERIALIZER, 'json', [1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #13
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = "test"
    check_global_invariants(subject, [])
    
    # Test with passing invariants
    def invariant1(x):
        return (True, None)
    
    def invariant2(x):
        return (True, "OK")
    
    check_global_invariants(subject, [invariant1, invariant2])
    
    # Test with failing invariants
    def failing_invariant1(x):
        return (False, "error1")
    
    def failing_invariant2(x):
        return (False, "error2")
    
    try:
        check_global_invariants(subject, [failing_invariant1, failing_invariant2])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("error1", "error2")
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with mixed passing and failing invariants
    def passing_invariant(x):
        return (True, "should_not_appear")
    
    try:
        check_global_invariants(subject, [passing_invariant, failing_invariant1])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("error1",)
        assert e.field_errors == ()
    
    # Test with invariants returning different error code types
    def failing_int_invariant(x):
        return (False, 42)
    
    def failing_tuple_invariant(x):
        return (False, ("nested", "error"))
    
    try:
        check_global_invariants(subject, [failing_int_invariant, failing_tuple_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == (42, ("nested", "error"))
    
    # Test with empty subject
    check_global_invariants(None, [invariant1])
    
    # Test invariants that use the subject
    def length_invariant(s):
        return (len(s) > 0, "empty_string")
    
    check_global_invariants("hello", [length_invariant])
    
    try:
        check_global_invariants("", [length_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("empty_string",)


# LLM-generated content at query #14
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    from unittest.mock import Mock
    import inspect
    
    # Test 1: ignore_extra is False
    assert is_field_ignore_extra_complaint(CheckedPVector, Mock(type=(CheckedPVector,)), False) == False
    
    # Test 2: ignore_extra is True but field type is not a sequence type
    assert is_field_ignore_extra_complaint(CheckedPVector, Mock(type=(int,)), True) == False
    
    # Test 3: ignore_extra is True, field type is sequence, but factory doesn't have ignore_extra parameter
    mock_field = Mock(type=(CheckedPVector,))
    mock_factory = Mock()
    mock_field.factory = mock_factory
    mock_field.factory.parameters = {}
    with Mock() as mock_sig:
        mock_sig.parameters = {}
        mock_field.factory.__signature__ = mock_sig
    assert is_field_ignore_extra_complaint(CheckedPVector, mock_field, True) == False
    
    # Test 4: ignore_extra is True, field type is sequence, factory has ignore_extra parameter
    mock_field = Mock(type=(CheckedPVector,))
    mock_factory = Mock()
    mock_field.factory = mock_factory
    mock_params = {'ignore_extra': Mock()}
    with Mock() as mock_sig:
        mock_sig.parameters = mock_params
        mock_field.factory.__signature__ = mock_sig
    assert is_field_ignore_extra_complaint(CheckedPVector, mock_field, True) == True
    
    # Test 5: Test with CheckedPSet
    mock_field = Mock(type=(CheckedPSet,))
    mock_factory = Mock()
    mock_field.factory = mock_factory
    mock_params = {'ignore_extra': Mock()}
    with Mock() as mock_sig:
        mock_sig.parameters = mock_params
        mock_field.factory.__signature__ = mock_sig
    assert is_field_ignore_extra_complaint(CheckedPSet, mock_field, True) == True
    
    # Test 6: Test with multiple types in field.type
    mock_field = Mock(type=(CheckedPVector, CheckedPSet))
    mock_factory = Mock()
    mock_field.factory = mock_factory
    mock_params = {'ignore_extra': Mock()}
    with Mock() as mock_sig:
        mock_sig.parameters = mock_params
        mock_field.factory.__signature__ = mock_sig
    assert is_field_ignore_extra_complaint(CheckedPVector, mock_field, True) == True
    
    # Test 7: Test with empty type tuple
    mock_field = Mock(type=())
    assert is_field_ignore_extra_complaint(CheckedPVector, mock_field, True) == False
    
    # Test 8: Test with set type (special case)
    mock_field = Mock(type=set([CheckedPVector]))
    mock_factory = Mock()
    mock_field.factory = mock_factory
    mock_params = {'ignore_extra': Mock()}
    with Mock() as mock_sig:
        mock_sig.parameters = mock_params
        mock_field.factory.__signature__ = mock_sig
    assert is_field_ignore_extra_complaint(CheckedPVector, mock_field, True) == True


# LLM-generated content at query #15
#--------------------------

```python
def test_field():
    # Test basic field creation with default parameters
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial is PFIELD_NO_INITIAL
    assert f.mandatory is False
    assert f.factory is PFIELD_NO_FACTORY
    assert f.serializer is PFIELD_NO_SERIALIZER

    # Test field with single type
    f = field(type=int)
    assert len(f.type) == 1
    assert int in f.type

    # Test field with multiple types as list
    f = field(type=[int, str])
    assert len(f.type) == 2
    assert int in f.type
    assert str in f.type

    # Test field with multiple types as tuple
    f = field(type=(int, str))
    assert len(f.type) == 2
    assert int in f.type
    assert str in f.type

    # Test field with multiple types as set
    f = field(type={int, str})
    assert len(f.type) == 2
    assert int in f.type
    assert str in f.type

    # Test field with custom invariant
    def my_invariant(value):
        return value > 0, "Must be positive"

    f = field(type=int, invariant=my_invariant)
    assert f.invariant != PFIELD_NO_INVARIANT
    assert f.invariant(5) == (True, "Must be positive")
    assert f.invariant(-1) == (False, "Must be positive")

    # Test field with initial value
    f = field(type=int, initial=10)
    assert f.initial == 10

    # Test field with callable initial
    f = field(type=int, initial=lambda: 42)
    assert callable(f.initial)
    assert f.initial() == 42

    # Test mandatory field
    f = field(mandatory=True)
    assert f.mandatory is True

    # Test field with custom factory
    def my_factory(x):
        return str(x)

    f = field(factory=my_factory)
    assert f.factory is my_factory
    assert f.factory(123) == "123"

    # Test field with custom serializer
    def my_serializer(format, value):
        return f"{format}:{value}"

    f = field(serializer=my_serializer)
    assert f.serializer is my_serializer
    assert f.serializer("json", 123) == "json:123"

    # Test field with all parameters
    f = field(
        type=[int, float],
        invariant=lambda x: (x >= 0, "Must be non-negative"),
        initial=0,
        mandatory=True,
        factory=lambda x: float(x),
        serializer=lambda fmt, val: str(val)
    )
    assert len(f.type) == 2
    assert int in f.type
    assert float in f.type
    assert f.invariant(5) == (True, "Must be non-negative")
    assert f.invariant(-1) == (False, "Must be non-negative")
    assert f.initial == 0
    assert f.mandatory is True
    assert f.factory(5) == 5.0
    assert f.serializer("any", 3.14) == "3.14"

    # Test that non-callable invariant raises TypeError
    try:
        field(invariant="not callable")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test that non-callable factory raises TypeError
    try:
        field(factory="not callable")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test that non-callable serializer raises TypeError
    try:
        field(serializer="not callable")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test that invalid type in type parameter raises TypeError
    try:
        field(type=[int, "not a type", 123])
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test that initial value with wrong type raises TypeError
    try:
        field(type=int, initial="string")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test field with string type specification
    f = field(type="int")
    assert len(f.type) == 1
    # Note: The actual type resolution happens elsewhere, so we just check it's in the set
    assert "int" in f.type

    # Test field with optional type wrapper
    f = field(type=optional_type(int))
    assert len(f.type) == 1
    # optional_type returns a special wrapper that should be in the type set
    assert optional_type(int) in f.type


# LLM-generated content at query #16
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    f = pmap_field(int, str)
    assert isinstance(f, _PField)
    assert f.mandatory is True
    assert f.initial == {}
    assert len(f.type) == 1
    map_type = next(iter(f.type))
    assert issubclass(map_type, CheckedPMap)
    assert map_type.__key_type__ == int
    assert map_type.__value_type__ == str
    
    # Test optional pmap_field
    f_optional = pmap_field(int, str, optional=True)
    assert f_optional.mandatory is True
    assert len(f_optional.type) == 1
    optional_type = next(iter(f_optional.type))
    assert optional_type.__name__ == "Optional"
    
    # Test with invariant
    def custom_invariant(m):
        return (len(m) < 3, "Too many items")
    
    f_with_invariant = pmap_field(int, str, invariant=custom_invariant)
    assert f_with_invariant.invariant is not PFIELD_NO_INVARIANT
    
    # Test factory function for non-optional field
    map_type = _make_pmap_field_type(int, str)
    test_map = {"a": 1, "b": 2}
    result = f.factory(test_map)
    assert isinstance(result, map_type)
    assert dict(result) == test_map
    
    # Test factory function for optional field
    result_none = f_optional.factory(None)
    assert result_none is None
    
    result_map = f_optional.factory(test_map)
    assert isinstance(result_map, map_type)
    
    # Test type name generation
    map_type = _make_pmap_field_type(int, str)
    assert "IntToStrPMap" in map_type.__name__
    
    # Test that same type is reused
    map_type1 = _make_pmap_field_type(int, str)
    map_type2 = _make_pmap_field_type(int, str)
    assert map_type1 is map_type2
    
    # Test with different types
    f2 = pmap_field(str, list)
    map_type2 = next(iter(f2.type))
    assert map_type2.__key_type__ == str
    assert map_type2.__value_type__ == list
    
    # Test initial value
    assert f.initial == {}
    assert isinstance(f.initial, CheckedPMap)
    
    # Test that field is properly configured
    assert f.serializer is PFIELD_NO_SERIALIZER
    assert f.invariant is PFIELD_NO_INVARIANT or callable(f.invariant)


# LLM-generated content at query #17
#--------------------------

```python
def test_serialize():
    # Test 1: Serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}_{self.__class__.__name__}"
    
    mock_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_value)
    assert result == "serialized_json_MockCheckedType"
    
    # Test 2: Serializer is PFIELD_NO_SERIALIZER and value is not CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "json", "test_value")
    assert result == "test_value"
    
    # Test 3: Custom serializer function
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "xml:data"
    
    # Test 4: Serializer with CheckedType but not PFIELD_NO_SERIALIZER
    # Should use custom serializer instead of value.serialize()
    mock_value = MockCheckedType()
    result = serialize(custom_serializer, "yaml", mock_value)
    assert result == "yaml:MockCheckedType"
    
    # Test 5: Serializer with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 6: Serializer with numeric value
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test 7: Serializer with list value
    result = serialize(PFIELD_NO_SERIALIZER, "json", [1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #18
#--------------------------

```python
def test_check_global_invariants():
    # Test with passing invariants
    subject = {"name": "test"}
    invariants = [
        lambda x: (True, None),
        lambda x: (True, "OK"),
        lambda x: (True, 200)
    ]
    check_global_invariants(subject, invariants)
    
    # Test with single failing invariant
    subject = {"value": 5}
    invariants = [
        lambda x: (False, "Value too small")
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("Value too small",)
    
    # Test with multiple failing invariants
    subject = {"value": 10}
    invariants = [
        lambda x: (False, "Error1"),
        lambda x: (True, "OK"),
        lambda x: (False, "Error2"),
        lambda x: (False, "Error3")
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.invariant_errors) == {"Error1", "Error2", "Error3"}
    
    # Test with mixed error codes
    subject = {"data": "test"}
    invariants = [
        lambda x: (False, 404),
        lambda x: (False, "Not found"),
        lambda x: (True, None)
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == (404, "Not found")
    
    # Test with empty invariants list
    subject = {"empty": True}
    invariants = []
    check_global_invariants(subject, invariants)
    
    # Test invariants that use the subject
    subject = {"count": 15}
    invariants = [
        lambda x: (x["count"] > 10, "Count too low"),
        lambda x: (x["count"] < 20, "Count too high")
    ]
    check_global_invariants(subject, invariants)
    
    # Test invariants that use the subject and fail
    subject = {"count": 5}
    invariants = [
        lambda x: (x["count"] > 10, "Count too low"),
        lambda x: (x["count"] < 20, "Count too high")
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("Count too low",)


# LLM-generated content at query #19
#--------------------------

```python
def test_serialize():
    # Test with non-CheckedType value and default serializer
    result = serialize(PFIELD_NO_SERIALIZER, 'json', 'test_value')
    assert result == 'test_value'
    
    # Test with custom serializer
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, 'json', 'test')
    assert result == 'json:test'
    
    # Test with CheckedType value and default serializer
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}"
    
    checked_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, 'json', checked_value)
    assert result == 'serialized_json'
    
    # Test with CheckedType value and custom serializer
    # Custom serializer should be used even for CheckedType
    result = serialize(custom_serializer, 'xml', checked_value)
    assert result == 'xml:checked_value'
    
    # Test with None value
    result = serialize(PFIELD_NO_SERIALIZER, 'json', None)
    assert result is None
    
    # Test with integer value
    result = serialize(PFIELD_NO_SERIALIZER, 'json', 42)
    assert result == 42
    
    # Test with list value
    result = serialize(PFIELD_NO_SERIALIZER, 'json', [1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #20
#--------------------------

```python
def test_serialize():
    # Test 1: serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}_{self.__class__.__name__}"
    
    mock_checked = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_checked)
    assert result == "serialized_json_MockCheckedType"
    
    # Test 2: serializer is PFIELD_NO_SERIALIZER and value is not CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "json", "regular_value")
    assert result == "regular_value"
    
    # Test 3: custom serializer function
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "xml:data"
    
    # Test 4: serializer with CheckedType value (should use custom serializer, not CheckedType.serialize)
    result = serialize(custom_serializer, "xml", mock_checked)
    assert result == "xml:MockCheckedType"
    
    # Test 5: serializer with None value
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 6: serializer with numeric value
    result = serialize(PFIELD_NO_SERIALIZER, "binary", 42)
    assert result == 42
    
    # Test 7: serializer with list value
    result = serialize(PFIELD_NO_SERIALIZER, "yaml", [1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #21
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field1 = pmap_field(int, str)
    assert isinstance(field1, _PField)
    assert field1.mandatory is True
    assert isinstance(field1.initial, CheckedPMap)
    assert field1.type == {field1.factory({})}
    
    # Test optional pmap_field
    field2 = pmap_field(int, str, optional=True)
    assert isinstance(field2, _PField)
    assert field2.mandatory is True
    assert field2.initial == field2.factory({})
    assert len(field2.type) == 1
    
    # Test factory function for optional field
    assert field2.factory(None) is None
    test_map = {1: "a", 2: "b"}
    created_map = field2.factory(test_map)
    assert isinstance(created_map, CheckedPMap)
    assert dict(created_map) == test_map
    
    # Test factory function for non-optional field
    created_map_non_optional = field1.factory(test_map)
    assert isinstance(created_map_non_optional, CheckedPMap)
    assert dict(created_map_non_optional) == test_map
    
    # Test with invariant
    def custom_invariant(value):
        return (len(value) <= 2, "Too many items")
    
    field3 = pmap_field(int, str, invariant=custom_invariant)
    assert field3.invariant is not PFIELD_NO_INVARIANT
    
    # Test that invariant works
    valid_result = field3.invariant(field3.factory({1: "a", 2: "b"}))
    assert valid_result == (True, None)
    
    # Test type checking through factory
    TheMap = field1.factory({})
    assert isinstance(TheMap, CheckedPMap)
    
    # Test that different type combinations create different field types
    field4 = pmap_field(str, int)
    field5 = pmap_field(str, str)
    assert field4.type != field5.type
    
    # Test that same type combination returns same field type
    field6 = pmap_field(int, str)
    assert field1.factory({}).__class__ == field6.factory({}).__class__
    
    # Test with custom invariant that fails
    def failing_invariant(value):
        return (False, "Always fails")
    
    field7 = pmap_field(int, str, invariant=failing_invariant)
    result = field7.invariant(field7.factory({}))
    assert result == (False, "Always fails")


# LLM-generated content at query #22
#--------------------------

```python
def test_check_global_invariants():
    # Test case 1: All invariants pass
    subject = {"name": "test", "value": 42}
    invariants = [
        lambda x: (True, None),
        lambda x: (True, "OK"),
        lambda x: (True, 200)
    ]
    # Should not raise any exception
    check_global_invariants(subject, invariants)
    
    # Test case 2: Single invariant fails
    subject = {"data": "invalid"}
    invariants = [
        lambda x: (False, "ERROR_001")
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("ERROR_001",)
        assert e.invariant_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test case 3: Multiple invariants fail
    subject = {"value": -5}
    invariants = [
        lambda x: (False, "NEGATIVE"),
        lambda x: (True, "POSITIVE"),
        lambda x: (False, "TOO_SMALL"),
        lambda x: (False, "INVALID_TYPE")
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.error_codes) == {"NEGATIVE", "TOO_SMALL", "INVALID_TYPE"}
        assert len(e.error_codes) == 3
    
    # Test case 4: Mixed results with various error code types
    subject = object()
    invariants = [
        lambda x: (True, None),
        lambda x: (False, 404),
        lambda x: (False, "Not Found"),
        lambda x: (True, ""),
        lambda x: (False, ("err", "code"))
    ]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.error_codes == (404, "Not Found", ("err", "code"))
    
    # Test case 5: Empty invariants list
    subject = {"any": "thing"}
    invariants = []
    # Should not raise any exception
    check_global_invariants(subject, invariants)
    
    # Test case 6: All invariants pass with complex subject
    class ComplexSubject:
        def __init__(self):
            self.data = [1, 2, 3]
            self.name = "test"
    
    subject = ComplexSubject()
    invariants = [
        lambda x: (hasattr(x, 'data'), "NO_DATA"),
        lambda x: (hasattr(x, 'name'), "NO_NAME"),
        lambda x: (len(x.data) == 3, "WRONG_SIZE")
    ]
    check_global_invariants(subject, invariants)


# LLM-generated content at query #23
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = "test"
    invariants = []
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with passing invariants
    def invariant1(x):
        return (True, None)
    
    def invariant2(x):
        return (True, "OK")
    
    invariants = [invariant1, invariant2]
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with single failing invariant
    def failing_invariant(x):
        return (False, "error1")
    
    invariants = [failing_invariant]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("error1",)
        assert e.field_errors == ()

    # Test with multiple failing invariants
    def failing_invariant1(x):
        return (False, "error1")
    
    def failing_invariant2(x):
        return (False, "error2")
    
    def passing_invariant(x):
        return (True, "OK")
    
    invariants = [failing_invariant1, passing_invariant, failing_invariant2]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.invariant_errors) == {"error1", "error2"}
        assert e.field_errors == ()

    # Test with mixed invariants and error codes
    def invariant_with_none(x):
        return (False, None)
    
    def invariant_with_code(x):
        return (False, "code1")
    
    invariants = [invariant_with_none, invariant_with_code]
    try:
        check_global_invariants(subject, invariants)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert None in e.invariant_errors
        assert "code1" in e.invariant_errors
        assert len(e.invariant_errors) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    from unittest.mock import Mock
    import inspect
    
    # Mock field with type that is a set
    mock_field_set_type = Mock()
    mock_field_set_type.type = {Mock()}
    mock_field_set_type.factory = Mock()
    
    # Test 1: ignore_extra is False, should return False regardless of other conditions
    assert not is_field_ignore_extra_complaint(CheckedPVector, mock_field_set_type, False)
    
    # Test 2: ignore_extra is True but field type is not a sequence type
    mock_field_non_seq = Mock()
    mock_field_non_seq.type = {int}
    mock_field_non_seq.factory = Mock()
    assert not is_field_ignore_extra_complaint(CheckedPVector, mock_field_non_seq, True)
    
    # Test 3: ignore_extra is True, field type is sequence type, factory has ignore_extra parameter
    mock_field_with_param = Mock()
    mock_field_with_param.type = {Mock()}
    mock_field_with_param.factory = Mock()
    mock_signature = Mock()
    mock_signature.parameters = {'ignore_extra': Mock()}
    mock_field_with_param.factory.signature = mock_signature
    with unittest.mock.patch('inspect.signature', return_value=mock_signature):
        assert is_field_ignore_extra_complaint(CheckedPVector, mock_field_with_param, True)
    
    # Test 4: ignore_extra is True, field type is sequence type, factory does NOT have ignore_extra parameter
    mock_field_no_param = Mock()
    mock_field_no_param.type = {Mock()}
    mock_field_no_param.factory = Mock()
    mock_signature_no_param = Mock()
    mock_signature_no_param.parameters = {'other_param': Mock()}
    with unittest.mock.patch('inspect.signature', return_value=mock_signature_no_param):
        assert not is_field_ignore_extra_complaint(CheckedPVector, mock_field_no_param, True)
    
    # Test 5: field.type is empty set
    mock_field_empty_type = Mock()
    mock_field_empty_type.type = set()
    mock_field_empty_type.factory = Mock()
    assert not is_field_ignore_extra_complaint(CheckedPVector, mock_field_empty_type, True)
    
    # Test 6: field.type is tuple with one element
    mock_field_tuple_type = Mock()
    mock_field_tuple_type.type = (Mock(),)
    mock_field_tuple_type.factory = Mock()
    mock_signature_with_param = Mock()
    mock_signature_with_param.parameters = {'ignore_extra': Mock()}
    with unittest.mock.patch('inspect.signature', return_value=mock_signature_with_param):
        assert is_field_ignore_extra_complaint(CheckedPVector, mock_field_tuple_type, True)


# LLM-generated content at query #25
#--------------------------

```python
def test_serialize():
    # Test 1: Serializer is PFIELD_NO_SERIALIZER and value is CheckedType
    class MockCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}_{self.__class__.__name__}"
    
    mock_value = MockCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "json", mock_value)
    assert result == "serialized_json_MockCheckedType"
    
    # Test 2: Serializer is PFIELD_NO_SERIALIZER and value is not CheckedType
    regular_value = "test_string"
    result = serialize(PFIELD_NO_SERIALIZER, "json", regular_value)
    assert result == regular_value
    
    # Test 3: Custom serializer function
    def custom_serializer(format, value):
        return f"{format}:{value}"
    
    result = serialize(custom_serializer, "xml", "data")
    assert result == "xml:data"
    
    # Test 4: Serializer with CheckedType value but custom serializer provided
    result = serialize(custom_serializer, "yaml", mock_value)
    assert result == "yaml:MockCheckedType"
    
    # Test 5: None value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", None)
    assert result is None
    
    # Test 6: Integer value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", 42)
    assert result == 42
    
    # Test 7: List value with PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "json", [1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #26
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = {"key": "value"}
    check_global_invariants(subject, [])
    
    # Test with passing invariants
    def invariant1(x):
        return True, None
    
    def invariant2(x):
        return True, "OK"
    
    check_global_invariants(subject, [invariant1, invariant2])
    
    # Test with single failing invariant
    def failing_invariant(x):
        return False, "ERROR_1"
    
    try:
        check_global_invariants(subject, [failing_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("ERROR_1",)
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with multiple failing invariants
    def failing_invariant2(x):
        return False, "ERROR_2"
    
    def failing_invariant3(x):
        return False, "ERROR_3"
    
    try:
        check_global_invariants(subject, [invariant1, failing_invariant, failing_invariant2, failing_invariant3])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.invariant_errors) == {"ERROR_1", "ERROR_2", "ERROR_3"}
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with mixed passing and failing invariants
    def failing_invariant4(x):
        return False, "ERROR_4"
    
    try:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2, failing_invariant4])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert set(e.invariant_errors) == {"ERROR_1", "ERROR_4"}
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"
    
    # Test with different subject types
    class TestSubject:
        def __init__(self, value):
            self.value = value
    
    test_obj = TestSubject(42)
    
    def obj_invariant(obj):
        return obj.value > 0, "VALUE_NOT_POSITIVE"
    
    check_global_invariants(test_obj, [obj_invariant])
    
    def failing_obj_invariant(obj):
        return obj.value < 0, "VALUE_NOT_NEGATIVE"
    
    try:
        check_global_invariants(test_obj, [failing_obj_invariant])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("VALUE_NOT_NEGATIVE",)
        assert e.field_errors == ()
        assert str(e) == "Global invariant failed"


