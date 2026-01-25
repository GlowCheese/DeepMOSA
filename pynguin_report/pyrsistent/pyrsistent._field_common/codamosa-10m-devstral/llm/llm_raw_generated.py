####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_set_fields():
    # Test basic field collection from bases
    class Base1:
        __fields__ = {'a': 1}

    class Base2:
        __fields__ = {'b': 2}

    class Child(Base1, Base2):
        pass

    dct = {'__fields__': {}, 'c': 3}
    bases = (Base1, Base2)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct[name] == {'a': 1, 'b': 2}
    assert 'c' not in dct

    # Test field collection with _PField instances
    field_a = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)
    field_b = _PField(type=str, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)

    class Base3:
        __fields__ = {'a': field_a}

    class Base4:
        __fields__ = {'b': field_b}

    class Child2(Base3, Base4):
        pass

    dct2 = {'__fields__': {}, 'c': field_a}
    bases2 = (Base3, Base4)
    name2 = '__fields__'

    set_fields(dct2, bases2, name2)

    assert dct2[name2] == {'a': field_a, 'b': field_b}
    assert 'c' not in dct2

    # Test with no bases
    dct3 = {'__fields__': {}, 'a': 1}
    bases3 = ()
    name3 = '__fields__'

    set_fields(dct3, bases3, name3)

    assert dct3[name3] == {}
    assert 'a' in dct3

    # Test with no existing __fields__ in bases
    class Base5:
        pass

    class Base6:
        pass

    dct4 = {'__fields__': {}, 'a': 1}
    bases4 = (Base5, Base6)
    name4 = '__fields__'

    set_fields(dct4, bases4, name4)

    assert dct4[name4] == {}
    assert 'a' in dct4


# LLM-generated content at query #2
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_spec = pmap_field(str, int)
    assert field_spec.type == {_make_pmap_field_type(str, int)}
    assert field_spec.mandatory is True
    assert field_spec.initial == _make_pmap_field_type(str, int)()
    assert field_spec.factory is _make_pmap_field_type(str, int).create
    assert field_spec.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_spec = pmap_field(str, int, optional=True)
    assert field_spec.type == {_make_pmap_field_type(str, int), type(None)}
    assert field_spec.mandatory is True
    assert field_spec.initial == _make_pmap_field_type(str, int)()
    assert callable(field_spec.factory)
    assert field_spec.factory(None) is None
    assert field_spec.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 10 else (False, "Too many items")

    field_spec = pmap_field(str, int, invariant=custom_invariant)
    assert field_spec.invariant == custom_invariant

    # Test factory behavior
    field_spec = pmap_field(str, int, optional=True)
    assert field_spec.factory({"a": 1, "b": 2}) == _make_pmap_field_type(str, int)({"a": 1, "b": 2})
    assert field_spec.factory(None) is None

    # Test that the field type is correctly created and cached
    type1 = _make_pmap_field_type(str, int)
    type2 = _make_pmap_field_type(str, int)
    assert type1 is type2
    assert type1.__name__ == "StrToIntPMap"

    # Test serialization/deserialization
    pmap = type1({"key": 42})
    serialized = pmap.__reduce__()
    assert serialized[0] == _restore_pmap_field_pickle
    assert serialized[1] == (str, int, {"key": 42})
    restored = _restore_pmap_field_pickle(str, int, {"key": 42})
    assert restored == pmap


# LLM-generated content at query #3
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    result = serialize(custom_serializer, "format", "test_value")
    assert result == "serialized_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "format", "test_value")
    assert result == "test_value"


# LLM-generated content at query #4
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(int, str)
    assert isinstance(field_obj, _PField)
    assert field_obj.type == {_make_pmap_field_type(int, str)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(int, str)()
    assert field_obj.factory == _make_pmap_field_type(int, str).create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj_optional = pmap_field(int, str, optional=True)
    assert field_obj_optional.type == {optional_type(_make_pmap_field_type(int, str))}
    assert field_obj_optional.factory(None) is None
    assert field_obj_optional.factory({1: "a"}) == _make_pmap_field_type(int, str).create({1: "a"})

    # Test with custom invariant
    custom_invariant = lambda x: (True, None)
    field_obj_inv = pmap_field(int, str, invariant=custom_invariant)
    assert field_obj_inv.invariant == custom_invariant

    # Test factory behavior
    test_map = {1: "a", 2: "b"}
    factory_result = field_obj.factory(test_map)
    assert isinstance(factory_result, CheckedPMap)
    assert dict(factory_result) == test_map

    # Test initial value
    initial_map = field_obj.initial
    assert isinstance(initial_map, CheckedPMap)
    assert dict(initial_map) == {}

    # Test that the created type is cached
    type1 = _make_pmap_field_type(int, str)
    type2 = _make_pmap_field_type(int, str)
    assert type1 is type2

    # Test type checking
    with pytest.raises(PTypeError):
        field_obj.factory({"a": 1})  # Wrong key type

    with pytest.raises(PTypeError):
        field_obj.factory({1: 1})  # Wrong value type

    # Test optional field with None
    optional_field = pmap_field(int, str, optional=True)
    assert optional_field.factory(None) is None


# LLM-generated content at query #5
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, 'format', checked_value)
    assert result == checked_value.serialize('format')

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}_{fmt}"
    result = serialize(custom_serializer, 'json', 'test_value')
    assert result == "serialized_test_value_json"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, 'format', 'plain_value')
    assert result == 'plain_value'


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}_{format}"

    result = serialize(custom_serializer, "json", "data")
    assert result == "serialized_data_json"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "format", "plain_value")
    assert result == "plain_value"


# LLM-generated content at query #7
#--------------------------

```python
def test_serialize():
    # Test with a CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with a non-CheckedType and a custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    assert serialize(custom_serializer, "format", "test_value") == "serialized_test_value"

    # Test with a non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "test_value") == "test_value"


# LLM-generated content at query #8
#--------------------------

```python
def test_set_fields():
    # Test basic field collection from base classes
    class Base1:
        __fields__ = {'a': 'field1'}

    class Base2:
        __fields__ = {'b': 'field2'}

    class Child(Base1, Base2):
        pass

    dct = {}
    set_fields(dct, (Base1, Base2), '__fields__')
    assert dct['__fields__'] == {'a': 'field1', 'b': 'field2'}

    # Test field override in child class
    class ChildOverride(Base1, Base2):
        __fields__ = {'a': 'override'}

    dct = {}
    set_fields(dct, (Base1, Base2), '__fields__')
    assert dct['__fields__'] == {'a': 'override', 'b': 'field2'}

    # Test _PField handling
    field1 = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, x: x)
    field2 = _PField(type=str, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, x: x)

    class BaseWithPField:
        __fields__ = {'field1': field1}

    class ChildWithPField(BaseWithPField):
        field2 = field2

    dct = {'field2': field2}
    set_fields(dct, (BaseWithPField,), '__fields__')
    assert dct['__fields__'] == {'field1': field1, 'field2': field2}
    assert 'field2' not in dct

    # Test empty base classes
    class EmptyChild:
        pass

    dct = {}
    set_fields(dct, (), '__fields__')
    assert dct['__fields__'] == {}


# LLM-generated content at query #9
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj, _PField)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.mandatory is True

    # Test pmap_field with invariant
    def test_invariant(pmap):
        return (True, None)
    field_obj = pmap_field(str, int, invariant=test_invariant)
    assert field_obj.invariant == test_invariant

    # Test pmap_field factory
    field_obj = pmap_field(str, int)
    test_map = field_obj.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)
    assert test_map == {"a": 1, "b": 2}

    # Test optional pmap_field factory with None
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None

    # Test pmap_field with initial value
    field_obj = pmap_field(str, int, initial={"x": 10})
    assert field_obj.initial == {"x": 10}

    # Test pmap_field type checking
    field_obj = pmap_field(str, int)
    test_map = field_obj.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)
    try:
        test_map = field_obj.factory({"a": "not_int"})
        assert False, "Should have raised an error"
    except:
        pass

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        if len(pmap) > 2:
            return (False, "Too many items")
        return (True, None)
    field_obj = pmap_field(str, int, invariant=custom_invariant)
    test_map = field_obj.factory({"a": 1, "b": 2})
    try:
        test_map = field_obj.factory({"a": 1, "b": 2, "c": 3})
        assert False, "Should have raised an error"
    except InvariantException:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def invariant1(obj):
        return (True, None)

    def invariant2(obj):
        return (True, None)

    check_global_invariants(subject, [invariant1, invariant2])

    # Test with failing invariant
    def failing_invariant(obj):
        return (False, "error_code")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant])

    assert "error_code" in str(excinfo.value)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "error_code2")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert "error_code" in str(excinfo.value)
    assert "error_code2" in str(excinfo.value)

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert "error_code" in str(excinfo.value)


# LLM-generated content at query #11
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPMap({"key": "value"})
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    non_checked_value = "test"
    assert serialize(PFIELD_NO_SERIALIZER, "format", non_checked_value) == PFIELD_NO_SERIALIZER("format", non_checked_value)

    # Test with custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"

    assert serialize(custom_serializer, "format", "value") == "custom_format_value"


# LLM-generated content at query #12
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"{format}:{value}"

    result = serialize(custom_serializer, "json", 42)
    assert result == "json:42"

    # Test with PFIELD_NO_SERIALIZER and non-CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "format", "test")
    assert result == "test"


# LLM-generated content at query #13
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)

    # Test with multiple matching types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "hello")

    # Test with non-matching type
    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "not_an_int")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "any_type_is_ok")

    # Test with subclass type
    class Parent:
        pass

    class Child(Parent):
        pass

    field = _PField(type={Parent}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", Child())


# LLM-generated content at query #14
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert callable(field_obj.factory)
    assert callable(field_obj.invariant)

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert callable(field_obj.factory)
    assert callable(field_obj.invariant)

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 10 else (False, "Too many items")

    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test factory with None for optional field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert isinstance(field_obj.factory({"a": 1}), _make_pmap_field_type(str, int))

    # Test factory with non-None value
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj.factory({"a": 1}), _make_pmap_field_type(str, int))

    # Test with multiple types
    field_obj = pmap_field((str, int), (int, str))
    assert field_obj.type == {_make_pmap_field_type((str, int), (int, str))}


# LLM-generated content at query #15
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test case 1: ignore_extra is False, should return False
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set, factory=lambda x: x), False) is False

    # Test case 2: ignore_extra is True but field type is not a set, should return False
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=lambda x: x), True) is False

    # Test case 3: ignore_extra is True, field type is a set, but factory doesn't have ignore_extra parameter, should return False
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set, factory=lambda x: x), True) is False

    # Test case 4: ignore_extra is True, field type is a set, and factory has ignore_extra parameter, should return True
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set, factory=factory_with_ignore_extra), True) is True

    # Test case 5: ignore_extra is True, field type is a tuple with a set, and factory has ignore_extra parameter, should return True
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=(set,), factory=factory_with_ignore_extra), True) is True

    # Test case 6: ignore_extra is True, field type is a list with a set, and factory has ignore_extra parameter, should return True
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=[set], factory=factory_with_ignore_extra), True) is True


# LLM-generated content at query #16
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)

    # Test with multiple matching types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "hello")

    # Test with non-matching type
    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "not_an_int")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "anything_goes")

    # Test with CheckedType subclass
    class TestCheckedType(CheckedType):
        pass

    field = _PField(type={TestCheckedType}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", TestCheckedType())


# LLM-generated content at query #17
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"

    result = serialize(custom_serializer, "json", "test_value")
    assert result == "custom_json_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "format", "plain_value")
    assert result == "plain_value"


# LLM-generated content at query #18
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test when ignore_extra is False
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=lambda x: x), False)

    # Test when ignore_extra is True but field type is not a set
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=lambda x: x), True)

    # Test when ignore_extra is True and field type is a set but factory doesn't have ignore_extra parameter
    assert not is_field_ignore_extra_complaint(CheckedPSet, _PField(type={int}, factory=lambda x: x), True)

    # Test when ignore_extra is True, field type is a set, and factory has ignore_extra parameter
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    assert is_field_ignore_extra_complaint(CheckedPSet, _PField(type={int}, factory=factory_with_ignore_extra), True)

    # Test with CheckedPVector
    assert is_field_ignore_extra_complaint(CheckedPVector, _PField(type={int}, factory=factory_with_ignore_extra), True)


# LLM-generated content at query #19
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}_{format}"

    result = serialize(custom_serializer, "json", "test_value")
    assert result == "serialized_test_value_json"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "format", "plain_value")
    assert result == "plain_value"


# LLM-generated content at query #20
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert isinstance(field_obj.initial, CheckedPMap)
    assert field_obj.factory is _make_pmap_field_type(str, int).create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert optional_field.factory(None) is None
    assert isinstance(optional_field.factory({}), CheckedPMap)

    # Test with custom invariant
    custom_inv = lambda x: (True, "") if len(x) < 5 else (False, "Too many items")
    field_with_inv = pmap_field(str, int, invariant=custom_inv)
    assert field_with_inv.invariant == custom_inv

    # Test field creation with the factory
    test_map = field_obj.factory({"a": 1, "b": 2})
    assert test_map["a"] == 1
    assert test_map["b"] == 2

    # Test that invalid types raise errors
    with pytest.raises(PTypeError):
        field_obj.factory({"a": "not an int"})

    # Test that the created map enforces types
    with pytest.raises(InvariantException):
        field_obj.factory({"a": 1, 123: 2})  # key is not str

    # Test initial value
    assert len(field_obj.initial) == 0
    assert field_obj.initial == field_obj.factory({})


# LLM-generated content at query #21
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj, _PField)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.factory == _make_pmap_field_type(str, int).create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test pmap_field with custom invariant
    custom_invariant = lambda x: (True, "")
    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test pmap_field with multiple key/value types
    field_obj = pmap_field((str, int), (float, bool))
    assert field_obj.type == {_make_pmap_field_type((str, int), (float, bool))}

    # Test pmap_field factory with initial value
    field_obj = pmap_field(str, int, optional=True, initial={"a": 1})
    assert field_obj.initial == _make_pmap_field_type(str, int).create({"a": 1})

    # Test pmap_field with None initial value when optional
    field_obj = pmap_field(str, int, optional=True, initial=None)
    assert field_obj.initial is None

    # Test pmap_field with invalid types
    with pytest.raises(TypeError):
        pmap_field("not_a_type", int)

    with pytest.raises(TypeError):
        pmap_field(str, "not_a_type")

    # Test pmap_field with callable invariant
    def test_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")
    field_obj = pmap_field(str, int, invariant=test_invariant)
    assert field_obj.invariant({"a": 1}) == (True, "")
    assert field_obj.invariant({}) == (False, "Map must not be empty")

    # Test pmap_field with non-callable invariant
    with pytest.raises(TypeError):
        pmap_field(str, int, invariant="not_callable")


# LLM-generated content at query #22
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    result = serialize(custom_serializer, "format", "test_value")
    assert result == "serialized_test_value"

    # Test with PFIELD_NO_SERIALIZER and non-CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "format", "test_value")
    assert result == "test_value"


# LLM-generated content at query #23
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}_{format}"

    assert serialize(custom_serializer, "json", "test_value") == "serialized_test_value_json"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "plain_value") == "plain_value"


# LLM-generated content at query #24
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.mandatory is True

    # Test with invariant
    def test_invariant(pmap):
        return (True, None)
    field_obj = pmap_field(str, int, invariant=test_invariant)
    assert field_obj.invariant == test_invariant

    # Test factory behavior
    field_obj = pmap_field(str, int)
    test_map = field_obj.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)
    assert test_map["a"] == 1
    assert test_map["b"] == 2

    # Test optional factory with None
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1})["a"] == 1

    # Test initial value
    field_obj = pmap_field(str, int, initial={"x": 10})
    initial_map = field_obj.initial
    assert initial_map["x"] == 10

    # Test type checking
    field_obj = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field_obj, "test_field", {"a": "not_int"})

    # Test with custom invariant that fails
    def failing_invariant(pmap):
        return (False, "Test error")
    field_obj = pmap_field(str, int, invariant=failing_invariant)
    with pytest.raises(InvariantException):
        check_global_invariants({"a": 1}, [field_obj.invariant])


# LLM-generated content at query #25
#--------------------------

```python
def test_serialize():
    # Test with a CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with a non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    assert serialize(custom_serializer, "format", "test_value") == "serialized_test_value"

    # Test with a CheckedType and custom serializer (should use custom serializer)
    assert serialize(custom_serializer, "format", checked_value) == custom_serializer("format", checked_value)


# LLM-generated content at query #26
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test case 1: ignore_extra is False
    field = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, value: value)
    assert not is_field_ignore_extra_complaint(int, field, False)

    # Test case 2: ignore_extra is True but field type is not a set
    field = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, value: value)
    assert not is_field_ignore_extra_complaint(int, field, True)

    # Test case 3: ignore_extra is True, field type is a set, but factory doesn't have ignore_extra parameter
    field = _PField(type={int}, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, value: value)
    assert not is_field_ignore_extra_complaint(int, field, True)

    # Test case 4: ignore_extra is True, field type is a set, and factory has ignore_extra parameter
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    field = _PField(type={int}, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=factory_with_ignore_extra, serializer=lambda _, value: value)
    assert is_field_ignore_extra_complaint(int, field, True)

    # Test case 5: ignore_extra is True, field type is a tuple with one element (set), and factory has ignore_extra parameter
    field = _PField(type=(set,), invariant=lambda x: (True, None), initial=0, mandatory=False, factory=factory_with_ignore_extra, serializer=lambda _, value: value)
    assert is_field_ignore_extra_complaint(int, field, True)


# LLM-generated content at query #27
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}_{format}"

    assert serialize(custom_serializer, "json", "test_value") == "serialized_test_value_json"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "plain_value") == "plain_value"


# LLM-generated content at query #28
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert isinstance(field_obj.initial, CheckedPMap)
    assert field_obj.factory is _make_pmap_field_type(str, int).create

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.mandatory is True
    assert field_obj.initial is None
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1}) == _make_pmap_field_type(str, int)({"a": 1})

    # Test with invariant
    def test_invariant(pmap):
        return (True, None) if len(pmap) < 2 else (False, "Too many items")

    field_obj = pmap_field(str, int, invariant=test_invariant)
    assert field_obj.invariant({"a": 1}) == (True, None)
    assert field_obj.invariant({"a": 1, "b": 2}) == (False, "Too many items")

    # Test factory with None when optional
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1}) == _make_pmap_field_type(str, int)({"a": 1})

    # Test initial value
    field_obj = pmap_field(str, int, optional=True, initial=None)
    assert field_obj.initial is None

    # Test type checking
    with pytest.raises(PTypeError):
        field_obj = pmap_field(str, int)
        check_type(type('TestClass', (), {}), field_obj, "test_field", {"a": "not_int"})


# LLM-generated content at query #29
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj, _PField)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.factory is not PFIELD_NO_FACTORY
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {optional_type(_make_pmap_field_type(str, int))}

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    field_with_inv = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_inv.invariant == custom_invariant

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    factory = field_obj.factory
    result = factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test optional factory with None
    optional_factory = optional_field.factory
    assert optional_factory(None) is None
    assert isinstance(optional_factory(test_map), CheckedPMap)

    # Test type checking
    with pytest.raises(PTypeError):
        check_type(type, field_obj, "test_field", [1, 2, 3])

    # Test that the field works in a class context
    class TestRecord:
        test_field = pmap_field(str, int)

    record = TestRecord()
    assert hasattr(record, 'test_field')
    assert isinstance(record.test_field, CheckedPMap)


# LLM-generated content at query #30
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field1 = pmap_field(str, int)
    assert field1.type == {_pmap_field_types[(str, int)]}
    assert field1.mandatory is True
    assert field1.initial == _pmap_field_types[(str, int)]()
    assert field1.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field2 = pmap_field(str, int, optional=True)
    assert field2.type == {_pmap_field_types[(str, int)], type(None)}
    assert field2.mandatory is True
    assert field2.initial == _pmap_field_types[(str, int)]()
    assert field2.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    custom_inv = lambda x: (True, "")
    field3 = pmap_field(str, int, invariant=custom_inv)
    assert field3.invariant == custom_inv

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    result = field1.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test optional factory with None
    result_none = field2.factory(None)
    assert result_none is None

    # Test initial value
    initial_map = field1.initial
    assert isinstance(initial_map, CheckedPMap)
    assert dict(initial_map) == {}

    # Test type checking
    with pytest.raises(PTypeError):
        check_type(str, field1, "test_field", 123)

    # Test that the same key/value types return the same field type
    field4 = pmap_field(str, int)
    assert field1.type == field4.type


# LLM-generated content at query #31
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert isinstance(field, _PField)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert isinstance(field.initial, CheckedPMap)
    assert field.initial == _make_pmap_field_type(str, int)()

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.initial is None

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test factory behavior
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert test_map == _make_pmap_field_type(str, int)({"a": 1, "b": 2})

    # Test optional factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int)({"a": 1})

    # Test type checking
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)

    # Test that invalid types are rejected
    with pytest.raises(PTypeError):
        field = pmap_field(str, int)
        test_map = field.factory({"a": 1, 2: "b"})  # int key should fail

    # Test that the field can be used in a class
    class TestRecord:
        test_field = pmap_field(str, int)

    record = TestRecord()
    assert isinstance(record.test_field, _PField)

    # Test serialization
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    serialized = field.serializer("json", test_map)
    assert serialized == test_map.serialize("json")


# LLM-generated content at query #32
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    result = serialize(custom_serializer, "format", "test_value")
    assert result == "serialized_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "format", "test_value")
    assert result == "test_value"


# LLM-generated content at query #33
#--------------------------

```python
def test_check_global_invariants():
    # Test that no exception is raised when all invariants pass
    subject = object()
    invariants = [lambda x: (True, None), lambda x: (True, None)]
    check_global_invariants(subject, invariants)

    # Test that InvariantException is raised when one invariant fails
    subject = object()
    invariants = [lambda x: (True, None), lambda x: (False, "error_code")]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.args[0] == ("error_code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'

    # Test that InvariantException is raised with multiple error codes
    subject = object()
    invariants = [lambda x: (False, "error1"), lambda x: (False, "error2")]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.args[0] == ("error1", "error2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'

    # Test that no exception is raised when invariants list is empty
    subject = object()
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #34
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def invariant1(obj):
        return (True, None)
    def invariant2(obj):
        return (True, None)
    check_global_invariants(subject, [invariant1, invariant2])

    # Test with failing invariant
    def failing_invariant(obj):
        return (False, "Error code")
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant])
    assert excinfo.value.error_codes == ("Error code",)
    assert excinfo.value.message == 'Global invariant failed'

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "Error code 2")
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])
    assert excinfo.value.error_codes == ("Error code", "Error code 2")
    assert excinfo.value.message == 'Global invariant failed'


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPMap()
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    non_checked_value = "test"
    assert serialize(PFIELD_NO_SERIALIZER, "format", non_checked_value) == "format"

    # Test with custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"

    assert serialize(custom_serializer, "format", "value") == "custom_format_value"


# LLM-generated content at query #2
#--------------------------

```python
def test_set_fields():
    # Test basic field collection from base classes
    class Base1:
        __fields__ = {'field1': 'value1'}

    class Base2:
        __fields__ = {'field2': 'value2'}

    class Child(Base1, Base2):
        pass

    dct = {'__fields__': {}}
    set_fields(dct, (Base1, Base2), '__fields__')
    assert dct['__fields__'] == {'field1': 'value1', 'field2': 'value2'}

    # Test field overwriting
    class Base3:
        __fields__ = {'field1': 'base_value'}

    class Base4:
        __fields__ = {'field1': 'overwritten_value'}

    class Child2(Base3, Base4):
        pass

    dct = {'__fields__': {}}
    set_fields(dct, (Base3, Base4), '__fields__')
    assert dct['__fields__'] == {'field1': 'overwritten_value'}

    # Test _PField handling
    class Base5:
        __fields__ = {'field1': 'value1'}

    class Child3(Base5):
        field2 = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)

    dct = {'__fields__': {}, 'field2': Child3.field2}
    set_fields(dct, (Base5,), '__fields__')
    assert dct['__fields__'] == {'field1': 'value1', 'field2': Child3.field2}
    assert 'field2' not in dct

    # Test empty base classes
    class Child4:
        pass

    dct = {'__fields__': {}}
    set_fields(dct, (), '__fields__')
    assert dct['__fields__'] == {}

    # Test no existing __fields__ in bases
    class Base6:
        pass

    class Child5(Base6):
        pass

    dct = {'__fields__': {}}
    set_fields(dct, (Base6,), '__fields__')
    assert dct['__fields__'] == {}


# LLM-generated content at query #3
#--------------------------

```python
def test_set_fields():
    # Test basic field collection from base classes
    class Base1:
        __fields__ = {'a': 1}

    class Base2:
        __fields__ = {'b': 2}

    class Child(Base1, Base2):
        pass

    dct = {'__fields__': {}}
    bases = (Base1, Base2)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct[name] == {'a': 1, 'b': 2}

    # Test field collection with _PField instances
    field1 = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)
    field2 = _PField(type=str, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)

    class Base3:
        __fields__ = {'c': field1}

    class Base4:
        __fields__ = {'d': field2}

    class Child2(Base3, Base4):
        pass

    dct2 = {'__fields__': {}, 'c': field1, 'd': field2}
    bases2 = (Base3, Base4)
    name2 = '__fields__'

    set_fields(dct2, bases2, name2)

    assert dct2[name2] == {'c': field1, 'd': field2}
    assert 'c' not in dct2
    assert 'd' not in dct2

    # Test with no fields in base classes
    class EmptyBase:
        pass

    dct3 = {'__fields__': {}}
    bases3 = (EmptyBase,)
    name3 = '__fields__'

    set_fields(dct3, bases3, name3)

    assert dct3[name3] == {}

    # Test with overlapping fields (last base class wins)
    class Base5:
        __fields__ = {'e': 1}

    class Base6:
        __fields__ = {'e': 2}

    class Child3(Base5, Base6):
        pass

    dct4 = {'__fields__': {}}
    bases4 = (Base5, Base6)
    name4 = '__fields__'

    set_fields(dct4, bases4, name4)

    assert dct4[name4] == {'e': 2}


# LLM-generated content at query #4
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    invariants = []
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with passing invariants
    def passing_invariant(obj):
        return (True, None)

    invariants = [passing_invariant]
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with failing invariant
    def failing_invariant(obj):
        return (False, "Error code 1")

    invariants = [failing_invariant]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("Error code 1",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "Error code 2")

    invariants = [failing_invariant, failing_invariant2]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("Error code 1", "Error code 2")

    # Test with mixed passing and failing invariants
    invariants = [passing_invariant, failing_invariant, passing_invariant]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert exc_info.value.error_codes == ("Error code 1",)


# LLM-generated content at query #5
#--------------------------

```python
def test_set_fields():
    # Test basic field collection from bases
    class Base1:
        __fields__ = {'a': 1}

    class Base2:
        __fields__ = {'b': 2}

    class Child(Base1, Base2):
        pass

    dct = {'__fields__': {}, 'c': 3}
    bases = (Base1, Base2)
    set_fields(dct, bases, '__fields__')

    assert dct['__fields__'] == {'a': 1, 'b': 2}
    assert 'c' not in dct

    # Test _PField handling
    field1 = _PField(type={int}, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)
    field2 = _PField(type={str}, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)

    class Base3:
        __fields__ = {'field1': field1}

    class Base4:
        __fields__ = {'field2': field2}

    dct = {'__fields__': {}, 'field1': field1, 'field2': field2, 'other': 'value'}
    bases = (Base3, Base4)
    set_fields(dct, bases, '__fields__')

    assert dct['__fields__'] == {'field1': field1, 'field2': field2}
    assert 'field1' not in dct
    assert 'field2' not in dct
    assert 'other' in dct

    # Test empty bases
    dct = {'__fields__': {}, 'x': 10}
    bases = ()
    set_fields(dct, bases, '__fields__')

    assert dct['__fields__'] == {}
    assert 'x' in dct

    # Test no existing __fields__ in bases
    class EmptyBase:
        pass

    dct = {'__fields__': {}, 'y': 20}
    bases = (EmptyBase,)
    set_fields(dct, bases, '__fields__')

    assert dct['__fields__'] == {}
    assert 'y' in dct


# LLM-generated content at query #6
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)

    # Test with multiple allowed types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "hello")

    # Test with non-matching type
    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError):
        check_type(TestClass, field, "test_field", "not_an_int")

    # Test with CheckedType subclass
    class TestCheckedType(CheckedType):
        pass

    field = _PField(type={TestCheckedType}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", TestCheckedType.create({}))


# LLM-generated content at query #7
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(key_type=str, value_type=int)
    assert isinstance(field_obj, _PField)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(key_type=str, value_type=int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test with custom invariant
    custom_inv = lambda x: (True, "")
    field_obj = pmap_field(key_type=str, value_type=int, invariant=custom_inv)
    assert field_obj.invariant == custom_inv

    # Test factory behavior
    field_obj = pmap_field(key_type=str, value_type=int)
    assert field_obj.factory({"a": 1, "b": 2}) == _make_pmap_field_type(str, int)({"a": 1, "b": 2})

    # Test optional factory with None
    field_obj = pmap_field(key_type=str, value_type=int, optional=True)
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1}) == _make_pmap_field_type(str, int)({"a": 1})

    # Test type checking
    with pytest.raises(PTypeError):
        field_obj = pmap_field(key_type=str, value_type=int)
        check_type(object, field_obj, "test_field", {"a": "not_int"})

    # Test that the field type is correctly registered in _pmap_field_types
    assert (str, int) in _pmap_field_types


# LLM-generated content at query #8
#--------------------------

```python
def test_set_fields():
    class Base1:
        __fields__ = {'field1': _PField(type={int}, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)}

    class Base2:
        __fields__ = {'field2': _PField(type={str}, invariant=lambda x: (True, None), initial='', mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)}

    class TestClass(Base1, Base2):
        pass

    dct = {'field3': _PField(type={float}, invariant=lambda x: (True, None), initial=0.0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)}
    bases = (Base1, Base2)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert 'field1' in dct['__fields__']
    assert 'field2' in dct['__fields__']
    assert 'field3' not in dct['__fields__']
    assert isinstance(dct['__fields__']['field1'], _PField)
    assert isinstance(dct['__fields__']['field2'], _PField)
    assert 'field3' not in dct


# LLM-generated content at query #9
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 42)

    # Test with multiple matching types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "hello")

    # Test with non-matching type
    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "not_an_int")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "any_type_is_ok")

    # Test with CheckedType subclass
    class CustomCheckedType(CheckedType):
        pass

    field = _PField(type={CustomCheckedType}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", CustomCheckedType())


# LLM-generated content at query #10
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_spec = pmap_field(str, int)
    assert isinstance(field_spec, _PField)
    assert field_spec.type == {_pmap_field_types[(str, int)]}
    assert field_spec.mandatory is True
    assert field_spec.initial == _pmap_field_types[(str, int)]()
    assert field_spec.factory is _pmap_field_types[(str, int)].create
    assert field_spec.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_spec = pmap_field(str, int, optional=True)
    assert field_spec.type == {_pmap_field_types[(str, int)], type(None)}
    assert field_spec.mandatory is True
    assert field_spec.initial == _pmap_field_types[(str, int)]()
    assert callable(field_spec.factory)
    assert field_spec.factory(None) is None
    assert field_spec.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, "") if len(pmap) < 5 else (False, "Too many items")

    field_spec = pmap_field(str, int, invariant=custom_invariant)
    assert field_spec.invariant == custom_invariant

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    field_spec = pmap_field(str, int)
    result = field_spec.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test optional factory with None
    field_spec = pmap_field(str, int, optional=True)
    assert field_spec.factory(None) is None
    assert isinstance(field_spec.factory(test_map), CheckedPMap)

    # Test initial value
    field_spec = pmap_field(str, int, initial={"x": 10})
    assert dict(field_spec.initial) == {"x": 10}

    # Test type checking
    field_spec = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field_spec, "test_field", {"a": "not_int"})

    # Test invariant failure
    field_spec = pmap_field(str, int, invariant=lambda x: (False, "Always fails"))
    test_pmap = _pmap_field_types[(str, int)].create({"a": 1})
    with pytest.raises(InvariantException):
        check_global_invariants(test_pmap, [field_spec.invariant])


# LLM-generated content at query #11
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}_{format}"

    assert serialize(custom_serializer, "json", "test_value") == "serialized_test_value_json"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "plain_value") == "plain_value"


# LLM-generated content at query #12
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    format = "test_format"
    assert serialize(PFIELD_NO_SERIALIZER, format, checked_value) == checked_value.serialize(format)

    # Test with non-CheckedType and custom serializer
    non_checked_value = {"key": "value"}
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    assert serialize(custom_serializer, format, non_checked_value) == f"serialized_{non_checked_value}"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, format, non_checked_value) == non_checked_value


# LLM-generated content at query #13
#--------------------------

```python
def test_serialize():
    # Test with a CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with a non-CheckedType and PFIELD_NO_SERIALIZER
    non_checked_value = [1, 2, 3]
    result = serialize(PFIELD_NO_SERIALIZER, "format", non_checked_value)
    assert result == PFIELD_NO_SERIALIZER("format", non_checked_value)

    # Test with a custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"

    result = serialize(custom_serializer, "format", "value")
    assert result == "custom_format_value"


# LLM-generated content at query #14
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_pmap_field_types[(str, int)]}
    assert field_obj.mandatory is True
    assert field_obj.initial == _pmap_field_types[(str, int)]()
    assert field_obj.factory == _pmap_field_types[(str, int)].create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {_pmap_field_types[(str, int)], type(None)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _pmap_field_types[(str, int)]()
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1}) == _pmap_field_types[(str, int)]({"a": 1})

    # Test with custom invariant
    custom_invariant = lambda x: (len(x) > 0, "Map must not be empty")
    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test factory with None when optional
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1}) == _pmap_field_types[(str, int)]({"a": 1})

    # Test that the created field type is cached
    field_obj1 = pmap_field(str, int)
    field_obj2 = pmap_field(str, int)
    assert field_obj1.type == field_obj2.type


# LLM-generated content at query #15
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test when ignore_extra is False
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=CheckedPMap.create), False)

    # Test when ignore_extra is True but field type is not a set
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=CheckedPMap.create), True)

    # Test when ignore_extra is True and field type is a set but factory doesn't have ignore_extra
    assert not is_field_ignore_extra_complaint(CheckedPSet, _PField(type={int}, factory=CheckedPSet.create), True)

    # Test when ignore_extra is True, field type is a set, and factory has ignore_extra
    def factory_with_ignore_extra(iterable, _factory_fields=None, ignore_extra=False):
        return CheckedPSet.create(iterable, _factory_fields=_factory_fields, ignore_extra=ignore_extra)

    assert is_field_ignore_extra_complaint(CheckedPSet, _PField(type={int}, factory=factory_with_ignore_extra), True)

    # Test with multiple types in a set
    def factory_with_ignore_extra_multi(iterable, _factory_fields=None, ignore_extra=False):
        return CheckedPSet.create(iterable, _factory_fields=_factory_fields, ignore_extra=ignore_extra)

    assert is_field_ignore_extra_complaint(CheckedPSet, _PField(type={int, str}, factory=factory_with_ignore_extra_multi), True)

    # Test with tuple type
    def factory_with_ignore_extra_tuple(iterable, _factory_fields=None, ignore_extra=False):
        return CheckedPSet.create(iterable, _factory_fields=_factory_fields, ignore_extra=ignore_extra)

    assert is_field_ignore_extra_complaint(CheckedPSet, _PField(type=(int,), factory=factory_with_ignore_extra_tuple), True)


# LLM-generated content at query #16
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def passing_invariant(obj):
        return (True, None)

    check_global_invariants(subject, [passing_invariant])

    # Test with failing invariant
    def failing_invariant(obj):
        return (False, "ERROR_CODE")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.error_codes == ("ERROR_CODE",)

    # Test with multiple invariants, some passing and some failing
    def another_failing_invariant(obj):
        return (False, "ANOTHER_ERROR")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [passing_invariant, failing_invariant, another_failing_invariant])

    assert exc_info.value.error_codes == ("ERROR_CODE", "ANOTHER_ERROR")

    # Test with all invariants passing
    check_global_invariants(subject, [passing_invariant, passing_invariant])


# LLM-generated content at query #17
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_pmap_field_types[(str, int)]}
    assert field_obj.mandatory is True
    assert field_obj.initial == _pmap_field_types[(str, int)]()
    assert field_obj.factory is _pmap_field_types[(str, int)].create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {_pmap_field_types[(str, int)], type(None)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _pmap_field_types[(str, int)]()
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1}) == _pmap_field_types[(str, int)]({"a": 1})

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test factory with None when optional
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1}) == _pmap_field_types[(str, int)]({"a": 1})

    # Test that the created type is correctly stored in _pmap_field_types
    assert (str, int) in _pmap_field_types
    assert isinstance(_pmap_field_types[(str, int)](), CheckedPMap)

    # Test that the same type is reused for same key/value types
    field_obj1 = pmap_field(str, int)
    field_obj2 = pmap_field(str, int)
    assert field_obj1.type == field_obj2.type


# LLM-generated content at query #18
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj, _PField)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.factory == _make_pmap_field_type(str, int).create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert optional_field.factory(None) is None
    assert optional_field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    invariant_field = pmap_field(str, int, invariant=custom_invariant)
    assert invariant_field.invariant == custom_invariant

    # Test factory behavior
    test_map = {"key1": 1, "key2": 2}
    factory_result = field_obj.factory(test_map)
    assert isinstance(factory_result, CheckedPMap)
    assert dict(factory_result) == test_map

    # Test initial value
    initial_map = field_obj.initial
    assert isinstance(initial_map, CheckedPMap)
    assert len(initial_map) == 0

    # Test type checking
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field_obj, "test_field", {"invalid": "type"})

    # Test that the created type is properly registered in _pmap_field_types
    assert (str, int) in _pmap_field_types
    assert isinstance(_pmap_field_types[(str, int)], type)
    assert issubclass(_pmap_field_types[(str, int)], CheckedPMap)


# LLM-generated content at query #19
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def invariant1(obj):
        return (True, None)
    def invariant2(obj):
        return (True, None)
    check_global_invariants(subject, [invariant1, invariant2])

    # Test with failing invariant
    def failing_invariant(obj):
        return (False, "error_code")
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant])
    assert "error_code" in str(excinfo.value)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "error_code2")
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])
    assert "error_code" in str(excinfo.value)
    assert "error_code2" in str(excinfo.value)


# LLM-generated content at query #20
#--------------------------

```python
def test_set_fields():
    # Test basic field collection from bases
    class Base1:
        __fields__ = {'a': 1}

    class Base2:
        __fields__ = {'b': 2}

    class Child(Base1, Base2):
        pass

    dct = {'__fields__': {}}
    bases = (Base1, Base2)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct[name] == {'a': 1, 'b': 2}

    # Test field removal from dct
    dct = {'field1': _PField(None, None, None, False, None, None), 'other': 'value'}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct[name] == {}
    assert 'field1' not in dct
    assert 'other' in dct

    # Test with no bases and no fields
    dct = {}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct[name] == {}

    # Test with overlapping fields in bases (last base wins)
    class Base3:
        __fields__ = {'a': 1}

    class Base4:
        __fields__ = {'a': 2, 'b': 3}

    class Child2(Base3, Base4):
        pass

    dct = {'__fields__': {}}
    bases = (Base3, Base4)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct[name] == {'a': 2, 'b': 3}


# LLM-generated content at query #21
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def invariant1(x):
        return (True, None)
    def invariant2(x):
        return (True, None)
    check_global_invariants(subject, [invariant1, invariant2])

    # Test with failing invariant
    def failing_invariant(x):
        return (False, "ERROR_CODE")
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])
    assert exc_info.value.error_codes == ("ERROR_CODE",)

    # Test with multiple failing invariants
    def failing_invariant2(x):
        return (False, "ERROR_CODE2")
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])
    assert exc_info.value.error_codes == ("ERROR_CODE", "ERROR_CODE2")

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])
    assert exc_info.value.error_codes == ("ERROR_CODE",)


# LLM-generated content at query #22
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def invariant1(obj):
        return (True, None)

    def invariant2(obj):
        return (True, None)

    check_global_invariants(subject, [invariant1, invariant2])

    # Test with failing invariant
    def failing_invariant(obj):
        return (False, "ERROR_CODE")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args == ("Global invariant failed",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE_2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE", "ERROR_CODE_2")
    assert exc_info.value.args == ("Global invariant failed",)


# LLM-generated content at query #23
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    invariants = []
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with passing invariants
    def invariant1(obj):
        return (True, None)

    def invariant2(obj):
        return (True, None)

    invariants = [invariant1, invariant2]
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with failing invariant
    def failing_invariant(obj):
        return (False, "Error code")

    invariants = [failing_invariant]
    with pytest.raises(InvariantException):
        check_global_invariants(subject, invariants)

    # Test with multiple failing invariants
    def failing_invariant1(obj):
        return (False, "Error code 1")

    def failing_invariant2(obj):
        return (False, "Error code 2")

    invariants = [failing_invariant1, failing_invariant2]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants)

    assert excinfo.value.error_codes == ("Error code 1", "Error code 2")


# LLM-generated content at query #24
#--------------------------

```python
def test_serialize():
    # Test with a non-CheckedType value and custom serializer
    def custom_serializer(format, value):
        return f"custom_{value}_{format}"

    result = serialize(custom_serializer, "json", "test_value")
    assert result == "custom_test_value_json"

    # Test with a CheckedType value and PFIELD_NO_SERIALIZER
    class TestCheckedType(CheckedType):
        def serialize(self, format):
            return f"serialized_{format}"

    checked_instance = TestCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "xml", checked_instance)
    assert result == "serialized_xml"

    # Test with a non-CheckedType value and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "csv", "plain_value")
    assert result == "plain_value"


# LLM-generated content at query #25
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False (default)
    field = _PField(type=int, invariant=_valid, initial=0, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    assert not is_field_ignore_extra_complaint(int, field, False)

    # Test with ignore_extra=True but non-matching field type
    assert not is_field_ignore_extra_complaint(int, field, True)

    # Test with ignore_extra=True and matching field type (set)
    field_set = _PField(type={int}, invariant=_valid, initial=0, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    assert not is_field_ignore_extra_complaint(int, field_set, True)

    # Test with ignore_extra=True, matching field type, and factory with ignore_extra param
    def factory_with_ignore_extra(value, _factory_fields=None, ignore_extra=False):
        return value
    field_with_factory = _PField(type={int}, invariant=_valid, initial=0, mandatory=False, factory=factory_with_ignore_extra, serializer=PFIELD_NO_SERIALIZER)
    assert is_field_ignore_extra_complaint(int, field_with_factory, True)

    # Test with ignore_extra=True, matching field type, but factory without ignore_extra param
    def factory_without_ignore_extra(value, _factory_fields=None):
        return value
    field_without_factory = _PField(type={int}, invariant=_valid, initial=0, mandatory=False, factory=factory_without_ignore_extra, serializer=PFIELD_NO_SERIALIZER)
    assert not is_field_ignore_extra_complaint(int, field_without_factory, True)


# LLM-generated content at query #26
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_value = {"key": "value"}
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    result = serialize(custom_serializer, "format", custom_value)
    assert result == "serialized_{'key': 'value'}"

    # Test with PFIELD_NO_SERIALIZER and non-CheckedType
    non_checked_value = [1, 2, 3]
    result = serialize(PFIELD_NO_SERIALIZER, "format", non_checked_value)
    assert result == non_checked_value


# LLM-generated content at query #27
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_pmap_field_types[(str, int)]}
    assert field_obj.factory is not None
    assert field_obj.mandatory is True
    assert isinstance(field_obj.initial, CheckedPMap)

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {optional_type(_pmap_field_types[(str, int)])}
    assert optional_field.factory(None) is None
    assert optional_field.factory({"a": 1}) == _pmap_field_types[(str, int)].create({"a": 1})

    # Test with invariant
    def test_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")

    field_with_inv = pmap_field(str, int, invariant=test_invariant)
    assert field_with_inv.invariant is not PFIELD_NO_INVARIANT

    # Test factory behavior
    test_map = {"key1": 1, "key2": 2}
    result = field_obj.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test initial value
    assert dict(field_obj.initial) == {}
    assert isinstance(field_obj.initial, CheckedPMap)

    # Test type checking
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field_obj, "test_field", "not_a_map")

    # Test with custom invariant that fails
    def failing_invariant(pmap):
        return (False, "Always fails")

    failing_field = pmap_field(str, int, invariant=failing_invariant)
    test_pmap = _pmap_field_types[(str, int)].create({"a": 1})
    with pytest.raises(InvariantException):
        check_global_invariants(test_pmap, [failing_field.invariant])


# LLM-generated content at query #28
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.factory is not None
    assert field_obj.mandatory is True
    assert isinstance(field_obj.initial, CheckedPMap)

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.factory(None) is None

    # Test with initial value
    initial_map = {"a": 1, "b": 2}
    field_obj = pmap_field(str, int, initial=initial_map)
    assert field_obj.initial == _make_pmap_field_type(str, int).create(initial_map)

    # Test with invariant
    def test_invariant(pmap):
        return (True, None) if len(pmap) > 0 else (False, "Map must not be empty")

    field_obj = pmap_field(str, int, invariant=test_invariant)
    assert field_obj.invariant is not None

    # Test factory with None for optional field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert isinstance(field_obj.factory({"x": 1}), CheckedPMap)

    # Test factory with non-None value
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj.factory({"x": 1}), CheckedPMap)

    # Test that the created field has correct type checking
    field_obj = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field_obj, "test_field", {"a": "not_int"})

    # Test that the created field accepts correct types
    try:
        check_type(type('TestClass', (), {}), field_obj, "test_field", {"a": 1})
    except PTypeError:
        pytest.fail("check_type raised PTypeError unexpectedly")


# LLM-generated content at query #29
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert isinstance(field_obj.initial, CheckedPMap)
    assert field_obj.initial == _make_pmap_field_type(str, int)()

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {_make_pmap_field_type(str, int), type(None)}
    assert optional_field.factory(None) is None

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, "") if len(pmap) < 5 else (False, "Too many items")
    field_with_inv = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_inv.invariant is not PFIELD_NO_INVARIANT

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    result = field_obj.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert result == _make_pmap_field_type(str, int)(test_map)

    # Test that factory returns None for optional field
    assert optional_field.factory(None) is None

    # Test that initial value is correct type
    assert isinstance(field_obj.initial, CheckedPMap)
    assert isinstance(optional_field.initial, CheckedPMap)

    # Test that the field can be used in a class
    class TestRecord:
        map_field = pmap_field(str, int)

    record = TestRecord()
    assert hasattr(record, 'map_field')


# LLM-generated content at query #30
#--------------------------

```python
def test_check_global_invariants():
    # Test with valid invariants
    subject = {"a": 1, "b": 2}
    invariants = [
        lambda x: (True, None),
        lambda x: (x["a"] < x["b"], "a should be less than b")
    ]
    check_global_invariants(subject, invariants)

    # Test with one failing invariant
    subject = {"a": 3, "b": 2}
    invariants = [
        lambda x: (True, None),
        lambda x: (x["a"] < x["b"], "a should be less than b")
    ]
    with pytest.raises(InvariantException):
        check_global_invariants(subject, invariants)

    # Test with multiple failing invariants
    subject = {"a": 3, "b": 2, "c": 1}
    invariants = [
        lambda x: (x["a"] < x["b"], "a should be less than b"),
        lambda x: (x["b"] < x["c"], "b should be less than c")
    ]
    with pytest.raises(InvariantException):
        check_global_invariants(subject, invariants)

    # Test with empty invariants
    subject = {"a": 1, "b": 2}
    invariants = []
    check_global_invariants(subject, invariants)

    # Test with no failing invariants but error codes
    subject = {"a": 1, "b": 2}
    invariants = [
        lambda x: (True, "error_code_1"),
        lambda x: (True, "error_code_2")
    ]
    check_global_invariants(subject, invariants)


