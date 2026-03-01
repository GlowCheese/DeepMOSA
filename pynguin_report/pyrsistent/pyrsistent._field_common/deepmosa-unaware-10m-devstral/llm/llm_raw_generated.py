####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_set_fields():
    # Test case 1: Basic field collection from bases
    class Base1:
        __fields__ = {'field1': 'value1'}

    class Base2:
        __fields__ = {'field2': 'value2'}

    class TestClass(Base1, Base2):
        pass

    dct = {'__fields__': {}}
    bases = (Base1, Base2)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct['__fields__'] == {'field1': 'value1', 'field2': 'value2'}

    # Test case 2: Field collection with _PField instances
    class Base3:
        __fields__ = {'field3': _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)}

    class TestClass2(Base3):
        pass

    dct2 = {'__fields__': {}}
    bases2 = (Base3,)
    name2 = '__fields__'

    set_fields(dct2, bases2, name2)

    assert 'field3' in dct2['__fields__']
    assert isinstance(dct2['__fields__']['field3'], _PField)
    assert 'field3' not in dct2

    # Test case 3: Empty bases
    class TestClass3:
        pass

    dct3 = {'__fields__': {}}
    bases3 = ()
    name3 = '__fields__'

    set_fields(dct3, bases3, name3)

    assert dct3['__fields__'] == {}


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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
    set_fields(dct, (Base1, Base2), '__fields__')
    assert dct['__fields__'] == {'a': 1, 'b': 2}
    assert 'c' not in dct

    # Test _PField handling
    field1 = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)
    field2 = _PField(type=str, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)

    class Base3:
        __fields__ = {'field1': field1}

    class Base4:
        __fields__ = {'field2': field2}

    dct = {'__fields__': {}, 'field1': field1, 'field2': field2, 'other': 'value'}
    set_fields(dct, (Base3, Base4), '__fields__')
    assert dct['__fields__'] == {'field1': field1, 'field2': field2}
    assert 'field1' not in dct
    assert 'field2' not in dct
    assert 'other' in dct

    # Test empty bases
    dct = {'__fields__': {}, 'a': 1}
    set_fields(dct, (), '__fields__')
    assert dct['__fields__'] == {}
    assert 'a' in dct

    # Test no existing __fields__ in bases
    class Base5:
        pass

    dct = {'__fields__': {}, 'a': 1}
    set_fields(dct, (Base5,), '__fields__')
    assert dct['__fields__'] == {}
    assert 'a' in dct


# LLM-generated content at query #4
#--------------------------

```python
def test_field():
    # Test basic field creation
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory is False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=[int, str])
    assert f.type == {int, str}

    # Test field with invariant
    def test_inv(value):
        return (value > 0, "Value must be positive")
    f = field(invariant=test_inv)
    assert f.invariant(5) == (True, None)
    assert f.invariant(-1) == (False, "Value must be positive")

    # Test field with initial value
    f = field(initial=10)
    assert f.initial == 10

    # Test field with mandatory
    f = field(mandatory=True)
    assert f.mandatory is True

    # Test field with factory
    def test_factory(value):
        return value * 2
    f = field(factory=test_factory)
    assert f.factory(5) == 10

    # Test field with serializer
    def test_serializer(format, value):
        return str(value)
    f = field(serializer=test_serializer)
    assert f.serializer(None, 5) == "5"

    # Test field with invalid type
    try:
        field(type=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid initial type
    try:
        field(type=int, initial="string")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable invariant
    try:
        field(invariant="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable factory
    try:
        field(factory="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable serializer
    try:
        field(serializer="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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

    assert '__fields__' in dct
    assert dct['__fields__'] == {'a': 1, 'b': 2}
    assert 'c' not in dct

    # Test _PField handling
    class TestField(_PField):
        pass

    dct = {'__fields__': {}, 'field1': TestField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert 'field1' in dct['__fields__']
    assert isinstance(dct['__fields__']['field1'], TestField)
    assert 'field1' not in dct

    # Test with no bases and no fields
    dct = {'__fields__': {}}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert dct['__fields__'] == {}

    # Test with overlapping fields (should be combined)
    class Base3:
        __fields__ = {'a': 1, 'b': 2}
    class Base4:
        __fields__ = {'b': 3, 'c': 4}

    dct = {'__fields__': {}}
    bases = (Base3, Base4)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert dct['__fields__'] == {'a': 1, 'b': 3, 'c': 4}


# LLM-generated content at query #2
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"custom_{val}"
    result = serialize(custom_serializer, "format", "test_value")
    assert result == "custom_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "format", "test_value")
    assert result == "test_value"


# LLM-generated content at query #3
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

    assert '__fields__' in dct
    assert 'c' not in dct
    assert dct['__fields__'] == {'a': 1, 'b': 2}

    # Test _PField handling
    field = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)
    dct = {'__fields__': {}, 'field1': field}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert 'field1' not in dct
    assert dct['__fields__'] == {'field1': field}

    # Test empty bases
    dct = {'__fields__': {}, 'field1': 1}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert 'field1' not in dct
    assert dct['__fields__'] == {}

    # Test no existing __fields__ in bases
    class EmptyBase:
        pass

    dct = {'__fields__': {}, 'field1': 1}
    bases = (EmptyBase,)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert 'field1' not in dct
    assert dct['__fields__'] == {}


# LLM-generated content at query #4
#--------------------------

```python
def test_field():
    # Test basic field creation
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory == False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=(int, str))
    assert f.type == {int, str}

    # Test field with invariant
    def my_invariant(x):
        return (True, None)
    f = field(invariant=my_invariant)
    assert f.invariant == my_invariant

    # Test field with initial value
    f = field(initial=5)
    assert f.initial == 5

    # Test field with mandatory
    f = field(mandatory=True)
    assert f.mandatory == True

    # Test field with factory
    def my_factory(x):
        return x
    f = field(factory=my_factory)
    assert f.factory == my_factory

    # Test field with serializer
    def my_serializer(format, value):
        return value
    f = field(serializer=my_serializer)
    assert f.serializer == my_serializer

    # Test field with all parameters
    f = field(type=int, invariant=my_invariant, initial=5, mandatory=True, factory=my_factory, serializer=my_serializer)
    assert f.type == {int}
    assert f.invariant == my_invariant
    assert f.initial == 5
    assert f.mandatory == True
    assert f.factory == my_factory
    assert f.serializer == my_serializer

    # Test field with invalid type
    try:
        field(type="invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid initial type
    try:
        field(type=int, initial="invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid invariant
    try:
        field(invariant="invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid factory
    try:
        field(factory="invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid serializer
    try:
        field(serializer="invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.factory is _make_pmap_field_type(str, int).create
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.factory(None) is None
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, "") if len(pmap) < 10 else (False, "Too many items")

    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test pmap_field factory behavior
    field = pmap_field(str, int)
    pmap = field.factory({"a": 1, "b": 2})
    assert isinstance(pmap, CheckedPMap)
    assert pmap["a"] == 1
    assert pmap["b"] == 2

    # Test optional pmap_field factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    pmap = field.factory({"a": 1})
    assert isinstance(pmap, CheckedPMap)
    assert pmap["a"] == 1

    # Test pmap_field with type checking
    field = pmap_field(str, int)
    pmap = field.factory({"a": 1, "b": 2})
    check_type(type(pmap), field, "test_field", pmap)  # Should not raise
    try:
        check_type(type(pmap), field, "test_field", {"a": "not_int"})
        assert False, "Expected PTypeError"
    except PTypeError:
        pass

    # Test pmap_field with invariant violation
    field = pmap_field(str, int, invariant=lambda pmap: (len(pmap) < 2, "Too small"))
    pmap = field.factory({"a": 1})
    try:
        check_global_invariants(pmap, [field.invariant])
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #6
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

    assert exc_info.value.args[0] == ("ERROR_CODE",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE_2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("ERROR_CODE", "ERROR_CODE_2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"


# LLM-generated content at query #7
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    class TestCheckedType(CheckedType):
        pass

    checked_instance = TestCheckedType()
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_instance) == checked_instance.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}"

    assert serialize(custom_serializer, "format", "test_value") == "serialized_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "test_value") == "test_value"


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

```python
def test_check_type():
    # Test valid type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, 'test_field', 10)
    check_type(TestClass, field, 'test_field', 0)

    # Test invalid type
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, 'test_field', 'string')
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == 'test_field'
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test multiple valid types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, 'test_field', 10)
    check_type(TestClass, field, 'test_field', 'string')

    # Test invalid type with multiple valid types
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, 'test_field', 1.5)
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == 'test_field'
    assert excinfo.value.expected_types == {int, str}
    assert excinfo.value.actual_type == float

    # Test with no type constraint
    field = _PField(type=PFIELD_NO_TYPE, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, 'test_field', 10)
    check_type(TestClass, field, 'test_field', 'string')
    check_type(TestClass, field, 'test_field', 1.5)
    check_type(TestClass, field, 'test_field', TestClass())


# LLM-generated content at query #10
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=lambda x: x), False)
    assert not is_field_ignore_extra_complaint(CheckedPSet, _PField(type=int, factory=lambda x: x), False)
    assert not is_field_ignore_extra_complaint(CheckedPVector, _PField(type=int, factory=lambda x: x), False)

    # Test with ignore_extra=True but field type not matching
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=lambda x: x), True)
    assert not is_field_ignore_extra_complaint(CheckedPSet, _PField(type=int, factory=lambda x: x), True)
    assert not is_field_ignore_extra_complaint(CheckedPVector, _PField(type=int, factory=lambda x: x), True)

    # Test with ignore_extra=True and field type matching but factory not accepting ignore_extra
    def factory_no_ignore_extra(x):
        return x
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set, factory=factory_no_ignore_extra), True)
    assert not is_field_ignore_extra_complaint(CheckedPSet, _PField(type=set, factory=factory_no_ignore_extra), True)
    assert not is_field_ignore_extra_complaint(CheckedPVector, _PField(type=set, factory=factory_no_ignore_extra), True)

    # Test with ignore_extra=True, field type matching, and factory accepting ignore_extra
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set, factory=factory_with_ignore_extra), True)
    assert is_field_ignore_extra_complaint(CheckedPSet, _PField(type=set, factory=factory_with_ignore_extra), True)
    assert is_field_ignore_extra_complaint(CheckedPVector, _PField(type=set, factory=factory_with_ignore_extra), True)

    # Test with field type as a set containing the type class
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type={set}, factory=factory_with_ignore_extra), True)
    assert is_field_ignore_extra_complaint(CheckedPSet, _PField(type={set}, factory=factory_with_ignore_extra), True)
    assert is_field_ignore_extra_complaint(CheckedPVector, _PField(type={set}, factory=factory_with_ignore_extra), True)


# LLM-generated content at query #11
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, 'format', checked_value) == checked_value.serialize('format')

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}_{format}"

    assert serialize(custom_serializer, 'json', 'test_value') == 'serialized_test_value_json'

    # Test with CheckedType and custom serializer (should use custom serializer)
    assert serialize(custom_serializer, 'json', checked_value) == custom_serializer('json', checked_value)


# LLM-generated content at query #12
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_pmap_field_types[(str, int)]}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(str, int)]()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {_pmap_field_types[(str, int)], type(None)}
    assert field.mandatory is True

    # Test with custom invariant
    custom_invariant = lambda x: (True, "")
    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test factory behavior
    field = pmap_field(str, int)
    result = field.factory({"a": 1, "b": 2})
    assert isinstance(result, CheckedPMap)
    assert result == {"a": 1, "b": 2}

    # Test optional factory with None
    field = pmap_field(str, int, optional=True)
    result = field.factory(None)
    assert result is None

    # Test that the created type is cached
    field1 = pmap_field(str, int)
    field2 = pmap_field(str, int)
    assert field1.type == field2.type

    # Test with different types
    field = pmap_field(int, str)
    assert field.type == {_pmap_field_types[(int, str)]}


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

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", {"a": 1})

    # Test with non-matching type
    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "not an int")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test with CheckedType subclass
    class TestCheckedType(CheckedType):
        pass

    field = _PField(type={TestCheckedType}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", TestCheckedType())


# LLM-generated content at query #14
#--------------------------

```python
def test_field():
    # Test basic field creation
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory is False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=[int, str])
    assert f.type == {int, str}

    # Test field with invariant
    def test_invariant(x):
        return (x > 0, "Value must be positive")
    f = field(invariant=test_invariant)
    assert f.invariant(5) == (True, None)
    assert f.invariant(-1) == (False, "Value must be positive")

    # Test field with initial value
    f = field(initial=10)
    assert f.initial == 10

    # Test field with mandatory
    f = field(mandatory=True)
    assert f.mandatory is True

    # Test field with factory
    def test_factory(x):
        return x * 2
    f = field(factory=test_factory)
    assert f.factory(5) == 10

    # Test field with serializer
    def test_serializer(format, value):
        return str(value)
    f = field(serializer=test_serializer)
    assert f.serializer("json", 10) == "10"

    # Test field with all parameters
    f = field(type=int, invariant=test_invariant, initial=10, mandatory=True, factory=test_factory, serializer=test_serializer)
    assert f.type == {int}
    assert f.invariant(5) == (True, None)
    assert f.initial == 10
    assert f.mandatory is True
    assert f.factory(5) == 10
    assert f.serializer("json", 10) == "10"

    # Test field with invalid type
    try:
        field(type=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid initial type
    try:
        field(type=int, initial="string")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid invariant
    try:
        field(invariant="not a function")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid factory
    try:
        field(factory="not a function")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid serializer
    try:
        field(serializer="not a function")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_check_type():
    # Test valid type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)

    # Test invalid type
    field = _PField(type={str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", 10)
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {str}
    assert excinfo.value.actual_type == int

    # Test multiple valid types
    field = _PField(type={int, float}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)
    check_type(TestClass, field, "test_field", 10.5)

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "any_type")


# LLM-generated content at query #16
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, 'format', checked_value) == checked_value.serialize('format')

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"custom_{val}"
    assert serialize(custom_serializer, 'format', "test") == "custom_test"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, 'format', "test") == "test"


# LLM-generated content at query #17
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"custom_{val}"
    assert serialize(custom_serializer, "format", "test_value") == "custom_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "test_value") == "test_value"


# LLM-generated content at query #18
#--------------------------

```python
def test_check_type():
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)

    # Test valid type
    check_type(TestClass, field, "test_field", 10)

    # Test invalid type
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "not_an_int")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test multiple valid types
    field.type = {int, str}
    check_type(TestClass, field, "test_field", 10)
    check_type(TestClass, field, "test_field", "string")

    # Test invalid type with multiple valid types
    with pytest.raises(PTypeError):
        check_type(TestClass, field, "test_field", 10.5)

    # Test no type constraint
    field.type = set()
    check_type(TestClass, field, "test_field", "anything")


# LLM-generated content at query #19
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_pmap_field_types[(str, int)]}
    assert field_obj.mandatory is True
    assert field_obj.initial == _pmap_field_types[(str, int)]()
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {_pmap_field_types[(str, int)], type(None)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _pmap_field_types[(str, int)]()
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    custom_invariant = lambda x: (True, "")
    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test pmap_field factory
    field_obj = pmap_field(str, int)
    test_map = field_obj.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)
    assert test_map == {"a": 1, "b": 2}

    # Test optional pmap_field factory with None
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1}) == {"a": 1}

    # Test pmap_field with invalid types
    with pytest.raises(TypeError):
        pmap_field("not a type", int)
    with pytest.raises(TypeError):
        pmap_field(str, "not a type")

    # Test pmap_field type checking
    field_obj = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field_obj, "test_field", {"a": "not an int"})
    with pytest.raises(PTypeError):
        check_type(object, field_obj, "test_field", {1: 2})  # key not str


# LLM-generated content at query #20
#--------------------------

```python
def test_field():
    # Test basic field creation
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial is PFIELD_NO_INITIAL
    assert f.mandatory is False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=[int, str])
    assert f.type == {int, str}

    # Test field with invariant
    def test_inv(value):
        return (True, None)
    f = field(invariant=test_inv)
    assert f.invariant == test_inv

    # Test field with initial value
    f = field(initial=5)
    assert f.initial == 5

    # Test field with mandatory
    f = field(mandatory=True)
    assert f.mandatory is True

    # Test field with factory
    def test_factory(x):
        return x
    f = field(factory=test_factory)
    assert f.factory == test_factory

    # Test field with serializer
    def test_serializer(format, value):
        return value
    f = field(serializer=test_serializer)
    assert f.serializer == test_serializer

    # Test field with all parameters
    f = field(type=int, invariant=test_inv, initial=5, mandatory=True, factory=test_factory, serializer=test_serializer)
    assert f.type == {int}
    assert f.invariant == test_inv
    assert f.initial == 5
    assert f.mandatory is True
    assert f.factory == test_factory
    assert f.serializer == test_serializer

    # Test field with invalid type
    try:
        field(type=5)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid initial type
    try:
        field(type=int, initial="string")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid invariant
    try:
        field(invariant="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid factory
    try:
        field(factory="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid serializer
    try:
        field(serializer="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #21
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


# LLM-generated content at query #22
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
    field = _PField(type={int, float}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10.5)

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "any_value")

    # Test with non-matching type
    field = _PField(type={str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field, "test_field", 10)
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == {str}
    assert exc_info.value.actual_type == int

    # Test with CheckedType subclass
    class TestCheckedType(CheckedType):
        pass

    field = _PField(type={TestCheckedType}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", TestCheckedType())


# LLM-generated content at query #23
#--------------------------

```python
def test_set_fields():
    # Test basic field collection from base classes
    class A:
        __fields__ = {'a': field()}
        b = field()

    class B:
        __fields__ = {'c': field()}
        d = field()

    class C(A, B):
        pass

    set_fields(C.__dict__, (A, B), '__fields__')
    assert C.__fields__ == {'a': A.__fields__['a'], 'c': B.__fields__['c'], 'b': A.b, 'd': B.d}

    # Test field collection with no base classes
    class D:
        e = field()

    set_fields(D.__dict__, (), '__fields__')
    assert D.__fields__ == {'e': D.e}

    # Test field collection with no fields in base classes
    class E:
        pass

    class F:
        f = field()

    set_fields(F.__dict__, (E,), '__fields__')
    assert F.__fields__ == {'f': F.f}

    # Test field collection with overlapping fields (last base class wins)
    class G:
        g = field()

    class H:
        g = field()

    class I(G, H):
        pass

    set_fields(I.__dict__, (G, H), '__fields__')
    assert I.__fields__ == {'g': H.g}

    # Test that original fields are removed from class dict
    class J:
        j = field()

    set_fields(J.__dict__, (), '__fields__')
    assert 'j' not in J.__dict__
    assert 'j' in J.__fields__


# LLM-generated content at query #24
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

    # Test with CheckedType and custom serializer (should use custom serializer)
    assert serialize(custom_serializer, "json", checked_value) == custom_serializer("json", checked_value)


# LLM-generated content at query #25
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

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args == ("Global invariant failed",)


# LLM-generated content at query #26
#--------------------------

```python
def test_field():
    # Test basic field creation
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory == False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=[int, str])
    assert f.type == {int, str}

    # Test field with invariant
    def test_invariant(x):
        return (True, None)
    f = field(invariant=test_invariant)
    assert f.invariant == test_invariant

    # Test field with initial value
    f = field(initial=5)
    assert f.initial == 5

    # Test field with mandatory
    f = field(mandatory=True)
    assert f.mandatory == True

    # Test field with factory
    def test_factory(x):
        return x
    f = field(factory=test_factory)
    assert f.factory == test_factory

    # Test field with serializer
    def test_serializer(format, x):
        return x
    f = field(serializer=test_serializer)
    assert f.serializer == test_serializer

    # Test field with all parameters
    f = field(type=int, invariant=test_invariant, initial=5, mandatory=True, factory=test_factory, serializer=test_serializer)
    assert f.type == {int}
    assert f.invariant == test_invariant
    assert f.initial == 5
    assert f.mandatory == True
    assert f.factory == test_factory
    assert f.serializer == test_serializer

    # Test field with invalid type
    try:
        field(type=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid initial type
    try:
        field(type=int, initial="not an int")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable invariant
    try:
        field(invariant="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable factory
    try:
        field(factory="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable serializer
    try:
        field(serializer="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #27
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
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.factory(None) is None
    assert isinstance(field_obj.factory({}), _make_pmap_field_type(str, int))

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    result = field_obj.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert result == test_map

    # Test initial value
    field_obj = pmap_field(str, int, initial={"x": 10})
    assert field_obj.initial == {"x": 10}

    # Test type checking
    field_obj = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type, field_obj, "test_field", {"a": "not_int"})

    # Test with None when optional
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None


# LLM-generated content at query #28
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
    with pytest.raises(PTypeError):
        check_type(TestClass, field, "test_field", "not_an_int")

    # Test with no type constraint
    field = _PField(type=PFIELD_NO_TYPE, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "anything_goes")


# LLM-generated content at query #29
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


# LLM-generated content at query #30
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
    with pytest.raises(PTypeError):
        check_type(TestClass, field, "test_field", "not_an_int")

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


# LLM-generated content at query #31
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


# LLM-generated content at query #32
#--------------------------

```python
def test_field():
    # Test basic field creation
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory == False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=[int, str])
    assert f.type == {int, str}

    # Test field with invariant
    def test_inv(value):
        return (value > 0, "Value must be positive")
    f = field(invariant=test_inv)
    assert f.invariant(5) == (True, None)
    assert f.invariant(-1) == (False, "Value must be positive")

    # Test field with initial value
    f = field(initial=10)
    assert f.initial == 10

    # Test field with mandatory
    f = field(mandatory=True)
    assert f.mandatory == True

    # Test field with factory
    def test_factory(x):
        return x * 2
    f = field(factory=test_factory)
    assert f.factory(5) == 10

    # Test field with serializer
    def test_serializer(format, value):
        return str(value)
    f = field(serializer=test_serializer)
    assert f.serializer(None, 10) == "10"

    # Test field with invalid type
    try:
        field(type=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid initial type
    try:
        field(type=int, initial="not an int")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable invariant
    try:
        field(invariant="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable factory
    try:
        field(factory="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable serializer
    try:
        field(serializer="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #33
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.factory is _make_pmap_field_type(str, int).create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test with optional=True
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.factory(None) is None
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test factory behavior
    field_obj = pmap_field(str, int)
    test_map = field_obj.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)
    assert test_map["a"] == 1
    assert test_map["b"] == 2

    # Test optional factory behavior
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    test_map = field_obj.factory({"a": 1})
    assert isinstance(test_map, CheckedPMap)
    assert test_map["a"] == 1

    # Test initial value
    field_obj = pmap_field(str, int, initial={"x": 10})
    initial_map = field_obj.initial
    assert isinstance(initial_map, CheckedPMap)
    assert initial_map["x"] == 10

    # Test that the same field type is reused
    field1 = pmap_field(str, int)
    field2 = pmap_field(str, int)
    assert field1.type == field2.type
    assert field1.initial == field2.initial


# LLM-generated content at query #34
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
        check_type(TestClass, field, "test_field", "hello")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "anything")

    # Test with CheckedType subclass
    class TestCheckedType(CheckedType):
        pass

    field = _PField(type={TestCheckedType}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", TestCheckedType())


# LLM-generated content at query #35
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)

    # Test with non-matching type
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "string")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test with multiple allowed types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)
    check_type(TestClass, field, "test_field", "string")

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "anything")


# LLM-generated content at query #36
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
    instance = TestCheckedType()
    check_type(TestClass, field, "test_field", instance)


# LLM-generated content at query #37
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

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args == ("Global invariant failed",)


# LLM-generated content at query #38
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
    field = _PField(type={int, float}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10.5)

    # Test with non-matching type
    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "string")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "any_type")

    # Test with CheckedType subclass
    class TestCheckedType(CheckedType):
        pass

    field = _PField(type={TestCheckedType}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", TestCheckedType())


# LLM-generated content at query #39
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    class TestCheckedType(CheckedType):
        pass

    # Test with ignore_extra=False (default)
    field = _PField(type=TestCheckedType, invariant=lambda x: (True, None),
                    initial=None, mandatory=False, factory=lambda x: x,
                    serializer=lambda _, value: value)
    assert not is_field_ignore_extra_complaint(CheckedPVector, field, False)
    assert not is_field_ignore_extra_complaint(CheckedPSet, field, False)
    assert not is_field_ignore_extra_complaint(CheckedPMap, field, False)

    # Test with ignore_extra=True but non-matching type
    assert not is_field_ignore_extra_complaint(CheckedPVector, field, True)
    assert not is_field_ignore_extra_complaint(CheckedPSet, field, True)
    assert not is_field_ignore_extra_complaint(CheckedPMap, field, True)

    # Test with ignore_extra=True and matching type but no ignore_extra in factory
    field = _PField(type=CheckedPVector, invariant=lambda x: (True, None),
                    initial=None, mandatory=False, factory=lambda x: x,
                    serializer=lambda _, value: value)
    assert not is_field_ignore_extra_complaint(CheckedPVector, field, True)

    # Test with ignore_extra=True, matching type, and ignore_extra in factory
    def factory_with_ignore_extra(x, _factory_fields=None, ignore_extra=False):
        return x
    field = _PField(type=CheckedPVector, invariant=lambda x: (True, None),
                    initial=None, mandatory=False, factory=factory_with_ignore_extra,
                    serializer=lambda _, value: value)
    assert is_field_ignore_extra_complaint(CheckedPVector, field, True)

    # Test with set type
    field = _PField(type={TestCheckedType}, invariant=lambda x: (True, None),
                    initial=None, mandatory=False, factory=factory_with_ignore_extra,
                    serializer=lambda _, value: value)
    assert is_field_ignore_extra_complaint(CheckedPVector, field, True)
    assert is_field_ignore_extra_complaint(CheckedPSet, field, True)
    assert is_field_ignore_extra_complaint(CheckedPMap, field, True)

    # Test with empty type
    field = _PField(type=(), invariant=lambda x: (True, None),
                    initial=None, mandatory=False, factory=factory_with_ignore_extra,
                    serializer=lambda _, value: value)
    assert not is_field_ignore_extra_complaint(CheckedPVector, field, True)
    assert not is_field_ignore_extra_complaint(CheckedPSet, field, True)
    assert not is_field_ignore_extra_complaint(CheckedPMap, field, True)


# LLM-generated content at query #40
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test case 1: ignore_extra is False
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set(), factory=lambda x: x), False)

    # Test case 2: ignore_extra is True but field type is not a set
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=lambda x: x), True)

    # Test case 3: ignore_extra is True, field type is a set, but factory doesn't have ignore_extra parameter
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set(), factory=lambda x: x), True)

    # Test case 4: ignore_extra is True, field type is a set, and factory has ignore_extra parameter
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set(), factory=factory_with_ignore_extra), True)

    # Test case 5: field type is a tuple with a set type
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=(set(),), factory=factory_with_ignore_extra), True)

    # Test case 6: field type is a list with a set type
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=[set()], factory=factory_with_ignore_extra), True)


# LLM-generated content at query #41
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_pmap_field_types[(str, int)]}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(str, int)]()
    assert field.factory is _pmap_field_types[(str, int)].create
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {_pmap_field_types[(str, int)], type(None)}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(str, int)]()
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _pmap_field_types[(str, int)]({"a": 1})

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test factory behavior
    field = pmap_field(str, int)
    assert field.factory({"a": 1, "b": 2}) == _pmap_field_types[(str, int)]({"a": 1, "b": 2})

    # Test that the created type is properly registered
    assert (str, int) in _pmap_field_types
    assert isinstance(_pmap_field_types[(str, int)](), CheckedPMap)

    # Test with different types
    field = pmap_field(int, str)
    assert field.type == {_pmap_field_types[(int, str)]}
    assert (int, str) in _pmap_field_types

    # Test that the field can be used in a class
    class TestRecord:
        test_field = pmap_field(str, int)

    record = TestRecord()
    assert hasattr(record, 'test_field')


# LLM-generated content at query #42
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_spec = pmap_field(str, int)
    assert isinstance(field_spec, _PField)
    assert field_spec.mandatory is True
    assert field_spec.initial == CheckedPMap()
    assert field_spec.type == {CheckedPMap}

    # Test optional pmap_field
    field_spec = pmap_field(str, int, optional=True)
    assert field_spec.type == {CheckedPMap, type(None)}

    # Test with invariant
    def test_invariant(pmap):
        return (True, None)
    field_spec = pmap_field(str, int, invariant=test_invariant)
    assert field_spec.invariant == test_invariant

    # Test factory behavior
    field_spec = pmap_field(str, int)
    result = field_spec.factory({"a": 1, "b": 2})
    assert isinstance(result, CheckedPMap)
    assert result == CheckedPMap({"a": 1, "b": 2})

    # Test optional factory with None
    field_spec = pmap_field(str, int, optional=True)
    result = field_spec.factory(None)
    assert result is None

    # Test with initial value
    field_spec = pmap_field(str, int, initial={"x": 10})
    assert field_spec.initial == CheckedPMap({"x": 10})

    # Test type checking
    field_spec = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field_spec, "test_field", {"a": "not_int"})

    # Test invariant failure
    def failing_invariant(pmap):
        return (False, "error") if len(pmap) > 0 else (True, None)
    field_spec = pmap_field(str, int, invariant=failing_invariant)
    with pytest.raises(InvariantException):
        check_global_invariants(CheckedPMap({"a": 1}), [field_spec.invariant])


# LLM-generated content at query #43
#--------------------------

```python
def test_field():
    # Test basic field creation
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial is PFIELD_NO_INITIAL
    assert f.mandatory is False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=[int, str])
    assert f.type == {int, str}

    # Test field with invariant
    def test_invariant(x):
        return (True, None)
    f = field(invariant=test_invariant)
    assert f.invariant == test_invariant

    # Test field with initial value
    f = field(initial=5)
    assert f.initial == 5

    # Test field with mandatory
    f = field(mandatory=True)
    assert f.mandatory is True

    # Test field with factory
    def test_factory(x):
        return x
    f = field(factory=test_factory)
    assert f.factory == test_factory

    # Test field with serializer
    def test_serializer(format, x):
        return x
    f = field(serializer=test_serializer)
    assert f.serializer == test_serializer

    # Test field with all parameters
    f = field(type=int, invariant=test_invariant, initial=5, mandatory=True,
              factory=test_factory, serializer=test_serializer)
    assert f.type == {int}
    assert f.invariant == test_invariant
    assert f.initial == 5
    assert f.mandatory is True
    assert f.factory == test_factory
    assert f.serializer == test_serializer

    # Test field with invalid type
    try:
        field(type=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid initial type
    try:
        field(type=int, initial="not an int")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable invariant
    try:
        field(invariant="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable factory
    try:
        field(factory="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable serializer
    try:
        field(serializer="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #44
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    format = "test_format"
    result = serialize(PFIELD_NO_SERIALIZER, format, checked_value)
    assert result == checked_value.serialize(format)

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    value = "test_value"
    result = serialize(custom_serializer, format, value)
    assert result == custom_serializer(format, value)

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    value = "test_value"
    result = serialize(PFIELD_NO_SERIALIZER, format, value)
    assert result == value


# LLM-generated content at query #45
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)

    check_type(TestClass, field, "test_field", 10)

    # Test with non-matching type
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field, "test_field", "not_an_int")

    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == {int}
    assert exc_info.value.actual_type == str

    # Test with multiple allowed types
    field.type = {int, str}
    check_type(TestClass, field, "test_field", "now_its_ok")

    # Test with no type constraint
    field.type = set()
    check_type(TestClass, field, "test_field", "anything_goes")

    # Test with custom type
    class CustomType:
        pass

    field.type = {CustomType}
    instance = CustomType()
    check_type(TestClass, field, "test_field", instance)

    with pytest.raises(PTypeError):
        check_type(TestClass, field, "test_field", "not_custom_type")


# LLM-generated content at query #46
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.factory is _make_pmap_field_type(str, int).create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert callable(field_obj.factory)
    assert field_obj.factory(None) is None
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    custom_inv = lambda x: (True, "")
    field_obj = pmap_field(str, int, invariant=custom_inv)
    assert field_obj.invariant == custom_inv

    # Test pmap_field factory behavior
    field_obj = pmap_field(str, int)
    test_map = {"a": 1, "b": 2}
    result = field_obj.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test optional pmap_field factory with None
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert field_obj.factory({"x": 10}) == _make_pmap_field_type(str, int).create({"x": 10})

    # Test pmap_field with multiple types
    field_obj = pmap_field((str, int), (float, bool))
    expected_type = _make_pmap_field_type((str, int), (float, bool))
    assert field_obj.type == {expected_type}


# LLM-generated content at query #47
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(int, str)
    assert field.type == {_pmap_field_types[(int, str)]}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(int, str)]()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(int, str, optional=True)
    assert field.type == {optional_type(_pmap_field_types[(int, str)])}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(int, str)]()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 10 else (False, "Too many items")

    field = pmap_field(int, str, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test pmap_field factory
    field = pmap_field(int, str)
    pmap = field.factory({1: "one", 2: "two"})
    assert isinstance(pmap, CheckedPMap)
    assert pmap[1] == "one"
    assert pmap[2] == "two"

    # Test optional pmap_field factory with None
    field = pmap_field(int, str, optional=True)
    assert field.factory(None) is None
    pmap = field.factory({1: "one"})
    assert isinstance(pmap, CheckedPMap)
    assert pmap[1] == "one"

    # Test pmap_field with invalid types
    with pytest.raises(PTypeError):
        field = pmap_field(int, str)
        pmap = field.factory({1: "one", "invalid_key": "two"})

    with pytest.raises(PTypeError):
        field = pmap_field(int, str)
        pmap = field.factory({1: "one", 2: 2})

    # Test pmap_field type name
    field = pmap_field(int, str)
    pmap_type = _pmap_field_types[(int, str)]
    assert pmap_type.__name__ == "IntToStrPMap"

    # Test pmap_field pickle support
    field = pmap_field(int, str)
    pmap = field.factory({1: "one", 2: "two"})
    restored = pickle.loads(pickle.dumps(pmap))
    assert isinstance(restored, CheckedPMap)
    assert restored[1] == "one"
    assert restored[2] == "two"


# LLM-generated content at query #48
#--------------------------

```python
def test_check_type():
    # Test valid type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)

    # Test invalid type
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "string")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test multiple valid types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)
    check_type(TestClass, field, "test_field", "string")

    # Test invalid type with multiple valid types
    with pytest.raises(PTypeError):
        check_type(TestClass, field, "test_field", 10.5)

    # Test with no type constraint
    field = _PField(type=PFIELD_NO_TYPE, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "anything")


# LLM-generated content at query #49
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


# LLM-generated content at query #50
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


# LLM-generated content at query #51
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True

    # Test pmap_field with invariant
    def test_invariant(pmap):
        return (True, None)
    field = pmap_field(str, int, invariant=test_invariant)
    assert field.invariant == test_invariant

    # Test pmap_field factory
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert test_map == _make_pmap_field_type(str, int)({"a": 1, "b": 2})

    # Test optional pmap_field factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None

    # Test pmap_field with initial value
    field = pmap_field(str, int, initial={"x": 10})
    assert field.initial == _make_pmap_field_type(str, int)({"x": 10})

    # Test pmap_field type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type, field, "test_field", {"a": "not_int"})

    # Test pmap_field with multiple types
    field = pmap_field((str, int), (float, bool))
    TheMap = _make_pmap_field_type((str, int), (float, bool))
    assert field.type == {TheMap}


# LLM-generated content at query #52
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj, _PField)
    assert field_obj.type == {_pmap_field_types[(str, int)]}
    assert field_obj.mandatory is True
    assert field_obj.initial == _pmap_field_types[(str, int)]()
    assert field_obj.factory is _pmap_field_types[(str, int)].create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_pmap_field_types[(str, int)])}
    assert field_obj.factory(None) is None
    assert isinstance(field_obj.factory({}), _pmap_field_types[(str, int)])

    # Test with custom invariant
    custom_invariant = lambda x: (True, None)
    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test that the created map enforces types
    TestMap = _pmap_field_types[(str, int)]
    test_map = TestMap({"a": 1, "b": 2})
    assert test_map == {"a": 1, "b": 2}

    # Test type enforcement
    with pytest.raises(TypeError):
        TestMap({1: "a"})  # Wrong key type
    with pytest.raises(TypeError):
        TestMap({"a": "b"})  # Wrong value type

    # Test factory function
    factory_field = pmap_field(str, int, optional=True)
    assert factory_field.factory(None) is None
    assert factory_field.factory({"x": 1}) == TestMap({"x": 1})

    # Test initial value
    field_with_initial = pmap_field(str, int, initial={"default": 0})
    assert field_with_initial.initial == TestMap({"default": 0})

    # Test pickle support
    original_map = TestMap({"key": 42})
    pickled = pickle.dumps(original_map)
    unpickled = pickle.loads(pickled)
    assert unpickled == original_map
    assert type(unpickled) == TestMap


# LLM-generated content at query #53
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

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.args[0] == ("error_code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "error_code2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("error_code", "error_code2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.args[0] == ("error_code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"


# LLM-generated content at query #54
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")
    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test pmap_field factory
    field = pmap_field(str, int)
    pmap = field.factory({"a": 1, "b": 2})
    assert isinstance(pmap, CheckedPMap)
    assert pmap["a"] == 1
    assert pmap["b"] == 2

    # Test optional pmap_field factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    pmap = field.factory({"a": 1})
    assert isinstance(pmap, CheckedPMap)
    assert pmap["a"] == 1

    # Test pmap_field with initial value
    field = pmap_field(str, int, initial={"x": 10})
    initial_pmap = field.initial
    assert isinstance(initial_pmap, CheckedPMap)
    assert initial_pmap["x"] == 10

    # Test pmap_field type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field, "test_field", {"a": "not_int"})

    with pytest.raises(PTypeError):
        check_type(object, field, "test_field", {1: 2})  # key not str

    # Test pmap_field with multiple types
    field = pmap_field((str, int), (int, float))
    assert len(field.type) == 1
    map_type = next(iter(field.type))
    assert issubclass(map_type, CheckedPMap)

    # Test pmap_field serialization
    field = pmap_field(str, int)
    pmap = field.factory({"a": 1})
    serialized = serialize(PFIELD_NO_SERIALIZER, "test_format", pmap)
    assert serialized == pmap.serialize("test_format")


# LLM-generated content at query #55
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


# LLM-generated content at query #56
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

    # Test with CheckedType and custom serializer (should use custom serializer)
    assert serialize(custom_serializer, "xml", checked_value) == custom_serializer("xml", checked_value)


# LLM-generated content at query #57
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
    set_fields(dct, bases, '__fields__')
    assert dct['__fields__'] == {'a': 1, 'b': 2}

    # Test field collection with _PField instances
    field1 = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)
    field2 = _PField(type=str, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)

    class Base3:
        __fields__ = {'field1': field1}

    class Base4:
        __fields__ = {'field2': field2}

    class Child2(Base3, Base4):
        pass

    dct = {'__fields__': {}, 'field1': field1, 'field2': field2}
    bases = (Base3, Base4)
    set_fields(dct, bases, '__fields__')
    assert dct['__fields__'] == {'field1': field1, 'field2': field2}
    assert 'field1' not in dct
    assert 'field2' not in dct

    # Test empty bases
    dct = {'__fields__': {}}
    bases = ()
    set_fields(dct, bases, '__fields__')
    assert dct['__fields__'] == {}

    # Test with no existing __fields__ in bases
    class Base5:
        pass

    dct = {'__fields__': {}}
    bases = (Base5,)
    set_fields(dct, bases, '__fields__')
    assert dct['__fields__'] == {}


# LLM-generated content at query #58
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


# LLM-generated content at query #59
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test factory behavior
    field = pmap_field(str, int)
    pmap = field.factory({"a": 1, "b": 2})
    assert isinstance(pmap, CheckedPMap)
    assert pmap["a"] == 1
    assert pmap["b"] == 2

    # Test optional factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1})["a"] == 1

    # Test type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field, "test_field", {"a": "not_int"})

    # Test with initial value
    field = pmap_field(str, int, initial={"x": 10})
    assert field.initial == _make_pmap_field_type(str, int)({"x": 10})

    # Test invariant failure
    field = pmap_field(str, int, invariant=lambda p: (False, "Error") if len(p) > 0 else (True, None))
    with pytest.raises(InvariantException):
        check_global_invariants(_make_pmap_field_type(str, int)({"a": 1}), [field.invariant])


# LLM-generated content at query #60
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test when ignore_extra is False
    assert not is_field_ignore_extra_complaint(CheckedPSet, _PField(type=set, factory=lambda x: x), False)

    # Test when type is not a set
    assert not is_field_ignore_extra_complaint(CheckedPSet, _PField(type=int, factory=lambda x: x), True)

    # Test when factory does not have ignore_extra parameter
    assert not is_field_ignore_extra_complaint(CheckedPSet, _PField(type=set, factory=lambda x: x), True)

    # Test when all conditions are met
    def factory_with_ignore_extra(arg, ignore_extra=False):
        return arg
    assert is_field_ignore_extra_complaint(CheckedPSet, _PField(type=set, factory=factory_with_ignore_extra), True)

    # Test with CheckedPVector
    assert not is_field_ignore_extra_complaint(CheckedPVector, _PField(type=tuple, factory=lambda x: x), False)
    assert is_field_ignore_extra_complaint(CheckedPVector, _PField(type=tuple, factory=factory_with_ignore_extra), True)


# LLM-generated content at query #61
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

    assert dct['__fields__'] == {'a': 1, 'b': 2}
    assert 'c' not in dct

    # Test _PField handling
    field = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)
    dct = {'__fields__': {}, 'field1': field}
    bases = ()

    set_fields(dct, bases, name)

    assert dct['__fields__'] == {'field1': field}
    assert 'field1' not in dct

    # Test empty bases
    dct = {'__fields__': {}, 'a': 1}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct['__fields__'] == {}
    assert 'a' in dct

    # Test no existing __fields__ in bases
    class Base:
        pass

    dct = {'__fields__': {}, 'a': 1}
    bases = (Base,)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct['__fields__'] == {}
    assert 'a' in dct


# LLM-generated content at query #62
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    non_checked_value = [1, 2, 3]
    assert serialize(PFIELD_NO_SERIALIZER, "format", non_checked_value) == PFIELD_NO_SERIALIZER("format", non_checked_value)

    # Test with custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"

    assert serialize(custom_serializer, "format", "value") == "custom_format_value"

    # Test with CheckedType and custom serializer (should use custom serializer)
    assert serialize(custom_serializer, "format", checked_value) == custom_serializer("format", checked_value)


# LLM-generated content at query #63
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

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.args[0] == ("Error code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "Error code 2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("Error code", "Error code 2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"


# LLM-generated content at query #64
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

    # Test with CheckedType and custom serializer (should use custom serializer)
    assert serialize(custom_serializer, "json", checked_value) == custom_serializer("json", checked_value)


# LLM-generated content at query #65
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False (default)
    field = _PField(type={int}, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)
    assert not is_field_ignore_extra_complaint(CheckedPSet, field, False)
    assert not is_field_ignore_extra_complaint(CheckedPVector, field, False)

    # Test with ignore_extra=True but non-matching field type
    field = _PField(type={str}, invariant=lambda x: (True, None), initial="", mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)
    assert not is_field_ignore_extra_complaint(CheckedPSet, field, True)
    assert not is_field_ignore_extra_complaint(CheckedPVector, field, True)

    # Test with ignore_extra=True and matching field type but factory without ignore_extra
    field = _PField(type={int}, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)
    assert not is_field_ignore_extra_complaint(CheckedPSet, field, True)
    assert not is_field_ignore_extra_complaint(CheckedPVector, field, True)

    # Test with ignore_extra=True, matching field type, and factory with ignore_extra
    def factory_with_ignore_extra(x, _factory_fields=None, ignore_extra=False):
        return x
    field = _PField(type={int}, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=factory_with_ignore_extra, serializer=lambda _, v: v)
    assert is_field_ignore_extra_complaint(CheckedPSet, field, True)
    assert is_field_ignore_extra_complaint(CheckedPVector, field, True)

    # Test with empty type (should return False)
    field = _PField(type=set(), invariant=lambda x: (True, None), initial=None, mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)
    assert not is_field_ignore_extra_complaint(CheckedPSet, field, True)
    assert not is_field_ignore_extra_complaint(CheckedPVector, field, True)


# LLM-generated content at query #66
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj, _PField)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.factory is not None
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert optional_field.factory(None) is None
    assert optional_field.factory({}) == _make_pmap_field_type(str, int).create({})

    # Test with invariant
    def test_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    field_with_inv = pmap_field(str, int, invariant=test_invariant)
    assert field_with_inv.invariant is not PFIELD_NO_INVARIANT

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    factory_result = field_obj.factory(test_map)
    assert isinstance(factory_result, CheckedPMap)
    assert dict(factory_result) == test_map

    # Test type checking
    wrong_type_field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, wrong_type_field, "test_field", ["not", "a", "map"])

    # Test initial value
    assert isinstance(field_obj.initial, CheckedPMap)
    assert len(field_obj.initial) == 0

    # Test that the created type is properly registered for unpickling
    key_type = str
    value_type = int
    map_type = _make_pmap_field_type(key_type, value_type)
    assert _pmap_field_types[(key_type, value_type)] is map_type


# LLM-generated content at query #67
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_pmap_field_types[(str, int)]}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(str, int)]()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {_pmap_field_types[(str, int)], type(None)}
    assert optional_field.mandatory is True
    assert optional_field.initial == _pmap_field_types[(str, int)]()
    assert optional_field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    custom_invariant = lambda x: (True, "")
    field_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_invariant.invariant == custom_invariant

    # Test factory function
    test_map = {"a": 1, "b": 2}
    created_map = field.factory(test_map)
    assert isinstance(created_map, CheckedPMap)
    assert dict(created_map) == test_map

    # Test optional factory with None
    optional_created_map = optional_field.factory(None)
    assert optional_created_map is None

    # Test that the same key_type and value_type reuse the same class
    field2 = pmap_field(str, int)
    assert field.type == field2.type


# LLM-generated content at query #68
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert isinstance(field.initial, CheckedPMap)
    assert field.factory is _make_pmap_field_type(str, int).create
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial is None
    assert callable(field.factory)
    assert field.factory(None) is None
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 10 else (False, "Too many items")

    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test pmap_field with initial value
    field = pmap_field(str, int, initial={"a": 1, "b": 2})
    assert field.initial == _make_pmap_field_type(str, int)({"a": 1, "b": 2})

    # Test pmap_field factory with None when optional
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"x": 10}) == _make_pmap_field_type(str, int)({"x": 10})

    # Test pmap_field factory without optional
    field = pmap_field(str, int)
    assert field.factory({"x": 10}) == _make_pmap_field_type(str, int)({"x": 10})

    # Test that pmap_field creates correct CheckedPMap type
    TheMap = _make_pmap_field_type(str, int)
    assert TheMap.__key_type__ == str
    assert TheMap.__value_type__ == int
    assert issubclass(TheMap, CheckedPMap)

    # Test pmap_field with multiple key/value types
    field = pmap_field((str, int), (float, bool))
    assert field.type == {_make_pmap_field_type((str, int), (float, bool))}


# LLM-generated content at query #69
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
    test_map = {"a": 1, "b": 2}
    assert optional_field.factory(test_map) == _make_pmap_field_type(str, int)(test_map)

    # Test with invariant
    def test_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")

    field_with_inv = pmap_field(str, int, invariant=test_invariant)
    assert field_with_inv.invariant({"a": 1}) == (True, None)
    assert field_with_inv.invariant({}) == (False, "Map must not be empty")

    # Test factory function
    test_data = {"x": 10, "y": 20}
    result = field_obj.factory(test_data)
    assert isinstance(result, CheckedPMap)
    assert result == _make_pmap_field_type(str, int)(test_data)

    # Test with different types
    float_int_field = pmap_field(float, int)
    assert float_int_field.type == {_make_pmap_field_type(float, int)}
    test_data = {1.5: 100, 2.5: 200}
    result = float_int_field.factory(test_data)
    assert result == _make_pmap_field_type(float, int)(test_data)

    # Test initial value
    custom_initial = pmap_field(str, int, initial={"default": 0})
    assert custom_initial.initial == _make_pmap_field_type(str, int)({"default": 0})

    # Test that factory handles None when optional
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.factory(None) is None
    assert optional_field.factory({"a": 1}) == _make_pmap_field_type(str, int)({"a": 1})


# LLM-generated content at query #70
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def invariant1(obj):
        return True, None

    def invariant2(obj):
        return True, None

    check_global_invariants(subject, [invariant1, invariant2])

    # Test with failing invariant
    def failing_invariant(obj):
        return False, "error_code"

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant])

    assert excinfo.value.args[0] == ("error_code",)
    assert excinfo.value.args[1] == ()
    assert excinfo.value.args[2] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return False, "error_code2"

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert excinfo.value.args[0] == ("error_code", "error_code2")
    assert excinfo.value.args[1] == ()
    assert excinfo.value.args[2] == "Global invariant failed"


# LLM-generated content at query #71
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
    assert optional_field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert optional_field.factory(None) is None

    # Test with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")

    field_with_inv = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_inv.invariant is not PFIELD_NO_INVARIANT

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    factory_result = field_obj.factory(test_map)
    assert isinstance(factory_result, CheckedPMap)
    assert factory_result == test_map

    # Test with None when optional
    assert optional_field.factory(None) is None
    assert optional_field.factory({"x": 10}) == {"x": 10}

    # Test type checking
    wrong_type_field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), wrong_type_field, "test_field", {"a": "not_int"})

    # Test that the field works with PRecord-like usage
    class TestRecord:
        test_field = pmap_field(str, int)

    record = type('TestRecord', (), {'test_field': TestRecord.test_field})
    instance = type('Instance', (), {})
    setattr(instance, 'test_field', TestRecord.test_field.factory({"key": 123}))
    assert instance.test_field == {"key": 123}

    # Test invariant failure
    empty_map_field = pmap_field(str, int, invariant=custom_invariant)
    with pytest.raises(InvariantException):
        empty_map_field.invariant(empty_map_field.factory({}))


# LLM-generated content at query #72
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
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test with custom invariant
    custom_invariant = lambda x: (True, "")
    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test factory function
    field_obj = pmap_field(str, int)
    test_map = field_obj.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)
    assert test_map["a"] == 1
    assert test_map["b"] == 2

    # Test optional factory with None
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    test_map = field_obj.factory({"a": 1})
    assert test_map["a"] == 1

    # Test type checking
    field_obj = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type, field_obj, "test_field", {"a": "not_int"})

    # Test with multiple key/value types
    field_obj = pmap_field((str, int), (int, float))
    assert isinstance(field_obj, _PField)
    test_map = field_obj.factory({"a": 1, 2: 3.0})
    assert test_map["a"] == 1
    assert test_map[2] == 3.0


# LLM-generated content at query #73
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


# LLM-generated content at query #74
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
    set_fields(dct, bases, '__fields__')
    assert dct['__fields__'] == {'a': 1, 'b': 2}

    # Test field removal from dct
    dct = {'x': _PField(None, None, None, False, None, None), '__fields__': {}}
    bases = ()
    set_fields(dct, bases, '__fields__')
    assert '__fields__' in dct
    assert 'x' not in dct
    assert dct['__fields__'] == {'x': dct['__fields__']['x']}

    # Test with no base classes
    dct = {'__fields__': {}}
    bases = ()
    set_fields(dct, bases, '__fields__')
    assert dct['__fields__'] == {}

    # Test with empty __fields__ in base classes
    class Base3:
        __fields__ = {}

    dct = {'__fields__': {}}
    bases = (Base3,)
    set_fields(dct, bases, '__fields__')
    assert dct['__fields__'] == {}

    # Test with overlapping fields (should be overwritten)
    class Base4:
        __fields__ = {'a': 1}

    class Base5:
        __fields__ = {'a': 2}

    dct = {'__fields__': {}}
    bases = (Base4, Base5)
    set_fields(dct, bases, '__fields__')
    assert dct['__fields__'] == {'a': 2}


# LLM-generated content at query #75
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

    # Test with PFIELD_NO_SERIALIZER and non-CheckedType
    result = serialize(PFIELD_NO_SERIALIZER, "format", "plain_value")
    assert result == "plain_value"


# LLM-generated content at query #76
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda format, value: f"serialized_{value}"
    assert serialize(custom_serializer, "format", "test_value") == "serialized_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "test_value") == "test_value"

    # Test with CheckedType and custom serializer (should use custom serializer)
    assert serialize(custom_serializer, "format", checked_value) == custom_serializer("format", checked_value)


# LLM-generated content at query #77
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial() == _make_pmap_field_type(str, int)()
    assert callable(field.factory)
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert optional_field.factory(None) is None

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, "") if len(pmap) < 5 else (False, "Too many items")
    field_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_invariant.invariant == custom_invariant

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    created_map = field.factory(test_map)
    assert isinstance(created_map, CheckedPMap)
    assert created_map == test_map

    # Test optional factory with None
    assert optional_field.factory(None) is None
    assert optional_field.factory(test_map) == test_map

    # Test initial value
    initial_map = field.initial
    assert isinstance(initial_map, CheckedPMap)
    assert len(initial_map) == 0

    # Test with different types
    field2 = pmap_field(int, str)
    assert field2.type == {_make_pmap_field_type(int, str)}


# LLM-generated content at query #78
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True

    # Test factory with None for optional field
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test factory with non-None for non-optional field
    field = pmap_field(str, int)
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test invariant
    def test_invariant(pmap):
        return (True, None) if all(isinstance(k, str) and isinstance(v, int) for k, v in pmap.items()) else (False, "Invalid types")

    field = pmap_field(str, int, invariant=test_invariant)
    assert field.invariant({"a": 1}) == (True, None)
    assert field.invariant({"a": "1"}) == (False, "Invalid types")

    # Test with custom invariant
    field = pmap_field(str, int, invariant=lambda x: (len(x) > 0, "Empty map"))
    assert field.invariant({"a": 1}) == (True, None)
    assert field.invariant({}) == (False, "Empty map")


# LLM-generated content at query #79
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test when ignore_extra is False
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=CheckedPMap.create), False)

    # Test when field type is not a set
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=CheckedPMap.create), True)

    # Test when factory does not have ignore_extra parameter
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type={int}, factory=lambda x: x), True)

    # Test when all conditions are met
    def factory_with_ignore_extra(arg, ignore_extra=False):
        return CheckedPMap(arg)

    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type={int}, factory=factory_with_ignore_extra), True)

    # Test with CheckedPSet
    def set_factory_with_ignore_extra(arg, ignore_extra=False):
        return CheckedPSet(arg)

    assert is_field_ignore_extra_complaint(CheckedPSet, _PField(type={int}, factory=set_factory_with_ignore_extra), True)

    # Test with CheckedPVector
    def vector_factory_with_ignore_extra(arg, ignore_extra=False):
        return CheckedPVector(arg)

    assert is_field_ignore_extra_complaint(CheckedPVector, _PField(type={int}, factory=vector_factory_with_ignore_extra), True)


# LLM-generated content at query #80
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
    assert exc_info.value.args[0] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE_2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE", "ERROR_CODE_2")
    assert exc_info.value.args[0] == "Global invariant failed"

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args[0] == "Global invariant failed"


# LLM-generated content at query #81
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

    # Test with PFIELD_NO_SERIALIZER and non-CheckedType
    assert serialize(PFIELD_NO_SERIALIZER, "format", "plain_value") == "plain_value"


# LLM-generated content at query #82
#--------------------------

```python
def test_check_global_invariants():
    # Test with valid subject and invariants
    subject = {"a": 1, "b": 2}
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None)
    ]
    check_global_invariants(subject, invariants)

    # Test with invalid subject and invariants
    subject = {"a": 1, "b": 2}
    invariants = [
        lambda x: (False, "Error1"),
        lambda x: (True, None)
    ]
    with pytest.raises(InvariantException):
        check_global_invariants(subject, invariants)

    # Test with multiple error codes
    subject = {"a": 1, "b": 2}
    invariants = [
        lambda x: (False, "Error1"),
        lambda x: (False, "Error2")
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants)
    assert excinfo.value.error_codes == ("Error1", "Error2")

    # Test with empty invariants
    subject = {"a": 1, "b": 2}
    invariants = []
    check_global_invariants(subject, invariants)


# LLM-generated content at query #83
#--------------------------

```python
def test_field():
    # Test basic field creation
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory is False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=[int, str])
    assert f.type == {int, str}

    # Test field with invariant
    def test_inv(value):
        return (value > 0, "Value must be positive")
    f = field(invariant=test_inv)
    assert f.invariant(5) == (True, None)
    assert f.invariant(-1) == (False, "Value must be positive")

    # Test field with initial value
    f = field(initial=10)
    assert f.initial == 10

    # Test field with mandatory flag
    f = field(mandatory=True)
    assert f.mandatory is True

    # Test field with factory
    def test_factory(x):
        return x * 2
    f = field(factory=test_factory)
    assert f.factory(5) == 10

    # Test field with serializer
    def test_serializer(format, value):
        return str(value)
    f = field(serializer=test_serializer)
    assert f.serializer("json", 10) == "10"

    # Test field with all parameters
    f = field(type=int, invariant=test_inv, initial=10, mandatory=True, factory=test_factory, serializer=test_serializer)
    assert f.type == {int}
    assert f.invariant(5) == (True, None)
    assert f.initial == 10
    assert f.mandatory is True
    assert f.factory(5) == 10
    assert f.serializer("json", 10) == "10"

    # Test field with invalid type
    try:
        field(type=123)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Type parameter expected" in str(e)

    # Test field with invalid initial type
    try:
        field(type=int, initial="not an int")
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Initial has invalid type" in str(e)

    # Test field with non-callable invariant
    try:
        field(invariant="not callable")
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Invariant must be callable" in str(e)

    # Test field with non-callable factory
    try:
        field(factory="not callable")
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Factory must be callable" in str(e)

    # Test field with non-callable serializer
    try:
        field(serializer="not callable")
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Serializer must be callable" in str(e)


# LLM-generated content at query #84
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
    invariants = [invariant1, failing_invariant, invariant2]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)

    assert exc_info.value.error_codes == ("Error code 1",)


# LLM-generated content at query #85
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(int, str)
    assert field.type == {_pmap_field_types[(int, str)]}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(int, str)]()
    assert field.factory is _pmap_field_types[(int, str)].create
    assert field.invariant is PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(int, str, optional=True)
    assert field.type == {_pmap_field_types[(int, str)], type(None)}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(int, str)]()
    assert field.factory(None) is None
    assert field.factory({1: "a"}) == _pmap_field_types[(int, str)]({1: "a"})

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 10 else (False, "Too large")
    field = pmap_field(int, str, invariant=custom_invariant)
    assert field.invariant is not PFIELD_NO_INVARIANT
    assert field.invariant({1: "a"}) == (True, None)
    assert field.invariant({i: "a" for i in range(10)}) == (False, "Too large")

    # Test pmap_field factory with None when optional
    field = pmap_field(int, str, optional=True)
    assert field.factory(None) is None
    assert field.factory({1: "a"}) == _pmap_field_types[(int, str)]({1: "a"})

    # Test pmap_field with initial value
    field = pmap_field(int, str, initial={1: "a"})
    assert field.initial == _pmap_field_types[(int, str)]({1: "a"})

    # Test pmap_field type checking
    field = pmap_field(int, str)
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field, 'test_field', {1: 2})
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field, 'test_field', {"a": "b"})
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field, 'test_field', [1, 2, 3])

    # Test pmap_field with valid input
    try:
        check_type(type('TestClass', (), {}), field, 'test_field', {1: "a"})
        check_type(type('TestClass', (), {}), field, 'test_field', _pmap_field_types[(int, str)]({1: "a"}))
    except PTypeError:
        pytest.fail("Unexpected PTypeError")

    # Test pmap_field serialization
    field = pmap_field(int, str)
    pmap = _pmap_field_types[(int, str)]({1: "a", 2: "b"})
    assert serialize(PFIELD_NO_SERIALIZER, "test_format", pmap) == pmap.serialize("test_format")


# LLM-generated content at query #86
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False (default)
    field = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)
    assert not is_field_ignore_extra_complaint(int, field, False)
    assert not is_field_ignore_extra_complaint(int, field, None)

    # Test with ignore_extra=True but non-matching field type
    assert not is_field_ignore_extra_complaint(str, field, True)

    # Test with ignore_extra=True and matching field type but factory without ignore_extra param
    field_with_set = _PField(type={int}, invariant=lambda x: (True, None), initial={0}, mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)
    assert not is_field_ignore_extra_complaint(set, field_with_set, True)

    # Test with ignore_extra=True, matching field type, and factory with ignore_extra param
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    field_with_factory = _PField(type={int}, invariant=lambda x: (True, None), initial={0}, mandatory=False, factory=factory_with_ignore_extra, serializer=lambda _, v: v)
    assert is_field_ignore_extra_complaint(set, field_with_factory, True)

    # Test with tuple type
    field_tuple = _PField(type=(int,), invariant=lambda x: (True, None), initial=(0,), mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)
    assert not is_field_ignore_extra_complaint(tuple, field_tuple, True)

    # Test with list type
    field_list = _PField(type=[int], invariant=lambda x: (True, None), initial=[0], mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)
    assert not is_field_ignore_extra_complaint(list, field_list, True)


# LLM-generated content at query #87
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
        return (False, "ERROR_CODE")

    invariants = [failing_invariant]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)

    assert exc_info.value.error_codes == ("ERROR_CODE",)

    # Test with multiple failing invariants
    def failing_invariant1(obj):
        return (False, "ERROR_CODE1")

    def failing_invariant2(obj):
        return (False, "ERROR_CODE2")

    invariants = [failing_invariant1, failing_invariant2]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)

    assert exc_info.value.error_codes == ("ERROR_CODE1", "ERROR_CODE2")

    # Test with mixed passing and failing invariants
    invariants = [invariant1, failing_invariant, invariant2]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)

    assert exc_info.value.error_codes == ("ERROR_CODE",)


# LLM-generated content at query #88
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


# LLM-generated content at query #89
#--------------------------

```python
def test_is_field_ignore_extra_complaint():
    # Test with ignore_extra=False (default)
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set, factory=lambda x: x), False)
    assert not is_field_ignore_extra_complaint(CheckedPVector, _PField(type=set, factory=lambda x: x), False)
    assert not is_field_ignore_extra_complaint(CheckedPSet, _PField(type=set, factory=lambda x: x), False)

    # Test with ignore_extra=True but field type not matching
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=int, factory=lambda x: x), True)
    assert not is_field_ignore_extra_complaint(CheckedPVector, _PField(type=str, factory=lambda x: x), True)

    # Test with ignore_extra=True and field type matching but factory not accepting ignore_extra
    def factory_no_ignore_extra(x):
        return x
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set, factory=factory_no_ignore_extra), True)
    assert not is_field_ignore_extra_complaint(CheckedPVector, _PField(type=set, factory=factory_no_ignore_extra), True)

    # Test with ignore_extra=True, field type matching, and factory accepting ignore_extra
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    assert is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set, factory=factory_with_ignore_extra), True)
    assert is_field_ignore_extra_complaint(CheckedPVector, _PField(type=set, factory=factory_with_ignore_extra), True)
    assert is_field_ignore_extra_complaint(CheckedPSet, _PField(type=set, factory=factory_with_ignore_extra), True)

    # Test with empty type set
    assert not is_field_ignore_extra_complaint(CheckedPMap, _PField(type=set(), factory=factory_with_ignore_extra), True)


# LLM-generated content at query #90
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_spec = pmap_field(str, int)
    assert isinstance(field_spec, _PField)
    assert field_spec.type == {_pmap_field_types[(str, int)]}
    assert field_spec.mandatory is True
    assert field_spec.initial == _pmap_field_types[(str, int)]()

    # Test optional pmap_field
    field_spec = pmap_field(str, int, optional=True)
    assert field_spec.type == {optional_type(_pmap_field_types[(str, int)])}

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, "") if len(pmap) < 5 else (False, "Too many items")

    field_spec = pmap_field(str, int, invariant=custom_invariant)
    assert field_spec.invariant == custom_invariant

    # Test factory behavior
    field_spec = pmap_field(str, int)
    result = field_spec.factory({"a": 1, "b": 2})
    assert isinstance(result, CheckedPMap)
    assert result == {"a": 1, "b": 2}

    # Test optional factory with None
    field_spec = pmap_field(str, int, optional=True)
    assert field_spec.factory(None) is None
    assert field_spec.factory({"a": 1}) == {"a": 1}

    # Test type checking
    field_spec = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field_spec, "test_field", {"a": "not_int"})

    # Test with initial value
    field_spec = pmap_field(str, int, initial={"x": 10})
    assert field_spec.initial == {"x": 10}

    # Test pickling/unpickling
    field_spec = pmap_field(str, int)
    pmap = field_spec.factory({"key": 123})
    pickled = pickle.dumps(pmap)
    unpickled = pickle.loads(pickled)
    assert unpickled == pmap
    assert isinstance(unpickled, CheckedPMap)


# LLM-generated content at query #91
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


# LLM-generated content at query #92
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, 'format', checked_value) == checked_value.serialize('format')

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}_{format}"

    assert serialize(custom_serializer, 'json', 'test_value') == "serialized_test_value_json"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, 'format', 'plain_value') == 'plain_value'

    # Test with CheckedType and custom serializer
    assert serialize(custom_serializer, 'xml', checked_value) == "serialized_<CheckedPVector: [1, 2, 3]>_xml"


# LLM-generated content at query #93
#--------------------------

```python
def test_serialize():
    # Test serialization with no serializer (default case)
    value = "test"
    result = serialize(PFIELD_NO_SERIALIZER, "format", value)
    assert result == value

    # Test serialization with a custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}"
    result = serialize(custom_serializer, "format", value)
    assert result == "serialized_test"

    # Test serialization with a CheckedType and no serializer
    class TestCheckedType(CheckedType):
        def serialize(self, format):
            return f"checked_{format}"

    checked_value = TestCheckedType()
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == "checked_format"

    # Test serialization with a CheckedType and a custom serializer
    result = serialize(custom_serializer, "format", checked_value)
    assert result == "serialized_" + str(checked_value)


# LLM-generated content at query #94
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 10 else (False, "Too many items")

    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test factory behavior
    field = pmap_field(str, int)
    pmap = field.factory({"a": 1, "b": 2})
    assert isinstance(pmap, CheckedPMap)
    assert pmap["a"] == 1
    assert pmap["b"] == 2

    # Test optional factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    pmap = field.factory({"a": 1})
    assert isinstance(pmap, CheckedPMap)
    assert pmap["a"] == 1

    # Test type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type, field, "test_field", {"a": "not an int"})

    # Test with multiple types
    field = pmap_field((str, int), (int, float))
    pmap = field.factory({1: 1.0, "a": 2})
    assert pmap[1] == 1.0
    assert pmap["a"] == 2


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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

    # Test _PField handling
    class TestField(_PField):
        pass

    dct = {'__fields__': {}, 'field1': TestField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct[name] == {'field1': dct['field1']}
    assert 'field1' not in dct

    # Test empty bases
    dct = {'__fields__': {}, 'x': 10}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct[name] == {}
    assert 'x' in dct

    # Test with no existing __fields__ in dct
    dct = {'a': 1, 'b': 2}
    bases = ()
    name = '__fields__'

    set_fields(dct, bases, name)

    assert dct[name] == {}
    assert 'a' in dct
    assert 'b' in dct


# LLM-generated content at query #2
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

    # Test with no fields in bases
    class Base5:
        pass

    dct3 = {'__fields__': {}}
    bases3 = (Base5,)
    name3 = '__fields__'

    set_fields(dct3, bases3, name3)

    assert dct3[name3] == {}

    # Test with overlapping fields (last base wins)
    class Base6:
        __fields__ = {'a': 1}

    class Base7:
        __fields__ = {'a': 2}

    class Child3(Base6, Base7):
        pass

    dct4 = {'__fields__': {}}
    bases4 = (Base6, Base7)
    name4 = '__fields__'

    set_fields(dct4, bases4, name4)

    assert dct4[name4] == {'a': 2}


# LLM-generated content at query #3
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


# LLM-generated content at query #4
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
    assert excinfo.value.args == ("Global invariant failed",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "Error code 2")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert excinfo.value.error_codes == ("Error code", "Error code 2")
    assert excinfo.value.args == ("Global invariant failed",)


# LLM-generated content at query #5
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert optional_field.mandatory is True
    assert optional_field.initial == _make_pmap_field_type(str, int)()
    assert optional_field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    custom_invariant = lambda x: (True, "")
    field_with_invariant = pmap_field(str, int, invariant=custom_invariant)
    assert field_with_invariant.invariant == custom_invariant

    # Test factory function for non-optional field
    test_map = {"a": 1, "b": 2}
    factory_result = field.factory(test_map)
    assert isinstance(factory_result, CheckedPMap)
    assert dict(factory_result) == test_map

    # Test factory function for optional field with None
    optional_factory_result = optional_field.factory(None)
    assert optional_factory_result is None

    # Test factory function for optional field with value
    optional_factory_result = optional_field.factory(test_map)
    assert isinstance(optional_factory_result, CheckedPMap)
    assert dict(optional_factory_result) == test_map

    # Test initial value
    initial_map = field.initial
    assert isinstance(initial_map, CheckedPMap)
    assert dict(initial_map) == {}

    # Test type checking
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field, "test_field", {"a": "not_an_int"})


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
    class TestCheckedType(CheckedType):
        pass

    field = _PField(type={TestCheckedType}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", TestCheckedType())


# LLM-generated content at query #7
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
        return (False, "ERROR_CODE2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE", "ERROR_CODE2")
    assert exc_info.value.args == ("Global invariant failed",)

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args == ("Global invariant failed",)


# LLM-generated content at query #8
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    class TestCheckedType(CheckedType):
        pass

    checked_instance = TestCheckedType()
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_instance) == checked_instance.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"serialized_{value}"

    assert serialize(custom_serializer, "format", "test_value") == "serialized_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "test_value") == "test_value"


# LLM-generated content at query #9
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

    assert exc_info.value.args[0] == ("ERROR_CODE",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE_2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("ERROR_CODE", "ERROR_CODE_2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.args[0] == ("ERROR_CODE",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'


# LLM-generated content at query #10
#--------------------------

```python
def test_is_type_cls():
    # Test with set type
    assert is_type_cls(set, set) is True
    assert is_type_cls(set, {int}) is True

    # Test with non-set type
    assert is_type_cls(list, set) is False
    assert is_type_cls(dict, set) is False

    # Test with empty type
    assert is_type_cls(set, ()) is False

    # Test with single type in tuple
    assert is_type_cls(set, (set,)) is True
    assert is_type_cls(list, (set,)) is False

    # Test with multiple types in tuple
    assert is_type_cls(set, (set, list)) is True
    assert is_type_cls(list, (set, list)) is True
    assert is_type_cls(dict, (set, list)) is False

    # Test with subclass
    class MySet(set):
        pass
    assert is_type_cls(set, MySet) is True
    assert is_type_cls(MySet, set) is False


# LLM-generated content at query #11
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
    field = _PField(type={int, float}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10.5)

    # Test with non-matching type
    field = _PField(type={str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError) as exc_info:
        check_type(TestClass, field, "test_field", 10)
    assert exc_info.value.source_class == TestClass
    assert exc_info.value.field == "test_field"
    assert exc_info.value.expected_types == {str}
    assert exc_info.value.actual_type == int

    # Test with no type constraint
    field = _PField(type=(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "any_type")

    # Test with CheckedType subclass
    class TestCheckedType(CheckedType):
        pass

    field = _PField(type={TestCheckedType}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    instance = TestCheckedType()
    check_type(TestClass, field, "test_field", instance)


# LLM-generated content at query #12
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, 'format', checked_value)
    assert result == checked_value.serialize('format')

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    result = serialize(custom_serializer, 'format', 'test_value')
    assert result == 'serialized_test_value'

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, 'format', 'test_value')
    assert result == 'test_value'


# LLM-generated content at query #13
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


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", 10)

    # Test with non-matching type
    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError):
        check_type(TestClass, field, "test_field", "not an int")

    # Test with multiple allowed types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", "a string")

    # Test with no type restriction
    field = _PField(type=(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestClass, field, "test_field", {"a": "dict"})


# LLM-generated content at query #16
#--------------------------

```python
def test_field():
    # Test basic field creation
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory == False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=(int, str))
    assert f.type == {int, str}

    # Test field with invariant
    def test_inv(value):
        return (True, None)
    f = field(invariant=test_inv)
    assert f.invariant == test_inv

    # Test field with initial value
    f = field(initial=5)
    assert f.initial == 5

    # Test field with mandatory
    f = field(mandatory=True)
    assert f.mandatory == True

    # Test field with factory
    def test_factory(x):
        return x
    f = field(factory=test_factory)
    assert f.factory == test_factory

    # Test field with serializer
    def test_serializer(format, value):
        return value
    f = field(serializer=test_serializer)
    assert f.serializer == test_serializer

    # Test field with invalid type
    try:
        field(type=5)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid initial type
    try:
        field(type=int, initial="not an int")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable invariant
    try:
        field(invariant="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable factory
    try:
        field(factory="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable serializer
    try:
        field(serializer="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, 'format', checked_value)
    assert result == checked_value.serialize('format')

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    result = serialize(custom_serializer, 'format', 'test_value')
    assert result == "serialized_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, 'format', 'test_value')
    assert result == 'test_value'


# LLM-generated content at query #18
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.factory is not None
    assert field.mandatory is True
    assert isinstance(field.initial, CheckedPMap)

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.factory(None) is None

    # Test with initial value
    initial_map = {"a": 1, "b": 2}
    field = pmap_field(str, int, initial=initial_map)
    assert field.initial == _make_pmap_field_type(str, int).create(initial_map)

    # Test with invariant
    def test_invariant(pmap):
        return (True, None) if all(v > 0 for v in pmap.values()) else (False, "Values must be positive")

    field = pmap_field(str, int, invariant=test_invariant)
    assert field.invariant({"a": 1, "b": 2}) == (True, None)
    assert field.invariant({"a": -1, "b": 2}) == (False, "Values must be positive")

    # Test factory function
    field = pmap_field(str, int)
    test_map = {"x": 10, "y": 20}
    result = field.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test optional factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None


# LLM-generated content at query #19
#--------------------------

```python
def test_check_type():
    # Test with matching type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)

    check_type(TestClass, field, "test_field", 10)

    # Test with non-matching type
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "not_an_int")

    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test with multiple allowed types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)

    check_type(TestClass, field, "test_field", 10)
    check_type(TestClass, field, "test_field", "string")

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)

    check_type(TestClass, field, "test_field", "any_type")


# LLM-generated content at query #20
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.factory is _make_pmap_field_type(str, int).create
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.factory(None) is None
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")
    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test pmap_field with multiple types
    field = pmap_field((str, int), (float, bool))
    assert field.type == {_make_pmap_field_type((str, int), (float, bool))}

    # Test pmap_field factory behavior
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)
    assert test_map["a"] == 1
    assert test_map["b"] == 2

    # Test optional pmap_field factory behavior
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    test_map = field.factory({"a": 1})
    assert isinstance(test_map, CheckedPMap)
    assert test_map["a"] == 1

    # Test pmap_field with invalid types
    with pytest.raises(TypeError):
        pmap_field("not_a_type", int)
    with pytest.raises(TypeError):
        pmap_field(str, "not_a_type")

    # Test pmap_field invariant execution
    field = pmap_field(str, int, invariant=lambda x: (len(x) > 0, "Empty map"))
    test_map = field.factory({"a": 1})
    assert field.invariant(test_map) == (True, None)
    empty_map = field.factory({})
    assert field.invariant(empty_map) == (False, "Empty map")


# LLM-generated content at query #21
#--------------------------

```python
def test_check_type():
    class TestRecord:
        pass

    # Test with valid type
    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestRecord, field, "age", 25)  # Should not raise

    # Test with invalid type
    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError):
        check_type(TestRecord, field, "age", "25")

    # Test with multiple valid types
    field = _PField(type={int, float}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestRecord, field, "value", 25)  # Should not raise
    check_type(TestRecord, field, "value", 25.5)  # Should not raise

    # Test with multiple invalid types
    field = _PField(type={int, float}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    with pytest.raises(PTypeError):
        check_type(TestRecord, field, "value", "25")

    # Test with no type constraint
    field = _PField(type=set(), invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestRecord, field, "value", "anything")  # Should not raise

    # Test with subclass type
    class Parent:
        pass

    class Child(Parent):
        pass

    field = _PField(type={Parent}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)
    check_type(TestRecord, field, "parent", Child())  # Should not raise


# LLM-generated content at query #22
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariant
    def passing_invariant(obj):
        return True, None
    check_global_invariants(subject, [passing_invariant])

    # Test with multiple passing invariants
    def another_passing_invariant(obj):
        return True, None
    check_global_invariants(subject, [passing_invariant, another_passing_invariant])

    # Test with failing invariant
    def failing_invariant(obj):
        return False, "ERROR_CODE"
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])
    assert exc_info.value.error_codes == ("ERROR_CODE",)

    # Test with multiple failing invariants
    def another_failing_invariant(obj):
        return False, "ANOTHER_ERROR"
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, another_failing_invariant])
    assert exc_info.value.error_codes == ("ERROR_CODE", "ANOTHER_ERROR")

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [passing_invariant, failing_invariant, another_passing_invariant])
    assert exc_info.value.error_codes == ("ERROR_CODE",)


# LLM-generated content at query #23
#--------------------------

```python
def test_field():
    # Test basic field creation with no parameters
    f = field()
    assert f.type == set()
    assert f.invariant == PFIELD_NO_INVARIANT
    assert f.initial == PFIELD_NO_INITIAL
    assert f.mandatory == False
    assert f.factory == PFIELD_NO_FACTORY
    assert f.serializer == PFIELD_NO_SERIALIZER

    # Test field with type parameter
    f = field(type=int)
    assert f.type == {int}

    # Test field with multiple types
    f = field(type=(int, str))
    assert f.type == {int, str}

    # Test field with invariant
    def test_invariant(x):
        return (x > 0, "Value must be positive")
    f = field(invariant=test_invariant)
    assert f.invariant(5) == (True, None)
    assert f.invariant(-1) == (False, "Value must be positive")

    # Test field with initial value
    f = field(initial=10)
    assert f.initial == 10

    # Test field with mandatory flag
    f = field(mandatory=True)
    assert f.mandatory == True

    # Test field with factory
    def test_factory(x):
        return x * 2
    f = field(factory=test_factory)
    assert f.factory(5) == 10

    # Test field with serializer
    def test_serializer(format, x):
        return str(x)
    f = field(serializer=test_serializer)
    assert f.serializer("json", 10) == "10"

    # Test field with all parameters
    f = field(type=int, invariant=test_invariant, initial=10, mandatory=True,
              factory=test_factory, serializer=test_serializer)
    assert f.type == {int}
    assert f.invariant(5) == (True, None)
    assert f.initial == 10
    assert f.mandatory == True
    assert f.factory(5) == 10
    assert f.serializer("json", 10) == "10"

    # Test field with invalid type parameter
    try:
        field(type=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with invalid initial type
    try:
        field(type=int, initial="string")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable invariant
    try:
        field(invariant="not a function")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable factory
    try:
        field(factory="not a function")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test field with non-callable serializer
    try:
        field(serializer="not a function")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, 'format', checked_value) == checked_value.serialize('format')

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    non_checked_value = [1, 2, 3]
    assert serialize(PFIELD_NO_SERIALIZER, 'format', non_checked_value) == PFIELD_NO_SERIALIZER('format', non_checked_value)

    # Test with custom serializer
    def custom_serializer(format, value):
        return f"custom_{format}_{value}"

    assert serialize(custom_serializer, 'format', 'value') == "custom_format_value"


# LLM-generated content at query #25
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

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.args[0] == ("Error code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "Error code 2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("Error code", "Error code 2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'


# LLM-generated content at query #26
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_pmap_field_types[(str, int)]}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(str, int)]()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {_pmap_field_types[(str, int)], type(None)}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(str, int)]()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    custom_invariant = lambda x: (True, None)
    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.type == {_pmap_field_types[(str, int)]}
    assert field.mandatory is True
    assert field.initial == _pmap_field_types[(str, int)]()
    assert field.invariant == custom_invariant

    # Test pmap_field factory
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert isinstance(test_map, _pmap_field_types[(str, int)])
    assert test_map == {"a": 1, "b": 2}

    # Test optional pmap_field factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    test_map = field.factory({"a": 1, "b": 2})
    assert isinstance(test_map, _pmap_field_types[(str, int)])
    assert test_map == {"a": 1, "b": 2}

    # Test pmap_field with invalid types
    with pytest.raises(TypeError):
        pmap_field(123, int)  # Invalid key_type
    with pytest.raises(TypeError):
        pmap_field(str, 123)  # Invalid value_type

    # Test pmap_field type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type, field, "test_field", {"a": "not_an_int"})

    # Test pmap_field with valid types
    field = pmap_field(str, int)
    check_type(type, field, "test_field", {"a": 1, "b": 2})

    # Test pmap_field with optional None
    field = pmap_field(str, int, optional=True)
    check_type(type, field, "test_field", None)


# LLM-generated content at query #27
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

    assert exc_info.value.args[0] == ("ERROR_CODE",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE_2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("ERROR_CODE", "ERROR_CODE_2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.args[0] == ("ERROR_CODE",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"


# LLM-generated content at query #28
#--------------------------

```python
def test_set_fields():
    # Test basic field collection from base classes
    class Base1:
        _fields = {'a': 1}

    class Base2:
        _fields = {'b': 2}

    class Child(Base1, Base2):
        pass

    dct = {'_fields': {}, 'c': 3}
    bases = (Base1, Base2)
    name = '_fields'

    set_fields(dct, bases, name)

    assert dct['_fields'] == {'a': 1, 'b': 2}
    assert 'c' not in dct

    # Test with _PField instances
    field1 = _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)
    field2 = _PField(type=str, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)

    class Base3:
        _fields = {'d': field1}

    class Base4:
        _fields = {'e': field2}

    class Child2(Base3, Base4):
        pass

    dct2 = {'_fields': {}, 'f': field1}
    bases2 = (Base3, Base4)
    name2 = '_fields'

    set_fields(dct2, bases2, name2)

    assert dct2['_fields'] == {'d': field1, 'e': field2}
    assert 'f' not in dct2

    # Test with empty base classes
    class EmptyBase:
        pass

    dct3 = {'_fields': {}, 'g': 4}
    bases3 = (EmptyBase,)
    name3 = '_fields'

    set_fields(dct3, bases3, name3)

    assert dct3['_fields'] == {}
    assert 'g' not in dct3

    # Test with no existing _fields in dct
    dct4 = {'h': 5}
    bases4 = (Base1,)
    name4 = '_fields'

    set_fields(dct4, bases4, name4)

    assert dct4['_fields'] == {'a': 1}
    assert 'h' not in dct4


# LLM-generated content at query #29
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    assert serialize(custom_serializer, "format", "test_value") == "serialized_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "test_value") == "test_value"


# LLM-generated content at query #30
#--------------------------

```python
def test_check_type():
    # Test valid type
    class TestClass:
        pass

    field = _PField(type={int}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)

    check_type(TestClass, field, "test_field", 10)
    check_type(TestClass, field, "test_field", 0)

    # Test invalid type
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", "string")
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int}
    assert excinfo.value.actual_type == str

    # Test multiple valid types
    field = _PField(type={int, str}, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)

    check_type(TestClass, field, "test_field", 10)
    check_type(TestClass, field, "test_field", "string")

    # Test invalid type with multiple valid types
    with pytest.raises(PTypeError) as excinfo:
        check_type(TestClass, field, "test_field", 10.5)
    assert excinfo.value.source_class == TestClass
    assert excinfo.value.field == "test_field"
    assert excinfo.value.expected_types == {int, str}
    assert excinfo.value.actual_type == float

    # Test with no type constraint
    field = _PField(type=PFIELD_NO_TYPE, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL,
                    mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER)

    check_type(TestClass, field, "test_field", 10)
    check_type(TestClass, field, "test_field", "string")
    check_type(TestClass, field, "test_field", 10.5)
    check_type(TestClass, field, "test_field", TestClass())


# LLM-generated content at query #31
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
    assert 'field3' in dct['__fields__']
    assert 'field3' not in dct


# LLM-generated content at query #32
#--------------------------

```python
def test_set_fields():
    class Base1:
        pass

    class Base2:
        pass

    class TestClass(Base1, Base2):
        pass

    # Test case 1: Empty bases and empty dct
    dct = {}
    bases = ()
    name = '_test_fields'
    set_fields(dct, bases, name)
    assert name in dct
    assert dct[name] == {}

    # Test case 2: Bases with fields
    class BaseWithFields1:
        _test_fields = {'field1': 'value1'}

    class BaseWithFields2:
        _test_fields = {'field2': 'value2'}

    dct = {}
    bases = (BaseWithFields1, BaseWithFields2)
    set_fields(dct, bases, name)
    assert name in dct
    assert dct[name] == {'field1': 'value1', 'field2': 'value2'}

    # Test case 3: Dct with _PField instances
    dct = {'field3': _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)}
    bases = ()
    set_fields(dct, bases, name)
    assert name in dct
    assert dct[name] == {'field3': dct['field3']}
    assert 'field3' not in dct

    # Test case 4: Mixed scenario
    class BaseWithFields3:
        _test_fields = {'field4': 'value4'}

    dct = {'field5': _PField(type=str, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)}
    bases = (BaseWithFields3,)
    set_fields(dct, bases, name)
    assert name in dct
    assert dct[name] == {'field4': 'value4', 'field5': dct['field5']}
    assert 'field5' not in dct


# LLM-generated content at query #33
#--------------------------

```python
def test_check_global_invariants():
    # Test with valid subject and invariants
    subject = {"key": "value"}
    invariants = [lambda x: (True, None)]
    check_global_invariants(subject, invariants)

    # Test with invalid subject and invariants
    subject = {"key": "value"}
    invariants = [lambda x: (False, "error_code")]
    with pytest.raises(InvariantException):
        check_global_invariants(subject, invariants)

    # Test with multiple invariants, some valid and some invalid
    subject = {"key": "value"}
    invariants = [
        lambda x: (True, None),
        lambda x: (False, "error_code1"),
        lambda x: (False, "error_code2")
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants)
    assert excinfo.value.error_codes == ("error_code1", "error_code2")

    # Test with empty invariants list
    subject = {"key": "value"}
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
        return (False, "ERROR_CODE")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant])

    assert excinfo.value.error_codes == ("ERROR_CODE",)
    assert excinfo.value.message == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE2")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert excinfo.value.error_codes == ("ERROR_CODE", "ERROR_CODE2")
    assert excinfo.value.message == "Global invariant failed"

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert excinfo.value.error_codes == ("ERROR_CODE",)
    assert excinfo.value.message == "Global invariant failed"


# LLM-generated content at query #35
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
        return (False, "Error code 1")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.error_codes == ("Error code 1",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "Error code 2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.error_codes == ("Error code 1", "Error code 2")


# LLM-generated content at query #36
#--------------------------

```python
def test_set_fields():
    # Test basic field collection from base classes
    class A:
        __fields__ = {'a': _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)}

    class B:
        __fields__ = {'b': _PField(type=str, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)}

    class C(A, B):
        pass

    dct = {}
    bases = (A, B)
    name = '__fields__'
    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert 'a' in dct['__fields__']
    assert 'b' in dct['__fields__']
    assert dct['__fields__']['a'].type == {int}
    assert dct['__fields__']['b'].type == {str}

    # Test field removal from dct
    assert 'a' not in dct
    assert 'b' not in dct

    # Test with no fields in base classes
    class D:
        pass

    class E:
        pass

    class F(D, E):
        pass

    dct = {}
    bases = (D, E)
    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert len(dct['__fields__']) == 0

    # Test with overlapping fields (should be combined)
    class G:
        __fields__ = {'x': _PField(type=int, invariant=lambda x: (True, None), initial=0, mandatory=True, factory=lambda x: x, serializer=lambda _, v: v)}

    class H:
        __fields__ = {'x': _PField(type=str, invariant=lambda x: (True, None), initial='', mandatory=False, factory=lambda x: x, serializer=lambda _, v: v)}

    class I(G, H):
        pass

    dct = {}
    bases = (G, H)
    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert 'x' in dct['__fields__']
    # The last occurrence should win (from H)
    assert dct['__fields__']['x'].type == {str}


# LLM-generated content at query #37
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def invariant1(obj):
        return True, None

    def invariant2(obj):
        return True, None

    check_global_invariants(subject, [invariant1, invariant2])

    # Test with failing invariant
    def failing_invariant(obj):
        return False, "error_code"

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.args[0] == ("error_code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return False, "error_code2"

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("error_code", "error_code2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"


# LLM-generated content at query #38
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


# LLM-generated content at query #39
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"custom_{val}"
    result = serialize(custom_serializer, "format", "test_value")
    assert result == "custom_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "format", "test_value")
    assert result == "test_value"


# LLM-generated content at query #40
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


# LLM-generated content at query #41
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def invariant1(obj):
        return True, None

    def invariant2(obj):
        return True, None

    check_global_invariants(subject, [invariant1, invariant2])

    # Test with failing invariant
    def failing_invariant(obj):
        return False, "Error code"

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.args[0] == ("Error code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return False, "Error code 2"

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("Error code", "Error code 2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'


# LLM-generated content at query #42
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert isinstance(field_obj.initial, CheckedPMap)
    assert field_obj.factory is _make_pmap_field_type(str, int).create

    # Test optional pmap_field creation
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.mandatory is True

    # Test factory with None for optional field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert isinstance(field_obj.factory({}), CheckedPMap)

    # Test invariant is passed through
    def test_invariant(x):
        return (True, None)
    field_obj = pmap_field(str, int, invariant=test_invariant)
    assert field_obj.invariant == test_invariant

    # Test field creation with custom initial value
    field_obj = pmap_field(str, int, initial={"a": 1})
    assert field_obj.initial == _make_pmap_field_type(str, int).create({"a": 1})

    # Test that the created field type is cached
    type1 = _make_pmap_field_type(str, int)
    type2 = _make_pmap_field_type(str, int)
    assert type1 is type2

    # Test that different key/value types create different field types
    type1 = _make_pmap_field_type(str, int)
    type2 = _make_pmap_field_type(int, str)
    assert type1 is not type2

    # Test that the created type has correct attributes
    TheMap = _make_pmap_field_type(str, int)
    assert TheMap.__key_type__ == str
    assert TheMap.__value_type__ == int
    assert "StrToIntPMap" in TheMap.__name__

    # Test that the created type can be pickled/unpickled
    import pickle
    original = TheMap({"a": 1, "b": 2})
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)
    assert dict(original) == dict(restored)


# LLM-generated content at query #43
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}"
    assert serialize(custom_serializer, "format", "test_value") == "serialized_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "test_value") == "test_value"


# LLM-generated content at query #44
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

    # Test with CheckedType and custom serializer (should use custom serializer)
    result = serialize(custom_serializer, "json", checked_value)
    assert result == custom_serializer("json", checked_value)


# LLM-generated content at query #45
#--------------------------

```python
def test_set_fields():
    class Base1:
        __fields__ = {'a': field(type=int)}

    class Base2:
        __fields__ = {'b': field(type=str)}

    class Child(Base1, Base2):
        __fields__ = {'c': field(type=float)}

    dct = {'c': field(type=float)}
    bases = (Base1, Base2)
    name = '__fields__'

    set_fields(dct, bases, name)

    assert '__fields__' in dct
    assert 'c' not in dct
    assert 'a' in dct['__fields__']
    assert 'b' in dct['__fields__']
    assert 'c' in dct['__fields__']
    assert isinstance(dct['__fields__']['a'], _PField)
    assert isinstance(dct['__fields__']['b'], _PField)
    assert isinstance(dct['__fields__']['c'], _PField)


# LLM-generated content at query #46
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"custom_{val}"
    result = serialize(custom_serializer, "format", "test_value")
    assert result == "custom_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "format", "test_value")
    assert result == "test_value"


# LLM-generated content at query #47
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

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args == ("Global invariant failed",)


# LLM-generated content at query #48
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.factory is _make_pmap_field_type(str, int).create
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.factory(None) is None
    assert field.factory({'a': 1}) == _make_pmap_field_type(str, int).create({'a': 1})

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")

    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test field type checking
    field = pmap_field(str, int)
    test_map = field.factory({'a': 1, 'b': 2})
    assert isinstance(test_map, CheckedPMap)
    assert test_map['a'] == 1
    assert test_map['b'] == 2

    # Test with different types
    field = pmap_field(int, str)
    test_map = field.factory({1: 'a', 2: 'b'})
    assert isinstance(test_map, CheckedPMap)
    assert test_map[1] == 'a'
    assert test_map[2] == 'b'

    # Test that invalid types raise PTypeError
    field = pmap_field(str, int)
    try:
        field.factory({1: 'a'})  # int key instead of str
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass

    try:
        field.factory({'a': 'b'})  # str value instead of int
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass

    # Test that None is handled correctly for optional fields
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None


# LLM-generated content at query #49
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True

    # Test pmap_field with custom invariant
    custom_invariant = lambda x: (True, "")
    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test pmap_field factory
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)
    assert test_map == {"a": 1, "b": 2}

    # Test optional pmap_field factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == {"a": 1}

    # Test pmap_field with initial value
    field = pmap_field(str, int, initial={"x": 10})
    assert field.initial == {"x": 10}

    # Test pmap_field type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field, "test_field", {"a": "not_int"})

    with pytest.raises(PTypeError):
        check_type(object, field, "test_field", {1: 2})  # key not str

    # Test pmap_field with multiple types
    field = pmap_field((str, int), (int, float))
    test_map = field.factory({"a": 1, 2: 3.0})
    assert test_map == {"a": 1, 2: 3.0}


# LLM-generated content at query #50
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_pmap_field_types[(str, int)]}
    assert field_obj.mandatory is True
    assert field_obj.initial == _pmap_field_types[(str, int)]()
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {_pmap_field_types[(str, int)], type(None)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _pmap_field_types[(str, int)]()
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 10 else (False, "Too many items")

    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test pmap_field factory
    field_obj = pmap_field(str, int)
    pmap = field_obj.factory({"a": 1, "b": 2})
    assert isinstance(pmap, CheckedPMap)
    assert pmap["a"] == 1
    assert pmap["b"] == 2

    # Test optional pmap_field factory with None
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert field_obj.factory({"a": 1})["a"] == 1

    # Test pmap_field with initial value
    field_obj = pmap_field(str, int, initial={"x": 10})
    assert field_obj.initial == _pmap_field_types[(str, int)]({"x": 10})

    # Test pmap_field type checking
    field_obj = pmap_field(str, int)
    pmap = field_obj.factory({"a": 1, "b": 2})
    assert isinstance(pmap, CheckedPMap)
    try:
        field_obj.factory({"a": "not an int"})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

    # Test pmap_field with multiple types
    field_obj = pmap_field((str, int), (int, float))
    assert (str, int) in field_obj.type or {str, int} in field_obj.type
    assert (int, float) in field_obj.type or {int, float} in field_obj.type


# LLM-generated content at query #51
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    custom_inv = lambda x: (True, None)
    field = pmap_field(str, int, invariant=custom_inv)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == custom_inv

    # Test pmap_field factory
    field = pmap_field(str, int)
    pm = field.factory({"a": 1, "b": 2})
    assert isinstance(pm, CheckedPMap)
    assert pm["a"] == 1
    assert pm["b"] == 2

    # Test optional pmap_field factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    pm = field.factory({"a": 1, "b": 2})
    assert isinstance(pm, CheckedPMap)
    assert pm["a"] == 1
    assert pm["b"] == 2

    # Test pmap_field with invalid types
    try:
        pmap_field("not a type", int)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        pmap_field(str, "not a type")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test pmap_field with invalid invariant
    try:
        pmap_field(str, int, invariant="not callable")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #52
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

    # Test with invalid invariant
    subject = {"a": 3, "b": 2}
    invariants = [
        lambda x: (x["a"] < x["b"], "a should be less than b")
    ]
    with pytest.raises(InvariantException):
        check_global_invariants(subject, invariants)

    # Test with multiple invalid invariants
    subject = {"a": 3, "b": 2, "c": 1}
    invariants = [
        lambda x: (x["a"] < x["b"], "a should be less than b"),
        lambda x: (x["b"] < x["c"], "b should be less than c")
    ]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)
    assert len(exc_info.value.error_codes) == 2


# LLM-generated content at query #53
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

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.args[0] == ("Error code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "Error code 2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("Error code", "Error code 2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"


# LLM-generated content at query #54
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.factory is not None
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test pmap_field with invariant
    def invariant(pmap):
        return (True, "") if all(v > 0 for v in pmap.values()) else (False, "Values must be positive")

    field = pmap_field(str, int, invariant=invariant)
    assert field.invariant({"a": 1}) == (True, "")
    assert field.invariant({"a": -1}) == (False, "Values must be positive")

    # Test pmap_field with initial value
    field = pmap_field(str, int, initial={"a": 1})
    assert field.initial == _make_pmap_field_type(str, int).create({"a": 1})

    # Test pmap_field with multiple key and value types
    field = pmap_field((str, int), (int, float))
    assert field.type == {_make_pmap_field_type((str, int), (int, float))}

    # Test pmap_field factory with None when optional
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test pmap_field factory without None when not optional
    field = pmap_field(str, int, optional=False)
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})


# LLM-generated content at query #55
#--------------------------

```python
def test_check_global_invariants():
    # Test with valid invariants
    subject = {"a": 1, "b": 2}
    invariants = [
        lambda x: (True, None),
        lambda x: (True, None)
    ]
    check_global_invariants(subject, invariants)

    # Test with one failing invariant
    invariants = [
        lambda x: (True, None),
        lambda x: (False, "error1")
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants)
    assert "error1" in str(excinfo.value)

    # Test with multiple failing invariants
    invariants = [
        lambda x: (False, "error1"),
        lambda x: (False, "error2")
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants)
    assert "error1" in str(excinfo.value) and "error2" in str(excinfo.value)

    # Test with empty invariants list
    check_global_invariants(subject, [])


# LLM-generated content at query #56
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    check_global_invariants(subject, [])

    # Test with passing invariants
    def invariant1(obj):
        return True, None

    def invariant2(obj):
        return True, None

    check_global_invariants(subject, [invariant1, invariant2])

    # Test with failing invariant
    def failing_invariant(obj):
        return False, "Error code"

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.args[0] == ("Error code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return False, "Error code 2"

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("Error code", "Error code 2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'


# LLM-generated content at query #57
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True

    # Test pmap_field with custom invariant
    custom_invariant = lambda x: (True, None)
    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test pmap_field factory with None for optional field
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test pmap_field factory with non-None value
    field = pmap_field(str, int)
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test pmap_field with initial value
    field = pmap_field(str, int, initial={"a": 1})
    assert field.initial == _make_pmap_field_type(str, int).create({"a": 1})

    # Test pmap_field type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type, field, "test_field", {"a": "not an int"})

    # Test pmap_field with multiple key/value types
    field = pmap_field((str, int), (int, float))
    assert field.type == {_make_pmap_field_type((str, int), (int, float))}


# LLM-generated content at query #58
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
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])
    assert exc_info.value.error_codes == ("error_code",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "error_code2")
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])
    assert exc_info.value.error_codes == ("error_code", "error_code2")

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])
    assert exc_info.value.error_codes == ("error_code",)


# LLM-generated content at query #59
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

    assert exc_info.value.args[0] == ("ERROR_CODE",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("ERROR_CODE", "ERROR_CODE2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"


# LLM-generated content at query #60
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert isinstance(field.initial, CheckedPMap)
    assert field.factory is _make_pmap_field_type(str, int).create
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial is None
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, "") if len(pmap) < 5 else (False, "Too many items")

    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test field type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field, "test_field", {"a": "not_an_int"})

    # Test field with valid value
    valid_map = _make_pmap_field_type(str, int).create({"a": 1, "b": 2})
    check_type(object, field, "test_field", valid_map)  # Should not raise

    # Test field with None when optional
    field = pmap_field(str, int, optional=True)
    check_type(object, field, "test_field", None)  # Should not raise

    # Test field with None when not optional
    field = pmap_field(str, int, optional=False)
    with pytest.raises(PTypeError):
        check_type(object, field, "test_field", None)


# LLM-generated content at query #61
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


# LLM-generated content at query #62
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

    assert excinfo.value.error_codes == ("error_code",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "error_code2")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert excinfo.value.error_codes == ("error_code", "error_code2")


# LLM-generated content at query #63
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_spec = pmap_field(str, int)
    assert isinstance(field_spec, _PField)
    assert field_spec.type == {_make_pmap_field_type(str, int)}
    assert field_spec.mandatory is True
    assert field_spec.initial == _make_pmap_field_type(str, int)()

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
    factory = field_spec.factory
    result = factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test optional factory with None
    optional_factory = optional_field.factory
    assert optional_factory(None) is None
    assert dict(optional_factory(test_map)) == test_map

    # Test with initial value
    initial_field = pmap_field(str, int, initial={"x": 10})
    assert dict(initial_field.initial) == {"x": 10}

    # Test type checking
    with pytest.raises(PTypeError):
        check_type(object, field_spec, "test_field", {"a": "not_int"})

    # Test invariant violation
    with pytest.raises(InvariantException):
        field_with_inv.invariant({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})

    # Test that the generated type is cached
    type1 = _make_pmap_field_type(str, int)
    type2 = _make_pmap_field_type(str, int)
    assert type1 is type2


# LLM-generated content at query #64
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "format", checked_value)
    assert result == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"{format}_{value}"

    result = serialize(custom_serializer, "json", "data")
    assert result == "json_data"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    result = serialize(PFIELD_NO_SERIALIZER, "format", "plain_value")
    assert result == "plain_value"


# LLM-generated content at query #65
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

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant])

    assert excinfo.value.error_codes == ("ERROR_CODE",)
    assert excinfo.value.args == ("Global invariant failed",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ANOTHER_ERROR")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert excinfo.value.error_codes == ("ERROR_CODE", "ANOTHER_ERROR")
    assert excinfo.value.args == ("Global invariant failed",)

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert excinfo.value.error_codes == ("ERROR_CODE",)
    assert excinfo.value.args == ("Global invariant failed",)


# LLM-generated content at query #66
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.factory is not None
    assert field.mandatory is True
    assert isinstance(field.initial, CheckedPMap)
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")
    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test factory with None when optional
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int).create({"a": 1})

    # Test initial value
    field = pmap_field(str, int, initial={"a": 1})
    assert field.initial == _make_pmap_field_type(str, int).create({"a": 1})

    # Test type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field, "test", {"a": "not an int"})

    # Test with multiple key/value types
    field = pmap_field((str, int), (int, float))
    assert field.type == {_make_pmap_field_type((str, int), (int, float))}

    # Test that the generated type is cached
    type1 = _make_pmap_field_type(str, int)
    type2 = _make_pmap_field_type(str, int)
    assert type1 is type2


# LLM-generated content at query #67
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
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {_pmap_field_types[(str, int)], type(None)}
    assert optional_field.factory(None) is None
    assert optional_field.factory({"a": 1}) == _pmap_field_types[(str, int)]({"a": 1})

    # Test with custom invariant
    def custom_inv(map_obj):
        return (True, "") if len(map_obj) < 5 else (False, "Too many items")
    inv_field = pmap_field(str, int, invariant=custom_inv)
    assert inv_field.invariant({"a": 1}) == (True, "")
    assert inv_field.invariant({str(i): i for i in range(10)}) == (False, "Too many items")

    # Test factory behavior
    test_map = {"key1": 1, "key2": 2}
    factory_result = field_obj.factory(test_map)
    assert isinstance(factory_result, CheckedPMap)
    assert dict(factory_result) == test_map

    # Test initial value
    assert isinstance(field_obj.initial, CheckedPMap)
    assert len(field_obj.initial) == 0

    # Test type checking
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field_obj, "test_field", [1, 2, 3])

    # Test that the same key/value types return the same field type
    field1 = pmap_field(str, int)
    field2 = pmap_field(str, int)
    assert field1.type == field2.type


# LLM-generated content at query #68
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"custom_{val}"
    assert serialize(custom_serializer, "format", "test_value") == "custom_test_value"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "test_value") == "test_value"


# LLM-generated content at query #69
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
    assert field.factory(None) is None

    # Test with invariant
    def test_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")

    field = pmap_field(str, int, invariant=test_invariant)
    assert field.invariant is not PFIELD_NO_INVARIANT

    # Test factory behavior
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert test_map == _make_pmap_field_type(str, int)({"a": 1, "b": 2})

    # Test with None when optional
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == _make_pmap_field_type(str, int)({"a": 1})

    # Test type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type('TestClass', (), {}), field, "test_field", {"a": "not_int"})

    # Test initial value
    field = pmap_field(str, int, initial={"x": 1})
    assert field.initial == _make_pmap_field_type(str, int)({"x": 1})

    # Test that the field is properly configured
    field = pmap_field(str, int)
    assert field.mandatory is True
    assert field.serializer is PFIELD_NO_SERIALIZER


# LLM-generated content at query #70
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert isinstance(field, _PField)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True

    # Test with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")

    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test factory behavior
    field = pmap_field(str, int)
    pmap = field.factory({"a": 1, "b": 2})
    assert isinstance(pmap, CheckedPMap)
    assert pmap["a"] == 1
    assert pmap["b"] == 2

    # Test optional factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1})["a"] == 1

    # Test type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(str, field, "test_field", {"a": "not_an_int"})

    # Test with initial value
    field = pmap_field(str, int, initial={"x": 10})
    assert field.initial == _make_pmap_field_type(str, int)({"x": 10})

    # Test invariant failure
    field = pmap_field(str, int, invariant=custom_invariant)
    with pytest.raises(InvariantException):
        check_global_invariants(_make_pmap_field_type(str, int)(), [field.invariant])


# LLM-generated content at query #71
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.factory is _make_pmap_field_type(str, int).create
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()
    assert field.factory(None) is None
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 5 else (False, "Too many items")
    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test pmap_field with None value when optional
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None

    # Test pmap_field with actual data
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert test_map == _make_pmap_field_type(str, int)({"a": 1, "b": 2})

    # Test pmap_field type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type, field, "test_field", {"a": "not_an_int"})

    # Test pmap_field with multiple types
    field = pmap_field((str, int), (float, bool))
    assert field.type == {_make_pmap_field_type((str, int), (float, bool))}


# LLM-generated content at query #72
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
        return (False, "Error code 1")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant])

    assert excinfo.value.error_codes == ("Error code 1",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "Error code 2")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert excinfo.value.error_codes == ("Error code 1", "Error code 2")

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert excinfo.value.error_codes == ("Error code 1",)


# LLM-generated content at query #73
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


# LLM-generated content at query #74
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    result = serialize(PFIELD_NO_SERIALIZER, "test_format", checked_value)
    assert result == checked_value.serialize("test_format")

    # Test with non-CheckedType and custom serializer
    custom_serializer = lambda fmt, val: f"serialized_{val}_{fmt}"
    result = serialize(custom_serializer, "json", "test_value")
    assert result == "serialized_test_value_json"

    # Test with CheckedType and custom serializer (should use custom serializer)
    result = serialize(custom_serializer, "json", checked_value)
    assert result == custom_serializer("json", checked_value)


# LLM-generated content at query #75
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


# LLM-generated content at query #76
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert isinstance(field, _PField)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()

    # Test optional pmap_field
    optional_field = pmap_field(str, int, optional=True)
    assert optional_field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert optional_field.initial == _make_pmap_field_type(str, int)()

    # Test with custom invariant
    def custom_invariant(pmap):
        return (len(pmap) < 5, "Too many items")

    invariant_field = pmap_field(str, int, invariant=custom_invariant)
    assert invariant_field.invariant == custom_invariant

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    result = field.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test optional factory with None
    result = optional_field.factory(None)
    assert result is None

    # Test initial value
    initial_map = field.initial
    assert isinstance(initial_map, CheckedPMap)
    assert len(initial_map) == 0

    # Test type checking
    with pytest.raises(PTypeError):
        check_type(object, field, "test_field", {"a": "not_int"})

    # Test with valid types
    check_type(object, field, "test_field", {"a": 1, "b": 2})

    # Test invariant violation
    with pytest.raises(InvariantException):
        test_map = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        invariant_field.invariant(test_map)


# LLM-generated content at query #77
#--------------------------

```python
def test_check_global_invariants():
    # Test with no invariants
    subject = object()
    invariants = []
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with invariants that pass
    def invariant1(obj):
        return (True, None)

    def invariant2(obj):
        return (True, None)

    invariants = [invariant1, invariant2]
    check_global_invariants(subject, invariants)  # Should not raise

    # Test with invariants that fail
    def invariant3(obj):
        return (False, "Error1")

    def invariant4(obj):
        return (False, "Error2")

    invariants = [invariant3, invariant4]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)

    assert exc_info.value.error_codes == ("Error1", "Error2")
    assert exc_info.value.message == "Global invariant failed"

    # Test with mixed invariants (some pass, some fail)
    invariants = [invariant1, invariant3, invariant2, invariant4]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)

    assert exc_info.value.error_codes == ("Error1", "Error2")
    assert exc_info.value.message == "Global invariant failed"


# LLM-generated content at query #78
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
    assert field_obj.factory(None) is None
    assert isinstance(field_obj.factory({}), _make_pmap_field_type(str, int))

    # Test with initial value
    field_obj = pmap_field(str, int, initial={'a': 1})
    assert field_obj.initial == _make_pmap_field_type(str, int).create({'a': 1})

    # Test with invariant
    def test_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")

    field_obj = pmap_field(str, int, invariant=test_invariant)
    assert field_obj.invariant({'a': 1}) == (True, None)
    assert field_obj.invariant({}) == (False, "Map must not be empty")

    # Test factory behavior
    field_obj = pmap_field(str, int)
    assert field_obj.factory({'a': 1, 'b': 2}) == _make_pmap_field_type(str, int).create({'a': 1, 'b': 2})

    # Test with optional and factory
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert field_obj.factory({'a': 1}) == _make_pmap_field_type(str, int).create({'a': 1})

    # Test type checking
    with pytest.raises(PTypeError):
        field_obj = pmap_field(str, int)
        check_type(type('TestClass', (), {}), field_obj, 'test_field', {'a': 'not_int'})


# LLM-generated content at query #79
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

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args == ("Global invariant failed",)


# LLM-generated content at query #80
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
    assert exc_info.value.args[0] == ("ERROR_CODE",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE2")
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])
    assert exc_info.value.args[0] == ("ERROR_CODE", "ERROR_CODE2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == 'Global invariant failed'


# LLM-generated content at query #81
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert isinstance(field.initial, CheckedPMap)

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.factory(None) is None

    # Test pmap_field with invariant
    def test_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")
    field = pmap_field(str, int, invariant=test_invariant)
    assert field.invariant({"a": 1}) == (True, None)
    assert field.invariant({}) == (False, "Map must not be empty")

    # Test pmap_field factory
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert test_map["a"] == 1
    assert test_map["b"] == 2

    # Test pmap_field with invalid types
    with pytest.raises(TypeError):
        pmap_field("not a type", int)
    with pytest.raises(TypeError):
        pmap_field(str, "not a type")

    # Test pmap_field serialization
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1})
    serialized = field.serializer("json", test_map)
    assert serialized == test_map.serialize("json")


# LLM-generated content at query #82
#--------------------------

```python
def test_serialize():
    # Test with CheckedType and PFIELD_NO_SERIALIZER
    checked_value = CheckedPVector([1, 2, 3])
    assert serialize(PFIELD_NO_SERIALIZER, "format", checked_value) == checked_value.serialize("format")

    # Test with non-CheckedType and custom serializer
    def custom_serializer(format, value):
        return f"{format}:{value}"
    assert serialize(custom_serializer, "json", 42) == "json:42"

    # Test with non-CheckedType and PFIELD_NO_SERIALIZER
    assert serialize(PFIELD_NO_SERIALIZER, "format", "test") == "test"


# LLM-generated content at query #83
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

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.args[0] == ("error_code",)
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "error_code2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.args[0] == ("error_code", "error_code2")
    assert exc_info.value.args[1] == ()
    assert exc_info.value.args[2] == "Global invariant failed"


# LLM-generated content at query #84
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

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant])

    assert excinfo.value.error_codes == ("ERROR_CODE",)
    assert excinfo.value.args[0] == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE2")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert excinfo.value.error_codes == ("ERROR_CODE", "ERROR_CODE2")
    assert excinfo.value.args[0] == "Global invariant failed"


# LLM-generated content at query #85
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

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE_2")
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])
    assert exc_info.value.error_codes == ("ERROR_CODE", "ERROR_CODE_2")

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])
    assert exc_info.value.error_codes == ("ERROR_CODE",)


# LLM-generated content at query #86
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert isinstance(field.initial, CheckedPMap)
    assert field.factory is _make_pmap_field_type(str, int).create
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial is None
    assert callable(field.factory)
    assert field.factory(None) is None
    assert field.invariant == PFIELD_NO_INVARIANT

    # Test pmap_field with custom invariant
    custom_invariant = lambda x: (True, None)
    field = pmap_field(str, int, invariant=custom_invariant)
    assert field.invariant == custom_invariant

    # Test pmap_field with multiple key/value types
    field = pmap_field((str, int), (float, bool))
    assert field.type == {_make_pmap_field_type((str, int), (float, bool))}

    # Test pmap_field factory behavior
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert test_map["a"] == 1
    assert test_map["b"] == 2

    # Test optional pmap_field factory behavior
    field = pmap_field(str, int, optional=True)
    assert field.factory({"a": 1})["a"] == 1
    assert field.factory(None) is None

    # Test pmap_field initial value
    field = pmap_field(str, int)
    assert isinstance(field.initial, CheckedPMap)
    assert len(field.initial) == 0

    # Test pmap_field with initial value
    field = pmap_field(str, int, initial={"x": 10})
    assert field.initial["x"] == 10

    # Test pmap_field type checking
    field = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(type, field, "test_field", {"a": "not_int"})

    # Test pmap_field with invalid types
    with pytest.raises(TypeError):
        pmap_field("not_a_type", int)

    with pytest.raises(TypeError):
        pmap_field(str, "not_a_type")


# LLM-generated content at query #87
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

    result = serialize(custom_serializer, "json", "test_value")
    assert result == "json:test_value"

    # Test with CheckedType and custom serializer (should use custom serializer)
    result = serialize(custom_serializer, "json", checked_value)
    assert result == custom_serializer("json", checked_value)


# LLM-generated content at query #88
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

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant])

    assert excinfo.value.error_codes == ("ERROR_CODE",)
    assert excinfo.value.args == ("Global invariant failed",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE2")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert excinfo.value.error_codes == ("ERROR_CODE", "ERROR_CODE2")
    assert excinfo.value.args == ("Global invariant failed",)


# LLM-generated content at query #89
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(int, str)
    assert field_obj.type == {_make_pmap_field_type(int, str)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(int, str)()

    # Test optional pmap_field
    optional_field = pmap_field(int, str, optional=True)
    assert optional_field.type == {optional_type(_make_pmap_field_type(int, str))}
    assert optional_field.mandatory is True

    # Test with invariant
    def test_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")
    field_with_inv = pmap_field(int, str, invariant=test_invariant)
    assert field_with_inv.invariant is not PFIELD_NO_INVARIANT

    # Test factory function
    test_map = {1: "one", 2: "two"}
    result = field_obj.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test optional factory with None
    result = optional_field.factory(None)
    assert result is None

    # Test initial value
    initial_map = field_obj.initial
    assert isinstance(initial_map, CheckedPMap)
    assert len(initial_map) == 0

    # Test with custom types
    class CustomKey: pass
    class CustomValue: pass
    custom_field = pmap_field(CustomKey, CustomValue)
    assert custom_field.type == {_make_pmap_field_type(CustomKey, CustomValue)}

    # Test that factory creates correct type
    custom_instance = custom_field.factory({CustomKey(): CustomValue()})
    assert isinstance(custom_instance, CheckedPMap)
    assert len(custom_instance) == 1


# LLM-generated content at query #90
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

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.error_codes == ("Error code",)
    assert exc_info.value.message == "Global invariant failed"

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "Error code 2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.error_codes == ("Error code", "Error code 2")
    assert exc_info.value.message == "Global invariant failed"


# LLM-generated content at query #91
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

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.error_codes == ("Error code",)

    # Test with multiple failing invariants
    def failing_invariant1(obj):
        return (False, "Error code 1")

    def failing_invariant2(obj):
        return (False, "Error code 2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant1, failing_invariant2])

    assert exc_info.value.error_codes == ("Error code 1", "Error code 2")

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.error_codes == ("Error code",)


# LLM-generated content at query #92
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
        return (False, "ERROR_CODE")

    invariants = [failing_invariant]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args == ("Global invariant failed",)

    # Test with multiple failing invariants
    def failing_invariant1(obj):
        return (False, "ERROR_CODE1")

    def failing_invariant2(obj):
        return (False, "ERROR_CODE2")

    invariants = [failing_invariant1, failing_invariant2]
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, invariants)

    assert exc_info.value.error_codes == ("ERROR_CODE1", "ERROR_CODE2")
    assert exc_info.value.args == ("Global invariant failed",)


# LLM-generated content at query #93
#--------------------------

```python
def test_check_global_invariants():
    # Test with valid subject and invariants
    subject = {"a": 1, "b": 2}
    invariants = [lambda x: (True, None)]
    check_global_invariants(subject, invariants)

    # Test with invalid subject and invariants
    subject = {"a": 1, "b": 2}
    invariants = [lambda x: (False, "error_code")]
    with pytest.raises(InvariantException):
        check_global_invariants(subject, invariants)

    # Test with multiple invariants, some valid and some invalid
    subject = {"a": 1, "b": 2}
    invariants = [
        lambda x: (True, None),
        lambda x: (False, "error_code1"),
        lambda x: (False, "error_code2")
    ]
    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, invariants)
    assert excinfo.value.error_codes == ("error_code1", "error_code2")


# LLM-generated content at query #94
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert isinstance(field, _PField)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial() == _make_pmap_field_type(str, int)()
    assert field.factory is _make_pmap_field_type(str, int).create

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.factory(None) is None
    assert isinstance(field.factory({}), _make_pmap_field_type(str, int))

    # Test pmap_field with invariant
    def test_invariant(pmap):
        return (len(pmap) > 0, "Map must not be empty")

    field = pmap_field(str, int, invariant=test_invariant)
    assert field.invariant is not PFIELD_NO_INVARIANT
    result, error = field.invariant({})
    assert result is False
    assert error == "Map must not be empty"

    # Test pmap_field with initial value
    field = pmap_field(str, int, initial={"a": 1})
    assert field.initial == {"a": 1}

    # Test pmap_field with multiple types
    field = pmap_field((str, int), (int, str))
    assert len(field.type) == 1
    map_type = next(iter(field.type))
    assert issubclass(map_type, CheckedPMap)

    # Test pmap_field factory with valid input
    field = pmap_field(str, int)
    result = field.factory({"a": 1, "b": 2})
    assert isinstance(result, CheckedPMap)
    assert result == {"a": 1, "b": 2}

    # Test pmap_field factory with invalid input (should raise TypeError)
    field = pmap_field(str, int)
    try:
        field.factory({1: "a"})
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test pmap_field with None when optional
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None


# LLM-generated content at query #95
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field = pmap_field(str, int)
    assert field.type == {_make_pmap_field_type(str, int)}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()

    # Test optional pmap_field
    field = pmap_field(str, int, optional=True)
    assert field.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field.mandatory is True
    assert field.initial == _make_pmap_field_type(str, int)()

    # Test with invariant
    def test_invariant(pmap):
        return (True, None)
    field = pmap_field(str, int, invariant=test_invariant)
    assert field.invariant == test_invariant

    # Test factory behavior
    field = pmap_field(str, int)
    test_map = field.factory({"a": 1, "b": 2})
    assert isinstance(test_map, CheckedPMap)
    assert test_map == {"a": 1, "b": 2}

    # Test optional factory with None
    field = pmap_field(str, int, optional=True)
    assert field.factory(None) is None
    assert field.factory({"a": 1}) == {"a": 1}

    # Test type checking
    field = pmap_field(str, int)
    assert isinstance(field.type, set)
    assert len(field.type) == 1
    map_type = next(iter(field.type))
    assert issubclass(map_type, CheckedPMap)

    # Test that the field is properly configured
    assert field.mandatory is True
    assert callable(field.factory)
    assert callable(field.invariant)


# LLM-generated content at query #96
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

    # Test with multiple passing invariants
    def another_passing_invariant(obj):
        return (True, None)

    check_global_invariants(subject, [passing_invariant, another_passing_invariant])

    # Test with failing invariant
    def failing_invariant(obj):
        return (False, "ERROR_CODE")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant])

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args == ("Global invariant failed",)

    # Test with multiple failing invariants
    def another_failing_invariant(obj):
        return (False, "ANOTHER_ERROR")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, another_failing_invariant])

    assert exc_info.value.error_codes == ("ERROR_CODE", "ANOTHER_ERROR")
    assert exc_info.value.args == ("Global invariant failed",)

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [passing_invariant, failing_invariant, another_passing_invariant])

    assert exc_info.value.error_codes == ("ERROR_CODE",)
    assert exc_info.value.args == ("Global invariant failed",)


# LLM-generated content at query #97
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

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "ERROR_CODE_2")

    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE", "ERROR_CODE_2")

    # Test with mixed passing and failing invariants
    with pytest.raises(InvariantException) as exc_info:
        check_global_invariants(subject, [invariant1, failing_invariant, invariant2])

    assert exc_info.value.error_codes == ("ERROR_CODE",)


# LLM-generated content at query #98
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert field_obj.type == {_make_pmap_field_type(str, int)}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.factory == _make_pmap_field_type(str, int).create
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {optional_type(_make_pmap_field_type(str, int))}
    assert field_obj.mandatory is True
    assert field_obj.initial == _make_pmap_field_type(str, int)()
    assert field_obj.factory(None) is None
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test with custom invariant
    custom_invariant = lambda x: (True, None)
    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant == custom_invariant

    # Test factory behavior
    test_map = {"a": 1, "b": 2}
    field_obj = pmap_field(str, int)
    result = field_obj.factory(test_map)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == test_map

    # Test optional factory with None
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert isinstance(field_obj.factory(test_map), CheckedPMap)

    # Test initial value
    field_obj = pmap_field(str, int, optional=False)
    assert field_obj.initial == _make_pmap_field_type(str, int)()

    # Test type checking
    field_obj = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field_obj, "test_field", {"a": "not_int"})

    # Test with multiple types in union
    field_obj = pmap_field((str, int), (float, bool))
    assert field_obj.type == {_make_pmap_field_type((str, int), (float, bool))}


# LLM-generated content at query #99
#--------------------------

```python
def test_pmap_field():
    # Test basic pmap_field creation
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj, _PField)
    assert field_obj.type == {CheckedPMap}
    assert field_obj.mandatory is True
    assert field_obj.initial == CheckedPMap()
    assert callable(field_obj.factory)
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test optional pmap_field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.type == {CheckedPMap, type(None)}
    assert field_obj.mandatory is True
    assert field_obj.initial == CheckedPMap()
    assert callable(field_obj.factory)
    assert field_obj.invariant == PFIELD_NO_INVARIANT

    # Test factory with None for optional field
    field_obj = pmap_field(str, int, optional=True)
    assert field_obj.factory(None) is None
    assert isinstance(field_obj.factory({}), CheckedPMap)

    # Test factory with non-None value
    field_obj = pmap_field(str, int)
    assert isinstance(field_obj.factory({'a': 1}), CheckedPMap)

    # Test with custom invariant
    def custom_invariant(pmap):
        return (True, None) if len(pmap) < 2 else (False, "Too many items")
    field_obj = pmap_field(str, int, invariant=custom_invariant)
    assert field_obj.invariant({'a': 1}) == (True, None)
    assert field_obj.invariant({'a': 1, 'b': 2}) == (False, "Too many items")

    # Test type checking
    field_obj = pmap_field(str, int)
    with pytest.raises(PTypeError):
        check_type(object, field_obj, 'test_field', 123)
    with pytest.raises(PTypeError):
        check_type(object, field_obj, 'test_field', {'a': 'b'})
    check_type(object, field_obj, 'test_field', CheckedPMap({'a': 1}))

    # Test _make_pmap_field_type caching
    type1 = _make_pmap_field_type(str, int)
    type2 = _make_pmap_field_type(str, int)
    assert type1 is type2
    assert type1.__name__ == "StrToIntPMap"


# LLM-generated content at query #100
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


# LLM-generated content at query #101
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

    assert excinfo.value.error_codes == ("error_code",)

    # Test with multiple failing invariants
    def failing_invariant2(obj):
        return (False, "error_code2")

    with pytest.raises(InvariantException) as excinfo:
        check_global_invariants(subject, [failing_invariant, failing_invariant2])

    assert excinfo.value.error_codes == ("error_code", "error_code2")


