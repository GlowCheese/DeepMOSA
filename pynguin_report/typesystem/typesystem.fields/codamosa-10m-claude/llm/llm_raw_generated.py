####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Array_validate():
    # Test with None and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None

    # Test with None and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"

    # Test with non-list type (dict)
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate({"key": "value"})
    assert exc_info.value.messages()[0].code == "type"

    # Test with empty list and min_items=1
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"

    # Test with list length less than min_items
    field = Array(min_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.messages()[0].code == "min_items"

    # Test with list length greater than max_items
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test with exact_items mismatch
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "exact_items"

    # Test with exact_items match
    field = Array(exact_items=2)
    result = field.validate([1, 2])
    assert result == [1, 2]

    # Test with simple items validator
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test with items validator that fails
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "invalid", 3])
    assert exc_info.value.messages()[0].code == "type"

    # Test with tuple items validators
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "test"])
    assert result == [1, "test"]

    # Test with tuple items validators and additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "test", "extra"])
    assert exc_info.value.messages()[0].code == "additional_items"

    # Test with tuple items validators and additional_items=Field
    field = Array(items=[Integer(), String()], additional_items=Integer())
    result = field.validate([1, "test", 3])
    assert result == [1, "test", 3]

    # Test with unique_items=True and duplicate items
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert exc_info.value.messages()[0].code == "unique_items"

    # Test with unique_items=True and no duplicates
    field = Array(unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test with unique_items=True and duplicate strings
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(["a", "b", "a"])
    assert exc_info.value.messages()[0].code == "unique_items"

    # Test with no validators and no constraints
    field = Array()
    result = field.validate([1, "test", None, {"key": "value"}])
    assert result == [1, "test", None, {"key": "value"}]

    # Test with items validator and validation errors
    field = Array(items=Integer(minimum=0))
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, -5, 3])
    assert exc_info.value.messages()[0].code == "minimum"

    # Test with nested Array
    field = Array(items=Array(items=Integer()))
    result = field.validate([[1, 2], [3, 4]])
    assert result == [[1, 2], [3, 4]]

    # Test with min_items=0 and empty list
    field = Array(min_items=0)
    result = field.validate([])
    assert result == []

    # Test with max_items=0 and empty list
    field = Array(max_items=0)
    result = field.validate([])
    assert result == []

    # Test with max_items=0 and non-empty list
    field = Array(max_items=0)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test with items list and additional_items=True
    field = Array(items=[Integer(), String()], additional_items=True)
    result = field.validate([1, "test", "extra", 100])
    assert result == [1, "test", "extra", 100]

    # Test validation error message includes index
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "invalid", 3])
    error = exc_info.value.messages()[0]
    assert error.index == [1]


# LLM-generated content at query #2
#--------------------------

```python
def test_Array():
    # Test basic instantiation with no arguments
    field = Array()
    assert field.items is None
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False

    # Test with single Field item
    item_field = String()
    field = Array(items=item_field)
    assert field.items is item_field
    assert field.additional_items is False

    # Test with list of Fields
    field1 = String()
    field2 = Integer()
    field = Array(items=[field1, field2])
    assert field.items == [field1, field2]
    assert field.min_items == 2
    assert field.max_items == 2

    # Test with list of Fields and additional_items=True
    field = Array(items=[String(), Integer()], additional_items=True)
    assert field.min_items == 2
    assert field.max_items is None

    # Test with list of Fields and additional_items as Field
    additional_field = Boolean()
    field = Array(items=[String()], additional_items=additional_field)
    assert field.additional_items is additional_field
    assert field.min_items == 1
    assert field.max_items is None

    # Test with min_items and max_items
    field = Array(items=String(), min_items=1, max_items=10)
    assert field.min_items == 1
    assert field.max_items == 10

    # Test with exact_items
    field = Array(items=String(), exact_items=5)
    assert field.min_items == 5
    assert field.max_items == 5

    # Test with unique_items
    field = Array(items=String(), unique_items=True)
    assert field.unique_items is True

    # Test with tuple of items (should be converted to list)
    field1 = String()
    field2 = Integer()
    field = Array(items=(field1, field2))
    assert isinstance(field.items, list)
    assert field.items == [field1, field2]

    # Test with all parameters
    field = Array(
        items=String(),
        additional_items=False,
        min_items=2,
        max_items=20,
        unique_items=True,
        allow_null=True,
        allow_blank=True
    )
    assert field.items is not None
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 20
    assert field.unique_items is True
    assert field.allow_null is True
    assert field.allow_blank is True

    # Test exact_items overrides min_items and max_items
    field = Array(items=String(), min_items=1, max_items=10, exact_items=5)
    assert field.min_items == 5
    assert field.max_items == 5

    # Test list of items with additional_items=False sets max_items
    field = Array(items=[String(), Integer(), Boolean()], additional_items=False)
    assert field.min_items == 3
    assert field.max_items == 3


# LLM-generated content at query #3
#--------------------------

```python
def test_Union_validate():
    # Test 1: Valid value matching first type
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate("hello")
    assert result == "hello"

    # Test 2: Valid value matching second type
    result = union_field.validate(42)
    assert result == 42

    # Test 3: None value with allow_null=True
    union_field_nullable = Union(any_of=[String(), Integer()], allow_null=True)
    result = union_field_nullable.validate(None)
    assert result is None

    # Test 4: None value without allow_null should raise
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test 5: Value matching neither type should raise union error
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "union"

    # Test 6: allow_null is True if any child allows null
    child_nullable = String(allow_null=True)
    union_field = Union(any_of=[child_nullable, Integer()])
    assert union_field.allow_null is True

    # Test 7: Validation error from child with correct type is propagated
    union_field = Union(any_of=[String(max_length=3), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("toolong")
    messages = exc_info.value.messages()
    assert messages[0].code == "max_length"

    # Test 8: Multiple children with type errors should raise union error
    union_field = Union(any_of=[String(max_length=2), Integer(minimum=100)])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("toolong")
    assert exc_info.value.messages()[0].code == "union"

    # Test 9: Float value with Integer type should work
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate(3.0)
    assert result == 3

    # Test 10: Empty string with String field
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate("")
    assert result == ""

    # Test 11: Boolean is rejected by both String and Integer
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(True)
    assert exc_info.value.messages()[0].code == "union"

    # Test 12: First matching type wins
    union_field = Union(any_of=[String(), Float()])
    result = union_field.validate("123")
    assert result == "123"
    assert isinstance(result, str)

    # Test 13: Complex nested validation with Object
    obj_field = Object(properties={"name": String()})
    union_field = Union(any_of=[obj_field, String()])
    result = union_field.validate({"name": "test"})
    assert result == {"name": "test"}

    # Test 14: Validation with custom constraints on child
    int_field = Integer(minimum=0, maximum=100)
    union_field = Union(any_of=[String(), int_field])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(150)
    assert exc_info.value.messages()[0].code == "maximum"

    # Test 15: None in child with indexed error
    union_field = Union(any_of=[Array(items=String()), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate({"key": "value"})
    assert exc_info.value.messages()[0].code == "union"


# LLM-generated content at query #4
#--------------------------

```python
def test_Array_validate():
    # Test with None and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None

    # Test with None and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"

    # Test with empty list and min_items=1
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"

    # Test with list shorter than min_items
    field = Array(min_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.messages()[0].code == "min_items"

    # Test with list longer than max_items
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test with exact_items mismatch
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "exact_items"

    # Test with valid exact_items
    field = Array(exact_items=2)
    assert field.validate([1, 2]) == [1, 2]

    # Test with single Field items validator
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]

    # Test with single Field items validator and invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "invalid", 3])
    assert len(exc_info.value.messages()) > 0

    # Test with list of Field items validators
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "test"]) == [1, "test"]

    # Test with list of Field items validators and additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "test", 3])
    assert len(exc_info.value.messages()) > 0

    # Test with list of Field items validators and additional_items as Field
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "extra"]) == [1, "extra"]

    # Test with unique_items=True and duplicate items
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert any(msg.code == "unique_items" for msg in exc_info.value.messages())

    # Test with unique_items=True and no duplicates
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

    # Test with simple valid array
    field = Array()
    assert field.validate([]) == []
    assert field.validate([1, "test", True]) == [1, "test", True]

    # Test with nested validators and errors
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, "not_int"])
    errors = exc_info.value.messages()
    assert len(errors) > 0
    assert errors[0].index == [2]

    # Test with multiple items validators
    field = Array(items=[Integer(), String(), Boolean()])
    result = field.validate([42, "hello", True])
    assert result == [42, "hello", True]

    # Test min_items and max_items together
    field = Array(min_items=2, max_items=4)
    assert field.validate([1, 2]) == [1, 2]
    assert field.validate([1, 2, 3, 4]) == [1, 2, 3, 4]
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "min_items"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3, 4, 5])
    assert exc_info.value.messages()[0].code == "max_items"


# LLM-generated content at query #5
#--------------------------

```python
def test_Const():
    # Test basic initialization with a constant value
    const_field = Const(const=42)
    assert const_field.const == 42
    
    # Test initialization with None
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    
    # Test initialization with string
    const_field_str = Const(const="hello")
    assert const_field_str.const == "hello"
    
    # Test initialization with list
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    # Test initialization with dict
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)
    
    # Test that other kwargs are passed to parent
    const_field_with_description = Const(const=100, description="A constant field")
    assert const_field_with_description.const == 100
    assert const_field_with_description.description == "A constant field"
    
    # Test with boolean constant
    const_field_bool = Const(const=True)
    assert const_field_bool.const is True
    
    # Test with float constant
    const_field_float = Const(const=3.14)
    assert const_field_float.const == 3.14


# LLM-generated content at query #6
#--------------------------

```python
def test_Array_validate():
    # Test null value with allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None

    # Test null value with allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"

    # Test empty list with min_items=1
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"

    # Test exact_items validation
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "exact_items"

    # Test min_items validation
    field = Array(min_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.messages()[0].code == "min_items"

    # Test max_items validation
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test unique_items validation
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert exc_info.value.messages()[0].code == "unique_items"

    # Test with single item validator
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test with single item validator - type mismatch
    field = Array(items=Integer())
    with pytest.raises(ValidationError):
        field.validate([1, "invalid", 3])

    # Test with tuple of validators
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]

    # Test with tuple of validators and additional_items=True
    field = Array(items=[Integer(), String()], additional_items=True)
    result = field.validate([1, "hello", "extra"])
    assert result == [1, "hello", "extra"]

    # Test with tuple of validators and additional_items=Field
    field = Array(items=[Integer()], additional_items=String())
    result = field.validate([1, "extra"])
    assert result == [1, "extra"]

    # Test with tuple of validators and additional_items=False
    field = Array(items=[Integer()], additional_items=False)
    with pytest.raises(ValidationError):
        field.validate([1, 2])

    # Test empty list with no constraints
    field = Array()
    result = field.validate([])
    assert result == []

    # Test valid list
    field = Array(min_items=1, max_items=5)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test unique items with valid unique values
    field = Array(unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test nested validation with item field
    field = Array(items=String(min_length=2))
    result = field.validate(["ab", "cd"])
    assert result == ["ab", "cd"]

    # Test nested validation with item field - validation error
    field = Array(items=String(min_length=2))
    with pytest.raises(ValidationError):
        field.validate(["a", "cd"])

    # Test with no items validator
    field = Array()
    result = field.validate([1, "mixed", None, {"key": "value"}])
    assert result == [1, "mixed", None, {"key": "value"}]


# LLM-generated content at query #7
#--------------------------

```python
def test_Const_validate():
    # Test matching const value
    field = Const(const="test_value")
    assert field.validate("test_value") == "test_value"
    
    # Test matching const value with integer
    field = Const(const=42)
    assert field.validate(42) == 42
    
    # Test matching const value with None
    field = Const(const=None)
    assert field.validate(None) is None
    
    # Test matching const value with boolean
    field = Const(const=True)
    assert field.validate(True) is True
    
    # Test matching const value with dict
    field = Const(const={"key": "value"})
    assert field.validate({"key": "value"}) == {"key": "value"}
    
    # Test non-matching value raises validation error
    field = Const(const="expected")
    with pytest.raises(ValidationError) as exc_info:
        field.validate("unexpected")
    assert exc_info.value.messages()[0].code == "const"
    
    # Test non-matching integer raises validation error
    field = Const(const=10)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(20)
    assert exc_info.value.messages()[0].code == "const"
    
    # Test non-matching None raises validation error with only_null code
    field = Const(const=None)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not_none")
    assert exc_info.value.messages()[0].code == "only_null"
    
    # Test matching None when const is None
    field = Const(const=None)
    assert field.validate(None) is None
    
    # Test non-matching None when const is not None raises only_null error
    field = Const(const="value")
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "only_null"
    
    # Test const with empty string
    field = Const(const="")
    assert field.validate("") == ""
    
    # Test const with zero
    field = Const(const=0)
    assert field.validate(0) == 0
    
    # Test const with list
    field = Const(const=[1, 2, 3])
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    # Test non-matching list raises validation error
    field = Const(const=[1, 2, 3])
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.messages()[0].code == "const"


# LLM-generated content at query #8
#--------------------------

```python
def test_Array_validate():
    # Test with None and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None

    # Test with None and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"

    # Test with dict type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate({"key": "value"})
    assert exc_info.value.messages()[0].code == "type"

    # Test exact_items validation
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "exact_items"

    # Test exact_items validation - correct length
    field = Array(exact_items=2)
    assert field.validate([1, 2]) == [1, 2]

    # Test min_items validation
    field = Array(min_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "min_items"

    # Test min_items=1 validation (empty error)
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"

    # Test max_items validation
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test valid array with min_items
    field = Array(min_items=1)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

    # Test valid array with max_items
    field = Array(max_items=3)
    assert field.validate([1, 2]) == [1, 2]

    # Test with single item validator
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]

    # Test with single item validator - invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "invalid", 3])
    assert exc_info.value.messages()[0].index == [1]

    # Test with multiple item validators (tuple validation)
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "hello"]) == [1, "hello"]

    # Test with multiple item validators - invalid second item
    field = Array(items=[Integer(), String()])
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 123])
    assert exc_info.value.messages()[0].index == [1]

    # Test with additional_items=True
    field = Array(items=[Integer()], additional_items=True)
    assert field.validate([1, "extra", "more"]) == [1, "extra", "more"]

    # Test with additional_items=False
    field = Array(items=[Integer()], additional_items=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "extra"])
    assert exc_info.value.messages()[0].index == [1]

    # Test with additional_items as Field validator
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "extra", "more"]) == [1, "extra", "more"]

    # Test with additional_items as Field validator - invalid additional item
    field = Array(items=[Integer()], additional_items=String())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 123])
    assert exc_info.value.messages()[0].index == [1]

    # Test unique_items validation
    field = Array(items=Integer(), unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

    # Test unique_items validation - duplicate items
    field = Array(items=Integer(), unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert exc_info.value.messages()[0].code == "unique_items"

    # Test empty array
    field = Array()
    assert field.validate([]) == []

    # Test array with nested objects
    field = Array(items=Object(properties={"name": String()}))
    result = field.validate([{"name": "John"}, {"name": "Jane"}])
    assert result == [{"name": "John"}, {"name": "Jane"}]

    # Test array with no items validator
    field = Array(items=None)
    assert field.validate([1, "string", {"key": "value"}]) == [1, "string", {"key": "value"}]

    # Test unique_items with strings
    field = Array(items=String(), unique_items=True)
    assert field.validate(["a", "b", "c"]) == ["a", "b", "c"]

    # Test unique_items with strings - duplicate
    field = Array(items=String(), unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(["a", "b", "a"])
    assert exc_info.value.messages()[0].code == "unique_items"


# LLM-generated content at query #9
#--------------------------

```python
def test_Object_validate():
    # Test with None and allow_null=True
    obj = Object(allow_null=True)
    assert obj.validate(None) is None
    
    # Test with None and allow_null=False
    obj = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with non-dict type
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate("not a dict")
    assert exc_info.value.code == "type"
    
    # Test with non-string keys
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value._messages)
    
    # Test with simple properties
    obj = Object(properties={"name": String(), "age": Integer()})
    result = obj.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test with required fields missing
    obj = Object(properties={"name": String()}, required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({})
    assert any(msg.code == "required" for msg in exc_info.value._messages)
    
    # Test with default values
    obj = Object(properties={"name": String(default="Unknown")})
    result = obj.validate({})
    assert result == {"name": "Unknown"}
    
    # Test with min_properties
    obj = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"key": "value"})
    assert exc_info.value.code == "min_properties"
    
    # Test with min_properties=1 (empty check)
    obj = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({})
    assert exc_info.value.code == "empty"
    
    # Test with max_properties
    obj = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"key1": "value1", "key2": "value2"})
    assert exc_info.value.code == "max_properties"
    
    # Test with additional_properties=True
    obj = Object(additional_properties=True)
    result = obj.validate({"extra": "field"})
    assert result == {"extra": "field"}
    
    # Test with additional_properties=False
    obj = Object(additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"extra": "field"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value._messages)
    
    # Test with additional_properties as Field
    obj = Object(additional_properties=String())
    result = obj.validate({"extra": "field"})
    assert result == {"extra": "field"}
    
    # Test with pattern_properties
    obj = Object(pattern_properties={"^S_": String()})
    result = obj.validate({"S_name": "value"})
    assert result == {"S_name": "value"}
    
    # Test with property_names validation
    obj = Object(property_names=String(pattern="^[a-z]+$"))
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"Invalid": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value._messages)
    
    # Test with nested object validation error
    obj = Object(properties={"age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"age": "not an integer"})
    assert exc_info.value._messages
    
    # Test with Mapping type
    from collections import OrderedDict
    obj = Object()
    result = obj.validate(OrderedDict([("key", "value")]))
    assert result == {"key": "value"}


# LLM-generated content at query #10
#--------------------------

```python
def test_Array_validate():
    # Test with None and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code() == "null"
    
    # Test with non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.code() == "type"
    
    # Test with exact_items - correct length
    field = Array(exact_items=2)
    assert field.validate([1, 2]) == [1, 2]
    
    # Test with exact_items - incorrect length
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.code() == "exact_items"
    
    # Test with min_items - below minimum
    field = Array(min_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.code() == "min_items"
    
    # Test with min_items=1 - empty list
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.code() == "empty"
    
    # Test with max_items - above maximum
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.code() == "max_items"
    
    # Test with items Field validator
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    # Test with items Field validator - invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError):
        field.validate([1, "invalid", 3])
    
    # Test with tuple items validators
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "test"]) == [1, "test"]
    
    # Test with tuple items validators - too many items without additional_items
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "test", 3])
    
    # Test with tuple items validators and additional_items Field
    field = Array(items=[Integer(), String()], additional_items=Integer())
    assert field.validate([1, "test", 3]) == [1, "test", 3]
    
    # Test with unique_items=True - duplicate items
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert exc_info.value.code() == "unique_items"
    
    # Test with unique_items=True - unique items
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    # Test empty array with min_items not set
    field = Array()
    assert field.validate([]) == []
    
    # Test with nested Array validators
    field = Array(items=Array(items=Integer()))
    assert field.validate([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    
    # Test with nested Array validators - invalid nested item
    field = Array(items=Array(items=Integer()))
    with pytest.raises(ValidationError):
        field.validate([[1, 2], [3, "invalid"]])
    
    # Test with additional_items=True and tuple items
    field = Array(items=[Integer()], additional_items=True)
    assert field.validate([1, "anything", 3]) == [1, "anything", 3]
    
    # Test with additional_items=False and tuple items - extra items
    field = Array(items=[Integer()], additional_items=False)
    with pytest.raises(ValidationError):
        field.validate([1, 2])
    
    # Test valid list returns validated list
    field = Array(items=String())
    result = field.validate(["a", "b", "c"])
    assert result == ["a", "b", "c"]
    
    # Test with min_items and max_items both set
    field = Array(min_items=2, max_items=4)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    # Test with no items validator (pass through)
    field = Array()
    assert field.validate([1, "mixed", None]) == [1, "mixed", None]


# LLM-generated content at query #11
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling
    field_allow_null = String(allow_null=True)
    assert field_allow_null.validate(None) is None
    
    field_no_null = String()
    with pytest.raises(ValidationError) as exc_info:
        field_no_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test blank string handling
    field_blank = String(allow_blank=True)
    assert field_blank.validate("") == ""
    
    field_no_blank = String(allow_blank=False)
    with pytest.raises(ValidationError) as exc_info:
        field_no_blank.validate("")
    assert exc_info.value.code == "blank"
    
    # Test whitespace trimming
    field_trim = String(trim_whitespace=True)
    assert field_trim.validate("  hello  ") == "hello"
    
    field_no_trim = String(trim_whitespace=False)
    assert field_no_trim.validate("  hello  ") == "  hello  "
    
    # Test max_length validation
    field_max = String(max_length=5)
    assert field_max.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_max.validate("toolong")
    assert exc_info.value.code == "max_length"
    
    # Test min_length validation
    field_min = String(min_length=3)
    assert field_min.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_min.validate("hi")
    assert exc_info.value.code == "min_length"
    
    # Test pattern validation
    field_pattern = String(pattern=r"^\d+$")
    assert field_pattern.validate("12345") == "12345"
    with pytest.raises(ValidationError) as exc_info:
        field_pattern.validate("abc")
    assert exc_info.value.code == "pattern"
    
    # Test pattern with compiled regex
    import re
    compiled_pattern = re.compile(r"^[a-z]+$")
    field_compiled = String(pattern=compiled_pattern)
    assert field_compiled.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_compiled.validate("Hello123")
    assert exc_info.value.code == "pattern"
    
    # Test type validation
    field_type = String()
    with pytest.raises(ValidationError) as exc_info:
        field_type.validate(123)
    assert exc_info.value.code == "type"
    
    # Test null character removal
    field_null_char = String()
    assert field_null_char.validate("hello\0world") == "helloworld"
    
    # Test coerce_types with allow_null and allow_blank
    field_coerce = String(allow_null=True, coerce_types=True)
    assert field_coerce.validate("") is None
    
    field_coerce_blank = String(allow_blank=True, coerce_types=True)
    assert field_coerce_blank.validate(None) == ""
    
    # Test no coerce_types
    field_no_coerce = String(allow_null=True, coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field_no_coerce.validate("")
    assert exc_info.value.code == "blank"
    
    # Test email format
    field_email = String(format="email")
    result = field_email.validate("test@example.com")
    assert result is not None
    
    # Test uuid format
    field_uuid = String(format="uuid")
    result = field_uuid.validate("550e8400-e29b-41d4-a716-446655440000")
    assert result is not None
    
    # Test format with native type
    from datetime import date
    field_date = String(format="date")
    native_date = date(2023, 1, 1)
    assert field_date.validate(native_date) == native_date


# LLM-generated content at query #12
#--------------------------

```python
def test_Field_get_default_value():
    # Test with no default value
    field = Field()
    assert field.get_default_value() is None
    
    # Test with a static default value
    field = Field(default="test_value")
    assert field.get_default_value() == "test_value"
    
    # Test with a numeric default value
    field = Field(default=42)
    assert field.get_default_value() == 42
    
    # Test with a callable default value
    def get_default():
        return "callable_value"
    
    field = Field(default=get_default)
    assert field.get_default_value() == "callable_value"
    
    # Test with a lambda default value
    field = Field(default=lambda: [1, 2, 3])
    assert field.get_default_value() == [1, 2, 3]
    
    # Test with allow_null=True and no explicit default
    field = Field(allow_null=True)
    assert field.get_default_value() is None
    
    # Test with allow_null=True and explicit default
    field = Field(allow_null=True, default="explicit")
    assert field.get_default_value() == "explicit"
    
    # Test with boolean default value
    field = Field(default=False)
    assert field.get_default_value() is False
    
    # Test with empty list default
    field = Field(default=[])
    assert field.get_default_value() == []
    
    # Test with empty dict default
    field = Field(default={})
    assert field.get_default_value() == {}
    
    # Test with callable that returns None
    field = Field(default=lambda: None)
    assert field.get_default_value() is None


# LLM-generated content at query #13
#--------------------------

```python
def test_Array_serialize():
    # Test serialize with None value
    array_field = Array(items=String())
    assert array_field.serialize(None) is None

    # Test serialize with list of items and items is a single Field
    array_field = Array(items=String())
    result = array_field.serialize(["hello", "world"])
    assert result == ["hello", "world"]

    # Test serialize with list of items and items is a list of Fields
    array_field = Array(items=[String(), Integer()])
    result = array_field.serialize(["hello", 42])
    assert result == ["hello", 42]

    # Test serialize with items=None
    array_field = Array(items=None)
    result = array_field.serialize([1, "test", 3.14])
    assert result == [1, "test", 3.14]

    # Test serialize with Integer items
    array_field = Array(items=Integer())
    result = array_field.serialize([1, 2, 3])
    assert result == [1, 2, 3]

    # Test serialize with Decimal items
    array_field = Array(items=Decimal())
    result = array_field.serialize([decimal.Decimal("1.5"), decimal.Decimal("2.5")])
    assert result == [1.5, 2.5]

    # Test serialize with nested Array
    inner_array = Array(items=Integer())
    outer_array = Array(items=inner_array)
    result = outer_array.serialize([[1, 2], [3, 4]])
    assert result == [[1, 2], [3, 4]]

    # Test serialize with Object items
    obj_field = Object(properties={"name": String(), "age": Integer()})
    array_field = Array(items=obj_field)
    result = array_field.serialize([{"name": "Alice", "age": 30}])
    assert result == [{"name": "Alice", "age": 30}]

    # Test serialize with Boolean items
    array_field = Array(items=Boolean())
    result = array_field.serialize([True, False, True])
    assert result == [True, False, True]

    # Test serialize with mixed types and items=None
    array_field = Array(items=None)
    result = array_field.serialize([1, "string", 3.14, True, None])
    assert result == [1, "string", 3.14, True, None]

    # Test serialize with empty list
    array_field = Array(items=String())
    result = array_field.serialize([])
    assert result == []

    # Test serialize with multiple Field types in list
    array_field = Array(items=[String(), Integer(), Boolean()])
    result = array_field.serialize(["test", 42, True])
    assert result == ["test", 42, True]

    # Test serialize with Decimal in list of items
    array_field = Array(items=[Decimal(), Integer()])
    result = array_field.serialize([decimal.Decimal("1.5"), 10])
    assert result == [1.5, 10]


# LLM-generated content at query #14
#--------------------------

```python
def test_Union():
    # Test basic initialization with multiple fields
    field1 = String()
    field2 = Integer()
    union = Union(any_of=[field1, field2])
    assert union.any_of == [field1, field2]
    assert union.allow_null is False

    # Test initialization with allow_null child
    field3 = String(allow_null=True)
    field4 = Integer()
    union = Union(any_of=[field3, field4])
    assert union.allow_null is True

    # Test initialization with multiple allow_null children
    field5 = String(allow_null=True)
    field6 = Integer(allow_null=True)
    union = Union(any_of=[field5, field6])
    assert union.allow_null is True

    # Test initialization with no allow_null children
    field7 = String(allow_null=False)
    field8 = Integer(allow_null=False)
    union = Union(any_of=[field7, field8])
    assert union.allow_null is False

    # Test initialization with kwargs passed to parent
    field9 = String()
    field10 = Integer()
    union = Union(any_of=[field9, field10], allow_null=True)
    assert union.allow_null is True
    assert union.any_of == [field9, field10]

    # Test initialization with single field
    field11 = String()
    union = Union(any_of=[field11])
    assert union.any_of == [field11]
    assert union.allow_null is False

    # Test initialization with many fields
    fields = [String(), Integer(), Float(), Boolean()]
    union = Union(any_of=fields)
    assert union.any_of == fields
    assert len(union.any_of) == 4

    # Test that allow_null is set to True if any child has allow_null
    field12 = String(allow_null=False)
    field13 = Integer(allow_null=False)
    field14 = Float(allow_null=True)
    union = Union(any_of=[field12, field13, field14])
    assert union.allow_null is True


# LLM-generated content at query #15
#--------------------------

```python
def test_Field_get_default_value():
    # Test with no default value
    field = Field()
    assert field.get_default_value() is None
    
    # Test with static default value
    field = Field(default="test_value")
    assert field.get_default_value() == "test_value"
    
    # Test with numeric default value
    field = Field(default=42)
    assert field.get_default_value() == 42
    
    # Test with callable default value
    def default_factory():
        return "generated_value"
    
    field = Field(default=default_factory)
    assert field.get_default_value() == "generated_value"
    
    # Test with lambda default value
    field = Field(default=lambda: [1, 2, 3])
    assert field.get_default_value() == [1, 2, 3]
    
    # Test with None as explicit default
    field = Field(default=None)
    assert field.get_default_value() is None
    
    # Test with allow_null=True (should set default to None)
    field = Field(allow_null=True)
    assert field.get_default_value() is None
    
    # Test with default=0 (falsy value)
    field = Field(default=0)
    assert field.get_default_value() == 0
    
    # Test with default=False (falsy value)
    field = Field(default=False)
    assert field.get_default_value() is False
    
    # Test with default=empty string (falsy value)
    field = Field(default="")
    assert field.get_default_value() == ""
    
    # Test callable that returns None
    field = Field(default=lambda: None)
    assert field.get_default_value() is None


# LLM-generated content at query #16
#--------------------------

```python
def test_Object():
    # Test basic initialization with no arguments
    obj = Object()
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

    # Test with properties
    properties = {"name": String(), "age": Integer()}
    obj = Object(properties=properties)
    assert len(obj.properties) == 2
    assert "name" in obj.properties
    assert "age" in obj.properties

    # Test with pattern_properties
    pattern_properties = {"^S_": String(), "^I_": Integer()}
    obj = Object(pattern_properties=pattern_properties)
    assert len(obj.pattern_properties) == 2
    assert "^S_" in obj.pattern_properties
    assert "^I_" in obj.pattern_properties

    # Test with additional_properties as boolean
    obj = Object(additional_properties=False)
    assert obj.additional_properties is False

    obj = Object(additional_properties=True)
    assert obj.additional_properties is True

    # Test with additional_properties as Field
    additional_field = String()
    obj = Object(additional_properties=additional_field)
    assert obj.additional_properties is additional_field

    # Test with property_names
    property_names_field = String(pattern="^[a-z]+$")
    obj = Object(property_names=property_names_field)
    assert obj.property_names is property_names_field

    # Test with min_properties and max_properties
    obj = Object(min_properties=1, max_properties=10)
    assert obj.min_properties == 1
    assert obj.max_properties == 10

    # Test with required fields
    obj = Object(required=["name", "email"])
    assert obj.required == ["name", "email"]

    # Test with required as tuple
    obj = Object(required=("name", "email"))
    assert obj.required == ["name", "email"]

    # Test with all parameters combined
    properties = {"name": String(), "age": Integer()}
    pattern_properties = {"^meta_": String()}
    required = ["name"]
    obj = Object(
        properties=properties,
        pattern_properties=pattern_properties,
        additional_properties=False,
        property_names=String(),
        min_properties=1,
        max_properties=20,
        required=required,
        title="User",
        description="A user object",
        allow_null=False,
        read_only=False,
    )
    assert obj.properties == properties
    assert obj.pattern_properties == pattern_properties
    assert obj.additional_properties is False
    assert obj.property_names is not None
    assert obj.min_properties == 1
    assert obj.max_properties == 20
    assert obj.required == required
    assert obj.title == "User"
    assert obj.description == "A user object"
    assert obj.allow_null is False
    assert obj.read_only is False

    # Test that properties dict is copied, not referenced
    original_props = {"name": String()}
    obj = Object(properties=original_props)
    original_props["age"] = Integer()
    assert len(obj.properties) == 1

    # Test with Field passed as properties (legacy behavior)
    field = String()
    obj = Object(field)
    assert obj.properties == {}
    assert obj.additional_properties is field

    # Test required as None defaults to empty list
    obj = Object(required=None)
    assert obj.required == []

    # Test error on invalid properties keys
    with pytest.raises(AssertionError):
        Object(properties={123: String()})

    # Test error on invalid properties values
    with pytest.raises(AssertionError):
        Object(properties={"name": "not a field"})

    # Test error on invalid pattern_properties keys
    with pytest.raises(AssertionError):
        Object(pattern_properties={123: String()})

    # Test error on invalid pattern_properties values
    with pytest.raises(AssertionError):
        Object(pattern_properties={"^pattern": "not a field"})

    # Test error on invalid additional_properties
    with pytest.raises(AssertionError):
        Object(additional_properties="invalid")

    # Test error on invalid min_properties
    with pytest.raises(AssertionError):
        Object(min_properties="not an int")

    # Test error on invalid max_properties
    with pytest.raises(AssertionError):
        Object(max_properties="not an int")

    # Test error on invalid required items
    with pytest.raises(AssertionError):
        Object(required=[123, "valid"])


# LLM-generated content at query #17
#--------------------------

```python
def test_Array():
    # Test basic construction with no arguments
    field = Array()
    assert field.items is None
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False

    # Test construction with single Field items
    item_field = String()
    field = Array(items=item_field)
    assert field.items is item_field
    assert field.additional_items is False

    # Test construction with list of Field items
    item_fields = [String(), Integer()]
    field = Array(items=item_fields)
    assert field.items == item_fields
    assert field.min_items == 2
    assert field.max_items == 2
    assert field.additional_items is False

    # Test construction with list of Field items and additional_items=True
    item_fields = [String(), Integer()]
    field = Array(items=item_fields, additional_items=True)
    assert field.items == item_fields
    assert field.min_items == 2
    assert field.max_items is None
    assert field.additional_items is True

    # Test construction with list of Field items and additional_items as Field
    additional_field = Boolean()
    field = Array(items=item_fields, additional_items=additional_field)
    assert field.items == item_fields
    assert field.additional_items is additional_field

    # Test construction with tuple of Field items (should be converted to list)
    item_fields_tuple = (String(), Integer())
    field = Array(items=item_fields_tuple)
    assert field.items == list(item_fields_tuple)

    # Test construction with min_items
    field = Array(items=String(), min_items=5)
    assert field.min_items == 5

    # Test construction with max_items
    field = Array(items=String(), max_items=10)
    assert field.max_items == 10

    # Test construction with exact_items
    field = Array(items=String(), exact_items=7)
    assert field.min_items == 7
    assert field.max_items == 7

    # Test construction with unique_items
    field = Array(items=String(), unique_items=True)
    assert field.unique_items is True

    # Test construction with all parameters
    item_field = Integer()
    field = Array(
        items=item_field,
        additional_items=False,
        min_items=1,
        max_items=5,
        unique_items=True,
        allow_null=True,
        allow_blank=True
    )
    assert field.items is item_field
    assert field.additional_items is False
    assert field.min_items == 1
    assert field.max_items == 5
    assert field.unique_items is True
    assert field.allow_null is True
    assert field.allow_blank is True

    # Test that exact_items overrides min_items and max_items
    field = Array(items=String(), min_items=2, max_items=10, exact_items=5)
    assert field.min_items == 5
    assert field.max_items == 5

    # Test list of items sets min_items and max_items automatically
    items = [String(), Integer(), Boolean()]
    field = Array(items=items, additional_items=False)
    assert field.min_items == 3
    assert field.max_items == 3

    # Test list of items with additional_items=True doesn't set max_items
    field = Array(items=items, additional_items=True)
    assert field.min_items == 3
    assert field.max_items is None

    # Test explicit min_items/max_items override list-based defaults
    field = Array(items=items, min_items=1, max_items=10, additional_items=False)
    assert field.min_items == 1
    assert field.max_items == 10


# LLM-generated content at query #18
#--------------------------

```python
def test_Object_validate():
    # Test with None and allow_null=True
    obj_field = Object(allow_null=True)
    assert obj_field.validate(None) is None

    # Test with None and allow_null=False
    obj_field = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/mapping type
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with non-string keys
    obj_field = Object()
    with pytest.raises(ValidationError):
        obj_field.validate({1: "value"})

    # Test basic valid dict
    obj_field = Object()
    result = obj_field.validate({})
    assert result == {}

    # Test with properties
    obj_field = Object(properties={"name": String(), "age": Integer()})
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with required properties missing
    obj_field = Object(properties={"name": String()}, required=["name"])
    with pytest.raises(ValidationError):
        obj_field.validate({})

    # Test with required properties present
    obj_field = Object(properties={"name": String()}, required=["name"])
    result = obj_field.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with default values
    obj_field = Object(properties={"name": String(default="Unknown")})
    result = obj_field.validate({})
    assert result == {"name": "Unknown"}

    # Test with pattern properties
    obj_field = Object(pattern_properties={"^S_": String()})
    result = obj_field.validate({"S_name": "John"})
    assert result == {"S_name": "John"}

    # Test with additional_properties=True
    obj_field = Object(additional_properties=True)
    result = obj_field.validate({"extra": "value"})
    assert result == {"extra": "value"}

    # Test with additional_properties=False
    obj_field = Object(additional_properties=False)
    with pytest.raises(ValidationError):
        obj_field.validate({"extra": "value"})

    # Test with additional_properties as Field
    obj_field = Object(additional_properties=String())
    result = obj_field.validate({"extra": "value"})
    assert result == {"extra": "value"}

    # Test min_properties
    obj_field = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1})
    assert exc_info.value.code == "min_properties"

    # Test min_properties=1 shows "empty" error
    obj_field = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert exc_info.value.code == "empty"

    # Test max_properties
    obj_field = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1, "b": 2})
    assert exc_info.value.code == "max_properties"

    # Test property_names validation
    obj_field = Object(property_names=String(pattern="^[a-z]+$"))
    with pytest.raises(ValidationError):
        obj_field.validate({"Invalid": "value"})

    # Test with nested properties and validation errors
    obj_field = Object(properties={"age": Integer(minimum=0)})
    with pytest.raises(ValidationError):
        obj_field.validate({"age": -5})

    # Test with valid nested properties
    obj_field = Object(properties={"age": Integer(minimum=0)})
    result = obj_field.validate({"age": 25})
    assert result == {"age": 25}

    # Test complex scenario with multiple constraints
    obj_field = Object(
        properties={"name": String(), "age": Integer(minimum=0)},
        required=["name"],
        min_properties=1,
        max_properties=3,
        additional_properties=False
    )
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test Mapping type (not just dict)
    from collections import OrderedDict
    obj_field = Object()
    mapping = OrderedDict([("key", "value")])
    result = obj_field.validate(mapping)
    assert result == {"key": "value"}


# LLM-generated content at query #19
#--------------------------

```python
def test_Array_validate():
    # Test null value with allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    # Test null value with allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"
    
    # Test exact_items validation
    field = Array(exact_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.messages()[0].code == "exact_items"
    
    # Test min_items validation
    field = Array(min_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "min_items"
    
    # Test min_items=1 shows "empty" error
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"
    
    # Test max_items validation
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"
    
    # Test with single Field item validator
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test with single Field item validator - invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError):
        field.validate([1, "not an int", 3])
    
    # Test with list of Field validators
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]
    
    # Test with list of Field validators and additional_items=True
    field = Array(items=[Integer(), String()], additional_items=True)
    result = field.validate([1, "hello", "extra"])
    assert result == [1, "hello", "extra"]
    
    # Test with list of Field validators and additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "hello", "extra"])
    assert exc_info.value.messages()[0].code == "additional_items"
    
    # Test with list of Field validators and additional_items as Field
    field = Array(items=[Integer(), String()], additional_items=Integer())
    result = field.validate([1, "hello", 42])
    assert result == [1, "hello", 42]
    
    # Test unique_items=True with duplicate items
    field = Array(items=Integer(), unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    messages = exc_info.value.messages()
    assert any(msg.code == "unique_items" for msg in messages)
    
    # Test unique_items=True with unique items
    field = Array(items=Integer(), unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test empty array with no min_items
    field = Array()
    result = field.validate([])
    assert result == []
    
    # Test empty array validator (no items specified)
    field = Array()
    result = field.validate([1, "hello", {"key": "value"}])
    assert result == [1, "hello", {"key": "value"}]
    
    # Test with multiple validation errors
    field = Array(items=Integer(), min_items=2)
    with pytest.raises(ValidationError):
        field.validate([1, "invalid"])
    
    # Test valid array with all constraints
    field = Array(
        items=Integer(),
        min_items=1,
        max_items=5,
        unique_items=True
    )
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #20
#--------------------------

```python
def test_Choice():
    # Test basic initialization with string choices
    field = Choice(choices=["red", "green", "blue"])
    assert len(field.choices) == 3
    assert field.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]
    assert field.allow_null is False
    assert field.coerce_types is True

    # Test initialization with tuple choices
    field = Choice(choices=[("r", "Red"), ("g", "Green"), ("b", "Blue")])
    assert len(field.choices) == 3
    assert field.choices == [("r", "Red"), ("g", "Green"), ("b", "Blue")]

    # Test initialization with mixed choices (strings and tuples)
    field = Choice(choices=["red", ("g", "Green"), "blue"])
    assert len(field.choices) == 3
    assert field.choices == [("red", "red"), ("g", "Green"), ("blue", "blue")]

    # Test initialization with empty choices
    field = Choice(choices=[])
    assert len(field.choices) == 0
    assert field.choices == []

    # Test initialization with None choices (defaults to empty list)
    field = Choice(choices=None)
    assert len(field.choices) == 0
    assert field.choices == []

    # Test with allow_null parameter
    field = Choice(choices=["a", "b"], allow_null=True)
    assert field.allow_null is True

    # Test with coerce_types parameter
    field = Choice(choices=["a", "b"], coerce_types=False)
    assert field.coerce_types is False

    # Test with title and description
    field = Choice(
        choices=["a", "b"],
        title="Select Option",
        description="Choose one option"
    )
    assert field.title == "Select Option"
    assert field.description == "Choose one option"

    # Test with default value
    field = Choice(choices=["a", "b"], default="a")
    assert field.default == "a"
    assert field.has_default() is True

    # Test with read_only parameter
    field = Choice(choices=["a", "b"], read_only=True)
    assert field.read_only is True

    # Test error messages are properly defined
    assert "null" in Choice.errors
    assert "required" in Choice.errors
    assert "choice" in Choice.errors

    # Test with list choices (should work like sequence)
    field = Choice(choices=["x", "y", "z"])
    assert len(field.choices) == 3

    # Test combination of parameters
    field = Choice(
        choices=[("1", "Option 1"), ("2", "Option 2")],
        allow_null=True,
        coerce_types=False,
        title="Test Field",
        default="1",
        read_only=False
    )
    assert field.choices == [("1", "Option 1"), ("2", "Option 2")]
    assert field.allow_null is True
    assert field.coerce_types is False
    assert field.title == "Test Field"
    assert field.default == "1"
    assert field.read_only is False


# LLM-generated content at query #21
#--------------------------

```python
def test_Boolean_validate():
    # Test with None and allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Boolean(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with True boolean
    field = Boolean()
    assert field.validate(True) is True
    
    # Test with False boolean
    field = Boolean()
    assert field.validate(False) is False
    
    # Test with string "true" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    
    # Test with string "false" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("false") is False
    
    # Test with string "True" (uppercase) and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("True") is True
    
    # Test with string "on" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("on") is True
    
    # Test with string "off" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("off") is False
    
    # Test with integer 1 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(1) is True
    
    # Test with integer 0 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(0) is False
    
    # Test with empty string and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("") is False
    
    # Test with empty string and allow_null=True and coerce_types=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test with "null" string and allow_null=True and coerce_types=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("null") is None
    
    # Test with "none" string and allow_null=True and coerce_types=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("none") is None
    
    # Test with invalid string and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("invalid")
    assert exc_info.value.code == "type"
    
    # Test with invalid type and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.code == "type"
    
    # Test with non-boolean type and coerce_types=False
    field = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("true")
    assert exc_info.value.code == "type"
    
    # Test with float and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(1.5)
    assert exc_info.value.code == "type"


# LLM-generated content at query #22
#--------------------------

```python
def test_Const():
    # Test basic Const initialization with a simple value
    const_field = Const(const=42)
    assert const_field.const == 42
    
    # Test Const with None value
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    
    # Test Const with string value
    const_field_str = Const(const="hello")
    assert const_field_str.const == "hello"
    
    # Test Const with dict value
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    # Test Const with list value
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    # Test Const with boolean value
    const_field_bool = Const(const=True)
    assert const_field_bool.const is True
    
    # Test that allow_null cannot be passed as kwarg
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)
    
    # Test Const with additional kwargs
    const_field_with_kwargs = Const(const="test", required=True)
    assert const_field_with_kwargs.const == "test"
    assert const_field_with_kwargs.required is True
    
    # Test Const with float value
    const_field_float = Const(const=3.14)
    assert const_field_float.const == 3.14
    
    # Test Const with empty string
    const_field_empty_str = Const(const="")
    assert const_field_empty_str.const == ""


# LLM-generated content at query #23
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling
    field_allow_null = String(allow_null=True)
    assert field_allow_null.validate(None) is None
    
    field_no_null = String()
    with pytest.raises(ValidationError) as exc_info:
        field_no_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test blank string handling
    field_allow_blank = String(allow_blank=True)
    assert field_allow_blank.validate("") == ""
    
    field_no_blank = String(allow_blank=False)
    with pytest.raises(ValidationError) as exc_info:
        field_no_blank.validate("")
    assert exc_info.value.code == "blank"
    
    # Test type validation
    field = String()
    with pytest.raises(ValidationError) as exc_info:
        field.validate(123)
    assert exc_info.value.code == "type"
    
    # Test whitespace trimming
    field_trim = String(trim_whitespace=True)
    assert field_trim.validate("  hello  ") == "hello"
    
    field_no_trim = String(trim_whitespace=False)
    assert field_no_trim.validate("  hello  ") == "  hello  "
    
    # Test max_length
    field_max = String(max_length=5)
    assert field_max.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_max.validate("hello world")
    assert exc_info.value.code == "max_length"
    
    # Test min_length
    field_min = String(min_length=3)
    assert field_min.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_min.validate("hi")
    assert exc_info.value.code == "min_length"
    
    # Test pattern validation
    field_pattern = String(pattern=r"^\d+$")
    assert field_pattern.validate("12345") == "12345"
    with pytest.raises(ValidationError) as exc_info:
        field_pattern.validate("abc")
    assert exc_info.value.code == "pattern"
    
    # Test pattern with compiled regex
    field_compiled_pattern = String(pattern=re.compile(r"^[a-z]+$"))
    assert field_compiled_pattern.validate("abc") == "abc"
    with pytest.raises(ValidationError) as exc_info:
        field_compiled_pattern.validate("ABC")
    assert exc_info.value.code == "pattern"
    
    # Test null character removal
    field = String()
    assert field.validate("hello\0world") == "helloworld"
    
    # Test coerce_types with allow_null
    field_coerce_null = String(allow_null=True, coerce_types=True)
    assert field_coerce_null.validate(None) is None
    
    # Test coerce_types with allow_blank
    field_coerce_blank = String(allow_blank=True, coerce_types=True)
    assert field_coerce_blank.validate(None) == ""
    
    # Test empty string to None coercion
    field_coerce_empty = String(allow_null=True, coerce_types=True)
    assert field_coerce_empty.validate("") is None
    
    # Test format validation (email)
    field_email = String(format="email")
    assert field_email.validate("test@example.com") is not None
    
    # Test native type for format
    field_uuid = String(format="uuid")
    import uuid
    test_uuid = uuid.uuid4()
    assert field_uuid.validate(test_uuid) == test_uuid
    
    # Test combined constraints
    field_combined = String(min_length=2, max_length=10, allow_blank=False)
    assert field_combined.validate("hello") == "hello"
    with pytest.raises(ValidationError):
        field_combined.validate("a")
    with pytest.raises(ValidationError):
        field_combined.validate("a" * 11)


# LLM-generated content at query #24
#--------------------------

```python
def test_Union_validate():
    # Test with None value when allow_null is True
    union_field = Union(any_of=[String(), Integer()], allow_null=True)
    assert union_field.validate(None) is None

    # Test with None value when allow_null is False
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test valid string value
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate("hello")
    assert result == "hello"

    # Test valid integer value
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate(42)
    assert result == 42

    # Test with child that has allow_null=True
    union_field = Union(any_of=[String(allow_null=True), Integer()])
    assert union_field.allow_null is True
    assert union_field.validate(None) is None

    # Test with multiple children having allow_null=True
    union_field = Union(
        any_of=[String(allow_null=True), Integer(allow_null=True)]
    )
    assert union_field.allow_null is True
    assert union_field.validate(None) is None

    # Test union with no matching type
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "union"

    # Test union with candidate errors (non-type errors)
    union_field = Union(
        any_of=[
            String(min_length=10),
            Integer(minimum=100),
        ]
    )
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("short")
    # Should raise the candidate error from the String field
    assert exc_info.value.messages()[0].code == "min_length"

    # Test union with single candidate error
    union_field = Union(
        any_of=[
            String(min_length=10),
            Integer(),
        ]
    )
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("short")
    # Should raise the error from String since it's the only candidate
    assert exc_info.value.messages()[0].code == "min_length"

    # Test union with float value matching Integer
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate(3.14)
    # Float should fail Integer validation (non-integer float)
    with pytest.raises(ValidationError):
        union_field.validate(3.14)

    # Test union with boolean (should fail type check)
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(True)
    assert exc_info.value.messages()[0].code == "union"

    # Test union with empty any_of list
    union_field = Union(any_of=[])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("anything")
    assert exc_info.value.messages()[0].code == "union"

    # Test union with complex nested field
    union_field = Union(
        any_of=[
            Object(properties={"name": String()}),
            Array(items=Integer()),
        ]
    )
    result = union_field.validate({"name": "test"})
    assert result == {"name": "test"}

    result = union_field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test union returns first matching validator
    union_field = Union(any_of=[Integer(), String()])
    result = union_field.validate(42)
    assert result == 42

    # Test union with coercion
    union_field = Union(any_of=[Integer(coerce_types=True), String()])
    result = union_field.validate("123")
    assert result == "123"


# LLM-generated content at query #25
#--------------------------

```python
def test_Const():
    # Test basic initialization with a value
    const_field = Const(const=42)
    assert const_field.const == 42
    assert const_field.allow_null is False
    
    # Test initialization with None
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    assert const_field_none.allow_null is False
    
    # Test initialization with string
    const_field_str = Const(const="test_value")
    assert const_field_str.const == "test_value"
    
    # Test initialization with list
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    # Test initialization with dict
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    # Test that allow_null cannot be passed as kwarg
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)
    
    # Test initialization with other kwargs (should be passed to parent)
    const_field_with_kwargs = Const(const=10, description="A constant field")
    assert const_field_with_kwargs.const == 10
    assert const_field_with_kwargs.description == "A constant field"
    
    # Test initialization with boolean value
    const_field_bool = Const(const=True)
    assert const_field_bool.const is True
    
    # Test initialization with float
    const_field_float = Const(const=3.14)
    assert const_field_float.const == 3.14


# LLM-generated content at query #26
#--------------------------

```python
def test_Const():
    # Test basic initialization with a value
    const_field = Const(const=42)
    assert const_field.const == 42
    assert const_field.allow_null is False

    # Test initialization with None
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    assert const_field_none.allow_null is False

    # Test initialization with string
    const_field_str = Const(const="test")
    assert const_field_str.const == "test"

    # Test initialization with list
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]

    # Test initialization with dict
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}

    # Test that allow_null cannot be passed as kwarg
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)

    # Test initialization with other Field kwargs
    const_field_with_kwargs = Const(const="test", description="A constant field")
    assert const_field_with_kwargs.const == "test"
    assert const_field_with_kwargs.description == "A constant field"

    # Test initialization with error_message
    const_field_error = Const(const=100, error_message="Custom error")
    assert const_field_error.const == 100
    assert const_field_error.error_message == "Custom error"

    # Test initialization with bool
    const_field_bool = Const(const=True)
    assert const_field_bool.const is True

    # Test initialization with float
    const_field_float = Const(const=3.14)
    assert const_field_float.const == 3.14

    # Test initialization with empty string
    const_field_empty_str = Const(const="")
    assert const_field_empty_str.const == ""

    # Test initialization with zero
    const_field_zero = Const(const=0)
    assert const_field_zero.const == 0


# LLM-generated content at query #27
#--------------------------

```python
def test_Choice_validate():
    # Test with valid choice
    field = Choice(choices=["red", "green", "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"

    # Test with tuple choices
    field = Choice(choices=[("1", "Option 1"), ("2", "Option 2")])
    assert field.validate("1") == "1"
    assert field.validate("2") == "2"

    # Test with invalid choice
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("yellow")
    assert exc_info.value.code == "choice"

    # Test with None when allow_null is False
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"

    # Test with None when allow_null is True
    field = Choice(choices=["red", "green", "blue"], allow_null=True)
    assert field.validate(None) is None

    # Test with empty string when allow_null is False
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with empty string when allow_null is True and coerce_types is True
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test with empty string when allow_null is True and coerce_types is False
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with empty choices
    field = Choice(choices=[])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("anything")
    assert exc_info.value.code == "choice"

    # Test with mixed tuple and string choices
    field = Choice(choices=["red", ("2", "Green"), "blue"])
    assert field.validate("red") == "red"
    assert field.validate("2") == "2"
    assert field.validate("blue") == "blue"

    # Test with numeric choice
    field = Choice(choices=["1", "2", "3"])
    assert field.validate("1") == "1"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("4")
    assert exc_info.value.code == "choice"


# LLM-generated content at query #28
#--------------------------

```python
def test_Choice_validate():
    # Test valid choice
    field = Choice(choices=["red", "green", "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"

    # Test invalid choice
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("yellow")
    assert exc_info.value.code == "choice"

    # Test with tuple choices
    field = Choice(choices=[("r", "Red"), ("g", "Green"), ("b", "Blue")])
    assert field.validate("r") == "r"
    assert field.validate("g") == "g"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("Red")
    assert exc_info.value.code == "choice"

    # Test null value with allow_null=False
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"

    # Test null value with allow_null=True
    field = Choice(choices=["red", "green", "blue"], allow_null=True)
    assert field.validate(None) is None

    # Test empty string with allow_null=False and coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test empty string with allow_null=True and coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test empty string with allow_null=False and coerce_types=False
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test empty string with allow_null=True and coerce_types=False
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=False)
    assert field.validate("") is None

    # Test with empty choices
    field = Choice(choices=[])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("any")
    assert exc_info.value.code == "choice"

    # Test numeric choices
    field = Choice(choices=["1", "2", "3"])
    assert field.validate("1") == "1"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("4")
    assert exc_info.value.code == "choice"


# LLM-generated content at query #29
#--------------------------

```python
def test_Object_validate():
    # Test basic valid object
    obj_field = Object(properties={"name": String(), "age": Integer()})
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with None and allow_null=True
    obj_field = Object(allow_null=True)
    result = obj_field.validate(None)
    assert result is None

    # Test with None and allow_null=False
    obj_field = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict type
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with non-string keys
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())

    # Test required properties
    obj_field = Object(properties={"name": String()}, required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test min_properties
    obj_field = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"key": "value"})
    assert exc_info.value.code == "min_properties"

    # Test min_properties=1 (empty error)
    obj_field = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert exc_info.value.code == "empty"

    # Test max_properties
    obj_field = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"key1": "value1", "key2": "value2"})
    assert exc_info.value.code == "max_properties"

    # Test with default values
    obj_field = Object(properties={"name": String(default="Unknown")})
    result = obj_field.validate({})
    assert result == {"name": "Unknown"}

    # Test additional_properties=True
    obj_field = Object(properties={"name": String()}, additional_properties=True)
    result = obj_field.validate({"name": "John", "extra": "data"})
    assert result == {"name": "John", "extra": "data"}

    # Test additional_properties=False
    obj_field = Object(properties={"name": String()}, additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"name": "John", "extra": "data"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test additional_properties with Field
    obj_field = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test pattern_properties
    obj_field = Object(pattern_properties={"^num_": Integer()})
    result = obj_field.validate({"num_1": 10, "num_2": 20})
    assert result == {"num_1": 10, "num_2": 20}

    # Test property_names validation
    name_field = String(pattern="^[a-z]+$")
    obj_field = Object(property_names=name_field)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"Invalid": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test nested object validation
    obj_field = Object(
        properties={
            "user": Object(properties={"name": String(), "age": Integer()})
        }
    )
    result = obj_field.validate({"user": {"name": "John", "age": 30}})
    assert result == {"user": {"name": "John", "age": 30}}

    # Test nested validation error
    obj_field = Object(
        properties={
            "user": Object(properties={"age": Integer()})
        }
    )
    with pytest.raises(ValidationError):
        obj_field.validate({"user": {"age": "not an int"}})

    # Test with Mapping type
    from collections import OrderedDict
    obj_field = Object(properties={"name": String()})
    result = obj_field.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}

    # Test child schema validation error propagation
    obj_field = Object(
        properties={"name": String(max_length=5)}
    )
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"name": "VeryLongName"})
    assert any(msg.code == "max_length" for msg in exc_info.value.messages())


# LLM-generated content at query #30
#--------------------------

```python
def test_Array_validate():
    # Test with None value and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None

    # Test with None value and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-list value
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"

    # Test exact_items validation
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "exact_items"

    # Test min_items validation
    field = Array(min_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "min_items"

    # Test min_items=1 (empty error)
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"

    # Test max_items validation
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test with single Field validator
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test with single Field validator - invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "not an int", 3])
    assert len(exc_info.value.messages()) == 1

    # Test with multiple Field validators
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "test"])
    assert result == [1, "test"]

    # Test with multiple Field validators and additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "test", 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test with multiple Field validators and additional_items as Field
    field = Array(items=[Integer(), String()], additional_items=Float())
    result = field.validate([1, "test", 3.5])
    assert result == [1, "test", 3.5]

    # Test unique_items=True with duplicate items
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert exc_info.value.messages()[0].code == "unique_items"

    # Test unique_items=True with unique items
    field = Array(unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test empty list
    field = Array()
    result = field.validate([])
    assert result == []

    # Test with nested validators
    field = Array(items=Object(properties={"key": String()}))
    result = field.validate([{"key": "value"}])
    assert result == [{"key": "value"}]

    # Test with nested validators - invalid nested item
    field = Array(items=Object(properties={"key": Integer()}))
    with pytest.raises(ValidationError) as exc_info:
        field.validate([{"key": "not an int"}])
    assert len(exc_info.value.messages()) == 1

    # Test with no items validator
    field = Array(items=None)
    result = field.validate([1, "test", 3.5, None])
    assert result == [1, "test", 3.5, None]

    # Test exact_items with correct count
    field = Array(exact_items=3)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test multiple validation errors
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "not int", "also not int"])
    assert len(exc_info.value.messages()) == 2


# LLM-generated content at query #31
#--------------------------

```python
def test_Boolean_validate():
    # Test with None and allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Boolean(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with boolean True
    field = Boolean()
    assert field.validate(True) is True
    
    # Test with boolean False
    field = Boolean()
    assert field.validate(False) is False
    
    # Test with string "true" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    
    # Test with string "false" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("false") is False
    
    # Test with string "True" (uppercase) and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("True") is True
    
    # Test with string "on" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("on") is True
    
    # Test with string "off" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("off") is False
    
    # Test with string "1" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("1") is True
    
    # Test with string "0" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("0") is False
    
    # Test with empty string and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("") is False
    
    # Test with integer 1 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(1) is True
    
    # Test with integer 0 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(0) is False
    
    # Test with string "null" and allow_null=True, coerce_types=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("null") is None
    
    # Test with string "none" and allow_null=True, coerce_types=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("none") is None
    
    # Test with empty string and allow_null=True, coerce_types=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test with invalid string and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("invalid")
    assert exc_info.value.code == "type"
    
    # Test with non-boolean type and coerce_types=False
    field = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("true")
    assert exc_info.value.code == "type"
    
    # Test with list type and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.code == "type"
    
    # Test with dict type and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate({})
    assert exc_info.value.code == "type"


# LLM-generated content at query #32
#--------------------------

```python
def test_Const():
    # Test basic initialization with a constant value
    const_field = Const(const=42)
    assert const_field.const == 42
    
    # Test initialization with None
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    
    # Test initialization with string
    const_field_str = Const(const="test_value")
    assert const_field_str.const == "test_value"
    
    # Test initialization with list
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    # Test initialization with dict
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)
    
    # Test initialization with other Field kwargs
    const_field_with_kwargs = Const(const=100, required=True)
    assert const_field_with_kwargs.const == 100
    assert const_field_with_kwargs.required is True
    
    # Test initialization with default value
    const_field_default = Const(const=50, default=50)
    assert const_field_default.const == 50
    assert const_field_default.default == 50
    
    # Test initialization with boolean
    const_field_bool = Const(const=True)
    assert const_field_bool.const is True
    
    # Test initialization with float
    const_field_float = Const(const=3.14)
    assert const_field_float.const == 3.14


# LLM-generated content at query #33
#--------------------------

```python
def test_Choice_validate():
    # Test with valid choice
    field = Choice(choices=["red", "green", "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"

    # Test with invalid choice
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("yellow")
    assert exc_info.value.code == "choice"

    # Test with tuple choices
    field = Choice(choices=[("1", "Option 1"), ("2", "Option 2")])
    assert field.validate("1") == "1"
    assert field.validate("2") == "2"

    # Test with invalid tuple choice
    field = Choice(choices=[("1", "Option 1"), ("2", "Option 2")])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("3")
    assert exc_info.value.code == "choice"

    # Test with None and allow_null=True
    field = Choice(choices=["red", "green"], allow_null=True)
    assert field.validate(None) is None

    # Test with None and allow_null=False
    field = Choice(choices=["red", "green"], allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"

    # Test with empty string and allow_null=True, coerce_types=True
    field = Choice(choices=["red", "green"], allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test with empty string and allow_null=False, coerce_types=True
    field = Choice(choices=["red", "green"], allow_null=False, coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with empty string and allow_null=True, coerce_types=False
    field = Choice(choices=["red", "green"], allow_null=True, coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with numeric choices
    field = Choice(choices=[1, 2, 3])
    assert field.validate(1) == 1
    assert field.validate(2) == 2

    # Test with invalid numeric choice
    field = Choice(choices=[1, 2, 3])
    with pytest.raises(ValidationError) as exc_info:
        field.validate(4)
    assert exc_info.value.code == "choice"

    # Test with empty choices list
    field = Choice(choices=[], allow_null=True)
    assert field.validate(None) is None

    # Test with empty string in choices
    field = Choice(choices=["", "red", "green"])
    assert field.validate("") == ""

    # Test with mixed tuple and string choices
    field = Choice(choices=["red", ("green", "Green Color"), "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"


# LLM-generated content at query #34
#--------------------------

```python
def test_Boolean_validate():
    # Test with None and allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Boolean(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with True boolean
    field = Boolean()
    assert field.validate(True) is True
    
    # Test with False boolean
    field = Boolean()
    assert field.validate(False) is False
    
    # Test string coercion with coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    assert field.validate("True") is True
    assert field.validate("false") is False
    assert field.validate("False") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    
    # Test empty string coercion
    field = Boolean(coerce_types=True)
    assert field.validate("") is False
    
    # Test null string coercion with allow_null=True
    field = Boolean(coerce_types=True, allow_null=True)
    assert field.validate("null") is None
    assert field.validate("none") is None
    
    # Test integer coercion
    field = Boolean(coerce_types=True)
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test invalid type with coerce_types=False
    field = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("true")
    assert exc_info.value.code == "type"
    
    # Test invalid value with coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("invalid")
    assert exc_info.value.code == "type"
    
    # Test invalid type (non-string, non-int) with coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.code == "type"
    
    # Test case insensitivity for string values
    field = Boolean(coerce_types=True)
    assert field.validate("TRUE") is True
    assert field.validate("FALSE") is False
    assert field.validate("ON") is True
    assert field.validate("OFF") is False


# LLM-generated content at query #35
#--------------------------

```python
def test_Object_validate():
    # Test with None and allow_null=True
    obj_field = Object(allow_null=True)
    assert obj_field.validate(None) is None

    # Test with None and allow_null=False
    obj_field = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict type
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with empty dict
    obj_field = Object()
    assert obj_field.validate({}) == {}

    # Test with simple properties
    obj_field = Object(properties={"name": String(), "age": Integer()})
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with required properties missing
    obj_field = Object(required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test with non-string keys
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())

    # Test with min_properties
    obj_field = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1})
    assert exc_info.value.code == "min_properties"

    # Test with min_properties=1
    obj_field = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert exc_info.value.code == "empty"

    # Test with max_properties
    obj_field = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1, "b": 2})
    assert exc_info.value.code == "max_properties"

    # Test with property_names validation
    obj_field = Object(property_names=String(pattern="^[a-z]+$"))
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"123": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test with additional_properties=True
    obj_field = Object(properties={"name": String()}, additional_properties=True)
    result = obj_field.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John", "extra": "value"}

    # Test with additional_properties=False
    obj_field = Object(properties={"name": String()}, additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"name": "John", "extra": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test with additional_properties as Field
    obj_field = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with pattern_properties
    obj_field = Object(pattern_properties={"^num_": Integer()})
    result = obj_field.validate({"num_1": 10, "num_2": 20})
    assert result == {"num_1": 10, "num_2": 20}

    # Test with default values for missing properties
    obj_field = Object(properties={"name": String(default="Unknown")})
    result = obj_field.validate({})
    assert result == {"name": "Unknown"}

    # Test with nested validation errors
    obj_field = Object(properties={"age": Integer(minimum=0)})
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"age": -5})
    assert any(msg.code == "minimum" for msg in exc_info.value.messages())

    # Test with Mapping type
    from collections import OrderedDict
    obj_field = Object()
    mapping = OrderedDict([("key", "value")])
    result = obj_field.validate(mapping)
    assert result == {"key": "value"}

    # Test with complex nested structure
    obj_field = Object(
        properties={
            "user": Object(
                properties={"name": String(), "age": Integer()},
                required=["name"]
            )
        }
    )
    result = obj_field.validate({"user": {"name": "John", "age": 30}})
    assert result == {"user": {"name": "John", "age": 30}}


# LLM-generated content at query #36
#--------------------------

```python
def test_Union_validate():
    # Test with null value and allow_null=True
    union_field = Union(any_of=[String(), Integer()], allow_null=True)
    assert union_field.validate(None) is None
    
    # Test with null value and allow_null=False
    union_field = Union(any_of=[String(), Integer()], allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test with valid string value
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate("hello") == "hello"
    
    # Test with valid integer value
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate(42) == 42
    
    # Test with invalid value (no matching type)
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate([])
    assert exc_info.value.messages()[0].code == "union"
    
    # Test with child field that has allow_null=True
    union_field = Union(any_of=[String(allow_null=True), Integer()])
    assert union_field.allow_null is True
    assert union_field.validate(None) is None
    
    # Test with multiple children having allow_null=True
    union_field = Union(any_of=[String(allow_null=True), Integer(allow_null=True)])
    assert union_field.allow_null is True
    assert union_field.validate(None) is None
    
    # Test validation error from child field (non-type error)
    union_field = Union(any_of=[Integer(minimum=10), String()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(5)
    # Should raise the child's error since it's not a type error
    assert exc_info.value.messages()[0].code == "minimum"
    
    # Test with valid value matching second child
    union_field = Union(any_of=[Integer(), String()])
    assert union_field.validate("test") == "test"
    
    # Test with float that can be converted to integer
    union_field = Union(any_of=[Integer(), String()])
    assert union_field.validate(3.0) == 3
    
    # Test with empty string and String() field
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate("") == ""
    
    # Test with boolean (should not match Integer if strict)
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(True)
    assert exc_info.value.messages()[0].code == "union"
    
    # Test with multiple validation errors, should use first candidate error
    union_field = Union(any_of=[Integer(minimum=100), Integer(minimum=50)])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(30)
    # Both fail with minimum error, should raise union error
    assert exc_info.value.messages()[0].code == "union"
    
    # Test with exact match in first field
    union_field = Union(any_of=[String(max_length=5), String()])
    assert union_field.validate("hi") == "hi"
    
    # Test with value that fails first field but passes second
    union_field = Union(any_of=[String(max_length=2), String()])
    assert union_field.validate("hello") == "hello"


# LLM-generated content at query #37
#--------------------------

```python
def test_Union_validate():
    # Test case 1: Valid value matching first child type
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate("hello")
    assert result == "hello"

    # Test case 2: Valid value matching second child type
    result = union_field.validate(42)
    assert result == 42

    # Test case 3: None value with allow_null=True
    union_field = Union(any_of=[String(), Integer()], allow_null=True)
    result = union_field.validate(None)
    assert result is None

    # Test case 4: None value with allow_null=False should raise
    union_field = Union(any_of=[String(), Integer()], allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test case 5: Value not matching any child type
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "union"

    # Test case 6: Child with allow_null=True sets parent allow_null=True
    union_field = Union(any_of=[String(allow_null=True), Integer()])
    assert union_field.allow_null is True
    result = union_field.validate(None)
    assert result is None

    # Test case 7: Multiple children with allow_null=True
    union_field = Union(any_of=[String(allow_null=True), Integer(allow_null=True)])
    assert union_field.allow_null is True
    result = union_field.validate(None)
    assert result is None

    # Test case 8: Float value with Integer and Float children
    union_field = Union(any_of=[Integer(), Float()])
    result = union_field.validate(3.14)
    assert result == 3.14

    # Test case 9: Error from child with constraint violation
    union_field = Union(any_of=[String(min_length=5), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("ab")
    assert exc_info.value.messages()[0].code == "min_length"

    # Test case 10: Multiple non-type errors, returns first candidate error
    union_field = Union(any_of=[Integer(minimum=10), String(min_length=5)])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(5)
    # Should raise an error from one of the children
    assert len(exc_info.value.messages()) > 0

    # Test case 11: Boolean value with String and Integer
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(True)
    assert exc_info.value.messages()[0].code == "union"

    # Test case 12: Empty string with String child
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate("")
    assert result == ""

    # Test case 13: Zero value with Integer child
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate(0)
    assert result == 0

    # Test case 14: Decimal value with Float child
    union_field = Union(any_of=[String(), Float()])
    result = union_field.validate(2.718)
    assert result == 2.718

    # Test case 15: Complex nested Union
    inner_union = Union(any_of=[String(), Integer()])
    outer_union = Union(any_of=[inner_union, Boolean()])
    result = outer_union.validate("test")
    assert result == "test"
    result = outer_union.validate(42)
    assert result == 42
    result = outer_union.validate(True)
    assert result is True


# LLM-generated content at query #38
#--------------------------

```python
def test_Choice_validate():
    # Test with valid choice
    field = Choice(choices=["red", "green", "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"

    # Test with tuple choices
    field = Choice(choices=[("1", "Option 1"), ("2", "Option 2")])
    assert field.validate("1") == "1"
    assert field.validate("2") == "2"

    # Test with invalid choice
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("yellow")
    assert exc_info.value.code == "choice"

    # Test with None when allow_null is False
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"

    # Test with None when allow_null is True
    field = Choice(choices=["red", "green", "blue"], allow_null=True)
    assert field.validate(None) is None

    # Test with empty string when allow_null is False and coerce_types is False
    field = Choice(choices=["red", "green", "blue"], coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with empty string when allow_null is True and coerce_types is True
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test with empty string when allow_null is False and coerce_types is True
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with mixed tuple and string choices
    field = Choice(choices=["red", ("2", "Green"), "blue"])
    assert field.validate("red") == "red"
    assert field.validate("2") == "2"
    assert field.validate("blue") == "blue"

    # Test with no choices
    field = Choice(choices=[])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("red")
    assert exc_info.value.code == "choice"

    # Test with numeric choice values
    field = Choice(choices=[("1", "One"), ("2", "Two")])
    assert field.validate("1") == "1"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("3")
    assert exc_info.value.code == "choice"


# LLM-generated content at query #39
#--------------------------

```python
def test_Const():
    # Test basic construction with a constant value
    const_field = Const(const=42)
    assert const_field.const == 42
    
    # Test construction with None as constant
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    
    # Test construction with string constant
    const_field_str = Const(const="hello")
    assert const_field_str.const == "hello"
    
    # Test construction with list constant
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    # Test construction with dict constant
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    # Test that allow_null kwarg is not allowed
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)
    
    # Test that allow_null=False also raises AssertionError
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=False)
    
    # Test construction with other kwargs
    const_field_with_kwargs = Const(const=99, description="A constant field")
    assert const_field_with_kwargs.const == 99
    assert const_field_with_kwargs.description == "A constant field"
    
    # Test that allow_null is not in the constructed object's kwargs
    const_field_check = Const(const=5)
    assert hasattr(const_field_check, 'const')
    assert const_field_check.const == 5


# LLM-generated content at query #40
#--------------------------

```python
def test_Choice_validate():
    # Test with valid choice
    field = Choice(choices=["red", "green", "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"

    # Test with tuple choices
    field = Choice(choices=[("r", "Red"), ("g", "Green"), ("b", "Blue")])
    assert field.validate("r") == "r"
    assert field.validate("g") == "g"
    assert field.validate("b") == "b"

    # Test with invalid choice
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("yellow")
    assert exc_info.value.code == "choice"

    # Test with None when allow_null=True
    field = Choice(choices=["red", "green", "blue"], allow_null=True)
    assert field.validate(None) is None

    # Test with None when allow_null=False
    field = Choice(choices=["red", "green", "blue"], allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"

    # Test with empty string and allow_null=True, coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test with empty string and allow_null=False, coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with empty string and allow_null=False, coerce_types=False
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with mixed tuple and string choices
    field = Choice(choices=["red", ("g", "Green"), "blue"])
    assert field.validate("red") == "red"
    assert field.validate("g") == "g"
    assert field.validate("blue") == "blue"

    # Test with numeric choices
    field = Choice(choices=[1, 2, 3])
    assert field.validate(1) == 1
    assert field.validate(2) == 2
    assert field.validate(3) == 3

    # Test with invalid numeric choice
    field = Choice(choices=[1, 2, 3])
    with pytest.raises(ValidationError) as exc_info:
        field.validate(4)
    assert exc_info.value.code == "choice"

    # Test empty choices list
    field = Choice(choices=[])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("anything")
    assert exc_info.value.code == "choice"

    # Test with None choices parameter
    field = Choice(choices=None)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("anything")
    assert exc_info.value.code == "choice"


# LLM-generated content at query #41
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling
    field_allow_null = String(allow_null=True)
    assert field_allow_null.validate(None) is None
    
    field_no_null = String()
    with raises(ValidationError) as exc_info:
        field_no_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test blank string handling
    field_allow_blank = String(allow_blank=True)
    assert field_allow_blank.validate("") == ""
    
    field_no_blank = String(allow_blank=False)
    with raises(ValidationError) as exc_info:
        field_no_blank.validate("")
    assert exc_info.value.code == "blank"
    
    # Test type validation
    field = String()
    with raises(ValidationError) as exc_info:
        field.validate(123)
    assert exc_info.value.code == "type"
    
    # Test whitespace trimming
    field_trim = String(trim_whitespace=True)
    assert field_trim.validate("  hello  ") == "hello"
    
    field_no_trim = String(trim_whitespace=False)
    assert field_no_trim.validate("  hello  ") == "  hello  "
    
    # Test min_length
    field_min = String(min_length=3)
    assert field_min.validate("hello") == "hello"
    with raises(ValidationError) as exc_info:
        field_min.validate("hi")
    assert exc_info.value.code == "min_length"
    
    # Test max_length
    field_max = String(max_length=5)
    assert field_max.validate("hello") == "hello"
    with raises(ValidationError) as exc_info:
        field_max.validate("hello world")
    assert exc_info.value.code == "max_length"
    
    # Test pattern validation
    field_pattern = String(pattern=r"^\d+$")
    assert field_pattern.validate("12345") == "12345"
    with raises(ValidationError) as exc_info:
        field_pattern.validate("abc")
    assert exc_info.value.code == "pattern"
    
    # Test null character removal
    field = String()
    assert field.validate("hel\0lo") == "hello"
    
    # Test allow_blank with coerce_types converts None to empty string
    field_blank_coerce = String(allow_blank=True, coerce_types=True)
    assert field_blank_coerce.validate(None) == ""
    
    # Test allow_null with coerce_types converts empty string to None
    field_null_coerce = String(allow_null=True, coerce_types=True, trim_whitespace=True)
    assert field_null_coerce.validate("") is None
    
    # Test format validation (email)
    field_email = String(format="email")
    result = field_email.validate("test@example.com")
    assert result is not None
    
    # Test that native format types pass through
    from datetime import date
    field_date = String(format="date")
    date_obj = date(2023, 1, 1)
    assert field_date.validate(date_obj) == date_obj


# LLM-generated content at query #42
#--------------------------

```python
def test_Union_validate():
    # Test with None value when allow_null is True
    union_field = Union(any_of=[String(), Integer()], allow_null=True)
    assert union_field.validate(None) is None

    # Test with None value when allow_null is False
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with valid string value
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate("hello") == "hello"

    # Test with valid integer value
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate(42) == 42

    # Test with value that doesn't match any type
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate([])
    assert exc_info.value.messages()[0].code == "union"

    # Test with child that has allow_null=True
    union_field = Union(any_of=[String(allow_null=True), Integer()])
    assert union_field.validate(None) is None
    assert union_field.allow_null is True

    # Test with multiple children having allow_null=True
    union_field = Union(any_of=[String(allow_null=True), Integer(allow_null=True)])
    assert union_field.validate(None) is None
    assert union_field.allow_null is True

    # Test returning error from single candidate child with non-type error
    union_field = Union(any_of=[String(max_length=2), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("toolong")
    assert exc_info.value.messages()[0].code == "max_length"

    # Test with float that matches Integer
    union_field = Union(any_of=[Float(), String()])
    assert union_field.validate(3.14) == 3.14

    # Test with boolean (should not match String or Integer)
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(True)
    assert exc_info.value.messages()[0].code == "union"

    # Test with dict (should not match String or Integer)
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate({"key": "value"})
    assert exc_info.value.messages()[0].code == "union"

    # Test with multiple type errors - should raise union error
    union_field = Union(any_of=[Integer(minimum=10), Integer(maximum=5)])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(7)
    assert exc_info.value.messages()[0].code == "union"

    # Test that first valid child is returned
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate("123")
    assert result == "123"
    assert isinstance(result, str)

    # Test with empty union (edge case)
    union_field = Union(any_of=[])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("any value")
    assert exc_info.value.messages()[0].code == "union"

    # Test with complex nested types
    union_field = Union(any_of=[
        Object(properties={"name": String()}),
        Array(items=String())
    ])
    assert union_field.validate({"name": "John"}) == {"name": "John"}
    assert union_field.validate(["a", "b"]) == ["a", "b"]

    # Test with candidate error that has index
    union_field = Union(any_of=[
        Array(items=String()),
        String()
    ])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate([123])
    assert exc_info.value.messages()[0].code == "union"


# LLM-generated content at query #43
#--------------------------

```python
def test_Choice_validate():
    # Test with valid choice
    field = Choice(choices=["red", "green", "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"

    # Test with tuple choices
    field = Choice(choices=[("r", "Red"), ("g", "Green"), ("b", "Blue")])
    assert field.validate("r") == "r"
    assert field.validate("g") == "g"
    assert field.validate("b") == "b"

    # Test with None when allow_null is False
    field = Choice(choices=["red", "green", "blue"], allow_null=False)
    try:
        field.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "null"

    # Test with None when allow_null is True
    field = Choice(choices=["red", "green", "blue"], allow_null=True)
    assert field.validate(None) is None

    # Test with invalid choice
    field = Choice(choices=["red", "green", "blue"])
    try:
        field.validate("yellow")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "choice"

    # Test with empty string and allow_null=True, coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test with empty string and allow_null=False, coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=True)
    try:
        field.validate("")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "required"

    # Test with empty string and coerce_types=False
    field = Choice(choices=["red", "green", "blue"], coerce_types=False)
    try:
        field.validate("")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "choice"

    # Test with empty choices list
    field = Choice(choices=[])
    try:
        field.validate("red")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "choice"

    # Test with numeric choice
    field = Choice(choices=[1, 2, 3])
    assert field.validate(1) == 1
    assert field.validate(2) == 2

    # Test with numeric choice invalid
    field = Choice(choices=[1, 2, 3])
    try:
        field.validate(4)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "choice"


# LLM-generated content at query #44
#--------------------------

```python
def test_Choice_validate():
    # Test valid choice
    field = Choice(choices=["red", "green", "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"

    # Test with tuple choices
    field = Choice(choices=[("r", "Red"), ("g", "Green"), ("b", "Blue")])
    assert field.validate("r") == "r"
    assert field.validate("g") == "g"
    assert field.validate("b") == "b"

    # Test invalid choice raises ValidationError
    field = Choice(choices=["red", "green", "blue"])
    try:
        field.validate("yellow")
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        assert error.code == "choice"
        assert error.text == "Not a valid choice."

    # Test null value with allow_null=False raises ValidationError
    field = Choice(choices=["red", "green"], allow_null=False)
    try:
        field.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        assert error.code == "null"
        assert error.text == "May not be null."

    # Test null value with allow_null=True returns None
    field = Choice(choices=["red", "green"], allow_null=True)
    assert field.validate(None) is None

    # Test empty string with allow_null=True and coerce_types=True returns None
    field = Choice(choices=["red", "green"], allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test empty string with allow_null=False raises required error
    field = Choice(choices=["red", "green"], allow_null=False, coerce_types=True)
    try:
        field.validate("")
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        assert error.code == "required"
        assert error.text == "This field is required."

    # Test empty string with allow_null=False and coerce_types=False raises choice error
    field = Choice(choices=["red", "green"], allow_null=False, coerce_types=False)
    try:
        field.validate("")
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        assert error.code == "choice"
        assert error.text == "Not a valid choice."

    # Test empty choices list
    field = Choice(choices=[])
    try:
        field.validate("anything")
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        assert error.code == "choice"

    # Test with None choices parameter
    field = Choice(choices=None)
    try:
        field.validate("anything")
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        assert error.code == "choice"

    # Test single choice
    field = Choice(choices=["only"])
    assert field.validate("only") == "only"
    try:
        field.validate("other")
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        assert error.code == "choice"

    # Test numeric choices
    field = Choice(choices=[1, 2, 3])
    assert field.validate(1) == 1
    assert field.validate(2) == 2
    try:
        field.validate(4)
        assert False, "Should raise ValidationError"
    except ValidationError as error:
        assert error.code == "choice"


# LLM-generated content at query #45
#--------------------------

```python
def test_Union():
    # Test basic Union initialization with multiple fields
    field1 = String()
    field2 = Integer()
    field3 = Boolean()
    
    union = Union(any_of=[field1, field2, field3])
    
    assert union.any_of == [field1, field2, field3]
    assert union.allow_null is False
    
    # Test Union with allow_null child field
    field_nullable = String(allow_null=True)
    field_non_nullable = Integer()
    
    union_with_null = Union(any_of=[field_nullable, field_non_nullable])
    
    assert union_with_null.allow_null is True
    assert union_with_null.any_of == [field_nullable, field_non_nullable]
    
    # Test Union with all nullable child fields
    nullable_field1 = String(allow_null=True)
    nullable_field2 = Integer(allow_null=True)
    
    union_all_null = Union(any_of=[nullable_field1, nullable_field2])
    
    assert union_all_null.allow_null is True
    
    # Test Union with single field
    single_field = String()
    union_single = Union(any_of=[single_field])
    
    assert union_single.any_of == [single_field]
    assert union_single.allow_null is False
    
    # Test Union with empty any_of list
    union_empty = Union(any_of=[])
    
    assert union_empty.any_of == []
    assert union_empty.allow_null is False
    
    # Test Union with kwargs passed to parent
    union_with_kwargs = Union(
        any_of=[String(), Integer()],
        allow_null=False,
        description="Test union field"
    )
    
    assert union_with_kwargs.any_of == union_with_kwargs.any_of
    assert union_with_kwargs.allow_null is False
    
    # Test Union inherits from Field
    assert isinstance(union, Field)
    
    # Test Union with mixed nullable and non-nullable fields
    mixed_union = Union(any_of=[
        String(),
        Integer(allow_null=True),
        Boolean()
    ])
    
    assert mixed_union.allow_null is True


####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Array_validate():
    # Test with None and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None

    # Test with None and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"

    # Test with dict type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate({})
    assert exc_info.value.messages()[0].code == "type"

    # Test empty list with min_items=1
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"

    # Test list too small
    field = Array(min_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.messages()[0].code == "min_items"

    # Test list too large
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test exact_items mismatch
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test exact_items match
    field = Array(exact_items=2)
    assert field.validate([1, 2]) == [1, 2]

    # Test with single item validator
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]

    # Test with single item validator - invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError):
        field.validate([1, "invalid", 3])

    # Test with tuple of validators
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "hello"]) == [1, "hello"]

    # Test with tuple of validators - too many items without additional_items
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "hello", 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test with tuple of validators - additional items allowed
    field = Array(items=[Integer(), String()], additional_items=True)
    assert field.validate([1, "hello", 3]) == [1, "hello", 3]

    # Test with tuple of validators - additional items with Field validator
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "hello"]) == [1, "hello"]

    # Test unique_items - valid
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

    # Test unique_items - duplicate items
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    messages = exc_info.value.messages()
    assert any(msg.code == "unique_items" for msg in messages)

    # Test unique_items with string duplicates
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(["a", "b", "a"])
    messages = exc_info.value.messages()
    assert any(msg.code == "unique_items" for msg in messages)

    # Test with no items validator
    field = Array()
    assert field.validate([1, "mixed", {"key": "value"}]) == [1, "mixed", {"key": "value"}]

    # Test with items validator and coercion
    field = Array(items=Integer())
    assert field.validate(["1", "2", "3"]) == [1, 2, 3]

    # Test error propagation from item validator
    field = Array(items=Integer(minimum=5))
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 10, 3])
    messages = exc_info.value.messages()
    assert any(msg.code == "minimum" for msg in messages)

    # Test complex nested structure
    field = Array(items=Object(properties={"id": Integer(), "name": String()}))
    result = field.validate([{"id": 1, "name": "test"}])
    assert result == [{"id": 1, "name": "test"}]

    # Test empty array with no constraints
    field = Array()
    assert field.validate([]) == []

    # Test array with None items and allow_null on field
    field = Array(items=Integer(allow_null=True))
    assert field.validate([1, None, 3]) == [1, None, 3]

    # Test min_items with exact count
    field = Array(min_items=2)
    assert field.validate([1, 2]) == [1, 2]

    # Test max_items with exact count
    field = Array(max_items=2)
    assert field.validate([1, 2]) == [1, 2]


# LLM-generated content at query #2
#--------------------------

```python
def test_Array_validate():
    # Test with None value and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None value and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test with non-list value
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"
    
    # Test with non-list value (dict)
    with pytest.raises(ValidationError) as exc_info:
        field.validate({"key": "value"})
    assert exc_info.value.messages()[0].code == "type"
    
    # Test exact_items validation - correct length
    field = Array(exact_items=3)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test exact_items validation - incorrect length
    field = Array(exact_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.messages()[0].code == "exact_items"
    
    # Test min_items validation - too few items
    field = Array(min_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "min_items"
    
    # Test min_items validation with min_items=1 (empty)
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"
    
    # Test max_items validation - too many items
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"
    
    # Test with single Field validator
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test with single Field validator - invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError):
        field.validate([1, "not an integer", 3])
    
    # Test with list of validators
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]
    
    # Test with list of validators and additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]
    
    # Test with list of validators and additional_items=Field
    field = Array(items=[Integer(), String()], additional_items=Integer())
    result = field.validate([1, "hello", 42])
    assert result == [1, "hello", 42]
    
    # Test unique_items=True with unique items
    field = Array(unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test unique_items=True with duplicate items
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert exc_info.value.messages()[0].code == "unique_items"
    
    # Test unique_items=True with duplicate strings
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(["a", "b", "a"])
    assert exc_info.value.messages()[0].code == "unique_items"
    
    # Test empty array with no constraints
    field = Array()
    result = field.validate([])
    assert result == []
    
    # Test with items validator and validation errors
    field = Array(items=Integer(minimum=5))
    with pytest.raises(ValidationError) as exc_info:
        field.validate([10, 3, 8])
    errors = exc_info.value.messages()
    assert len(errors) > 0
    
    # Test with list of validators - wrong type
    field = Array(items=[Integer(), String()])
    with pytest.raises(ValidationError):
        field.validate(["not an int", "hello"])
    
    # Test min_items and max_items together
    field = Array(min_items=2, max_items=4)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test min_items and max_items - below minimum
    field = Array(min_items=2, max_items=4)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "min_items"
    
    # Test min_items and max_items - above maximum
    field = Array(min_items=2, max_items=4)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3, 4, 5])
    assert exc_info.value.messages()[0].code == "max_items"
    
    # Test with no items validator
    field = Array(items=None)
    result = field.validate([1, "string", {"key": "value"}])
    assert result == [1, "string", {"key": "value"}]
    
    # Test additional_items=False with extra items
    field = Array(items=[Integer()], additional_items=False)
    result = field.validate([1])
    assert result == [1]
    
    # Test with coercion in nested validator
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #3
#--------------------------

```python
def test_Const():
    # Test basic initialization with various constant values
    const_field = Const(const=42)
    assert const_field.const == 42
    
    const_field_str = Const(const="hello")
    assert const_field_str.const == "hello"
    
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    const_field_bool = Const(const=True)
    assert const_field_bool.const is True
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)
    
    # Test that other valid Field kwargs work
    const_with_description = Const(const=100, description="A constant field")
    assert const_with_description.const == 100
    assert const_with_description.description == "A constant field"
    
    # Test that default value can be set
    const_with_default = Const(const=50, default=50)
    assert const_with_default.const == 50
    assert const_with_default.default == 50
    
    # Test with float constant
    const_float = Const(const=3.14)
    assert const_float.const == 3.14
    
    # Test with empty string constant
    const_empty_str = Const(const="")
    assert const_empty_str.const == ""
    
    # Test with zero constant
    const_zero = Const(const=0)
    assert const_zero.const == 0
    
    # Test with False constant
    const_false = Const(const=False)
    assert const_false.const is False


# LLM-generated content at query #4
#--------------------------

```python
def test_Object_validate():
    # Test with None and allow_null=True
    obj = Object(allow_null=True)
    assert obj.validate(None) is None

    # Test with None and allow_null=False
    obj = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-dict type
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate("not a dict")
    assert exc_info.value.messages()[0].code == "type"

    # Test with empty dict
    obj = Object()
    assert obj.validate({}) == {}

    # Test with simple properties
    obj = Object(properties={"name": String()})
    result = obj.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with required fields
    obj = Object(properties={"name": String()}, required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({})
    error_codes = [msg.code for msg in exc_info.value.messages()]
    assert "required" in error_codes

    # Test with invalid key type
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({123: "value"})
    assert exc_info.value.messages()[0].code == "invalid_key"

    # Test with default values
    obj = Object(properties={"name": String(default="Unknown")})
    result = obj.validate({})
    assert result == {"name": "Unknown"}

    # Test with property_names validation
    obj = Object(property_names=String(pattern="^[a-z]+$"))
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"Name": "value"})
    assert exc_info.value.messages()[0].code == "invalid_property"

    # Test min_properties
    obj = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"a": 1})
    assert exc_info.value.messages()[0].code == "min_properties"

    # Test min_properties=1 (empty check)
    obj = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({})
    assert exc_info.value.messages()[0].code == "empty"

    # Test max_properties
    obj = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"a": 1, "b": 2})
    assert exc_info.value.messages()[0].code == "max_properties"

    # Test additional_properties=True
    obj = Object(additional_properties=True)
    result = obj.validate({"extra": "value"})
    assert result == {"extra": "value"}

    # Test additional_properties=False
    obj = Object(additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"extra": "value"})
    assert exc_info.value.messages()[0].code == "invalid_property"

    # Test additional_properties with Field
    obj = Object(additional_properties=Integer())
    result = obj.validate({"count": 42})
    assert result == {"count": 42}

    # Test pattern_properties
    obj = Object(pattern_properties={"^num": Integer()})
    result = obj.validate({"number": 123})
    assert result == {"number": 123}

    # Test nested validation error
    obj = Object(properties={"name": String(min_length=5)})
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"name": "Jo"})
    assert exc_info.value.messages()[0].code == "min_length"

    # Test complex scenario with multiple validations
    obj = Object(
        properties={"id": Integer(), "name": String()},
        required=["id"],
        additional_properties=False,
    )
    result = obj.validate({"id": 1, "name": "Test"})
    assert result == {"id": 1, "name": "Test"}

    # Test with Mapping type
    from collections import OrderedDict
    obj = Object()
    mapping = OrderedDict([("key", "value")])
    result = obj.validate(mapping)
    assert result == {"key": "value"}


# LLM-generated content at query #5
#--------------------------

```python
def test_Boolean_validate():
    # Test with None and allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Boolean(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with True
    field = Boolean()
    assert field.validate(True) is True
    
    # Test with False
    field = Boolean()
    assert field.validate(False) is False
    
    # Test with string "true" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    
    # Test with string "false" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("false") is False
    
    # Test with string "on" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("on") is True
    
    # Test with string "off" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("off") is False
    
    # Test with string "1" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("1") is True
    
    # Test with string "0" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("0") is False
    
    # Test with empty string and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("") is False
    
    # Test with integer 1 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(1) is True
    
    # Test with integer 0 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(0) is False
    
    # Test with uppercase string "TRUE" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("TRUE") is True
    
    # Test with coerce_null_values and allow_null=True
    field = Boolean(coerce_types=True, allow_null=True)
    assert field.validate("null") is None
    assert field.validate("none") is None
    assert field.validate("") is None
    
    # Test with coerce_null_values and allow_null=False
    field = Boolean(coerce_types=True, allow_null=False)
    assert field.validate("") is False
    
    # Test with invalid string and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("invalid")
    assert exc_info.value.code == "type"
    
    # Test with non-boolean type and coerce_types=False
    field = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("true")
    assert exc_info.value.code == "type"
    
    # Test with list and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.code == "type"
    
    # Test with dict and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate({})
    assert exc_info.value.code == "type"


# LLM-generated content at query #6
#--------------------------

```python
def test_Choice():
    # Test basic instantiation with string choices
    field = Choice(choices=["red", "green", "blue"])
    assert len(field.choices) == 3
    assert field.choices[0] == ("red", "red")
    assert field.choices[1] == ("green", "green")
    assert field.choices[2] == ("blue", "blue")
    
    # Test instantiation with tuple choices
    field = Choice(choices=[("r", "Red"), ("g", "Green"), ("b", "Blue")])
    assert len(field.choices) == 3
    assert field.choices[0] == ("r", "Red")
    assert field.choices[1] == ("g", "Green")
    assert field.choices[2] == ("b", "Blue")
    
    # Test mixed choices (strings and tuples)
    field = Choice(choices=["red", ("g", "Green")])
    assert len(field.choices) == 2
    assert field.choices[0] == ("red", "red")
    assert field.choices[1] == ("g", "Green")
    
    # Test empty choices
    field = Choice(choices=[])
    assert field.choices == []
    
    # Test None choices
    field = Choice(choices=None)
    assert field.choices == []
    
    # Test with title and description
    field = Choice(
        choices=["a", "b"],
        title="Select Option",
        description="Choose one option"
    )
    assert field.title == "Select Option"
    assert field.description == "Choose one option"
    
    # Test with default value
    field = Choice(choices=["a", "b"], default="a")
    assert field.default == "a"
    assert field.has_default() is True
    
    # Test with allow_null
    field = Choice(choices=["a", "b"], allow_null=True)
    assert field.allow_null is True
    
    # Test with read_only
    field = Choice(choices=["a", "b"], read_only=True)
    assert field.read_only is True
    
    # Test coerce_types parameter
    field = Choice(choices=["a", "b"], coerce_types=False)
    assert field.coerce_types is False
    
    # Test coerce_types default value
    field = Choice(choices=["a", "b"])
    assert field.coerce_types is True
    
    # Test with list choices instead of tuple
    field = Choice(choices=[["x", "X"], ["y", "Y"]])
    assert field.choices[0] == ("x", "X")
    assert field.choices[1] == ("y", "Y")
    
    # Test allow_null with default None
    field = Choice(choices=["a", "b"], allow_null=True)
    assert field.default is None
    assert field.has_default() is True


# LLM-generated content at query #7
#--------------------------

```python
def test_Choice_validate():
    # Test valid choice
    field = Choice(choices=["red", "green", "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"

    # Test invalid choice
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("yellow")
    assert exc_info.value.code == "choice"

    # Test with tuple choices
    field = Choice(choices=[("r", "Red"), ("g", "Green"), ("b", "Blue")])
    assert field.validate("r") == "r"
    assert field.validate("g") == "g"
    assert field.validate("b") == "b"

    # Test null value with allow_null=False
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"

    # Test null value with allow_null=True
    field = Choice(choices=["red", "green", "blue"], allow_null=True)
    assert field.validate(None) is None

    # Test empty string with allow_null=False and coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test empty string with allow_null=True and coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test empty string with allow_null=False and coerce_types=False
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test empty string with allow_null=True and coerce_types=False
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=False)
    assert field.validate("") is None

    # Test with mixed tuple and string choices
    field = Choice(choices=[("1", "Option 1"), "two", ("3", "Option 3")])
    assert field.validate("1") == "1"
    assert field.validate("two") == "two"
    assert field.validate("3") == "3"

    # Test empty choices
    field = Choice(choices=[])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("anything")
    assert exc_info.value.code == "choice"

    # Test numeric choice values
    field = Choice(choices=["1", "2", "3"])
    assert field.validate("1") == "1"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("4")
    assert exc_info.value.code == "choice"


# LLM-generated content at query #8
#--------------------------

```python
def test_Choice_validate():
    # Test with valid choice
    field = Choice(choices=["red", "green", "blue"])
    assert field.validate("red") == "red"
    assert field.validate("green") == "green"
    assert field.validate("blue") == "blue"

    # Test with tuple choices (value, display)
    field = Choice(choices=[("1", "Option 1"), ("2", "Option 2")])
    assert field.validate("1") == "1"
    assert field.validate("2") == "2"

    # Test with invalid choice
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("yellow")
    assert exc_info.value.code == "choice"

    # Test with None when allow_null=False
    field = Choice(choices=["red", "green", "blue"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"

    # Test with None when allow_null=True
    field = Choice(choices=["red", "green", "blue"], allow_null=True)
    assert field.validate(None) is None

    # Test with empty string when allow_null=False and coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test with empty string when allow_null=False and coerce_types=False
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with empty string when allow_null=False and coerce_types=True
    field = Choice(choices=["red", "green", "blue"], allow_null=False, coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "required"

    # Test with empty choices
    field = Choice(choices=[])
    with pytest.raises(ValidationError) as exc_info:
        field.validate("anything")
    assert exc_info.value.code == "choice"

    # Test with numeric choice values
    field = Choice(choices=[(1, "One"), (2, "Two"), (3, "Three")])
    assert field.validate(1) == 1
    assert field.validate(2) == 2

    # Test with mixed string and tuple choices
    field = Choice(choices=["simple", ("complex", "Complex Option")])
    assert field.validate("simple") == "simple"
    assert field.validate("complex") == "complex"


# LLM-generated content at query #9
#--------------------------

```python
def test_Number_validate():
    # Test basic number validation
    field = Number()
    assert field.validate(42) == 42
    assert field.validate(3.14) == 3.14
    assert field.validate("42") == 42
    assert field.validate("3.14") == 3.14
    
    # Test null handling
    field_nullable = Number(allow_null=True)
    assert field_nullable.validate(None) is None
    
    field_not_nullable = Number(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field_not_nullable.validate(None)
    assert exc_info.value.code == "null"
    
    # Test empty string with allow_null and coerce_types
    field_coerce = Number(allow_null=True, coerce_types=True)
    assert field_coerce.validate("") is None
    
    # Test boolean rejection
    field = Number()
    with pytest.raises(ValidationError) as exc_info:
        field.validate(True)
    assert exc_info.value.code == "type"
    
    # Test integer field with non-integer float
    field_int = Number(numeric_type=int)
    with pytest.raises(ValidationError) as exc_info:
        field_int.validate(3.14)
    assert exc_info.value.code == "integer"
    
    # Test coerce_types=False
    field_no_coerce = Number(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field_no_coerce.validate("42")
    assert exc_info.value.code == "type"
    
    # Test minimum constraint
    field_min = Number(minimum=10)
    assert field_min.validate(10) == 10
    assert field_min.validate(20) == 20
    with pytest.raises(ValidationError) as exc_info:
        field_min.validate(5)
    assert exc_info.value.code == "minimum"
    
    # Test maximum constraint
    field_max = Number(maximum=100)
    assert field_max.validate(100) == 100
    assert field_max.validate(50) == 50
    with pytest.raises(ValidationError) as exc_info:
        field_max.validate(150)
    assert exc_info.value.code == "maximum"
    
    # Test exclusive_minimum
    field_exc_min = Number(exclusive_minimum=10)
    assert field_exc_min.validate(10.1) == 10.1
    with pytest.raises(ValidationError) as exc_info:
        field_exc_min.validate(10)
    assert exc_info.value.code == "exclusive_minimum"
    with pytest.raises(ValidationError) as exc_info:
        field_exc_min.validate(5)
    assert exc_info.value.code == "exclusive_minimum"
    
    # Test exclusive_maximum
    field_exc_max = Number(exclusive_maximum=100)
    assert field_exc_max.validate(99.9) == 99.9
    with pytest.raises(ValidationError) as exc_info:
        field_exc_max.validate(100)
    assert exc_info.value.code == "exclusive_maximum"
    with pytest.raises(ValidationError) as exc_info:
        field_exc_max.validate(150)
    assert exc_info.value.code == "exclusive_maximum"
    
    # Test finite constraint (inf, -inf, nan)
    field = Number()
    with pytest.raises(ValidationError) as exc_info:
        field.validate(float('inf'))
    assert exc_info.value.code == "finite"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate(float('-inf'))
    assert exc_info.value.code == "finite"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate(float('nan'))
    assert exc_info.value.code == "finite"
    
    # Test multiple_of with integer
    field_multiple = Number(multiple_of=5)
    assert field_multiple.validate(10) == 10
    assert field_multiple.validate(15) == 15
    with pytest.raises(ValidationError) as exc_info:
        field_multiple.validate(7)
    assert exc_info.value.code == "multiple_of"
    
    # Test multiple_of with decimal
    field_multiple_dec = Number(multiple_of=0.5)
    assert field_multiple_dec.validate(1.0) == 1.0
    assert field_multiple_dec.validate(1.5) == 1.5
    with pytest.raises(ValidationError) as exc_info:
        field_multiple_dec.validate(1.3)
    assert exc_info.value.code == "multiple_of"
    
    # Test precision
    field_precision = Number(precision="0.01", numeric_type=float)
    result = field_precision.validate(3.14159)
    assert result == 3.14
    
    # Test invalid string
    field = Number()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not_a_number")
    assert exc_info.value.code == "type"
    
    # Test numeric_type=int conversion
    field_int = Number(numeric_type=int)
    assert field_int.validate(42) == 42
    assert field_int.validate(42.0) == 42
    assert field_int.validate("42") == 42
    
    # Test numeric_type=float conversion
    field_float = Number(numeric_type=float)
    assert field_float.validate(42) == 42.0
    assert field_float.validate("3.14") == 3.14
    
    # Test with Decimal
    field_decimal = Number()
    result = field_decimal.validate(decimal.Decimal("42.50"))
    assert result == decimal.Decimal("42.50")


# LLM-generated content at query #10
#--------------------------

```python
def test_Field_get_default_value():
    # Test with no default value
    field_no_default = Field()
    assert field_no_default.get_default_value() is None
    
    # Test with a static default value
    field_with_static_default = Field(default="test_value")
    assert field_with_static_default.get_default_value() == "test_value"
    
    # Test with a numeric default value
    field_with_numeric_default = Field(default=42)
    assert field_with_numeric_default.get_default_value() == 42
    
    # Test with a callable default value
    def get_default():
        return "callable_result"
    
    field_with_callable_default = Field(default=get_default)
    assert field_with_callable_default.get_default_value() == "callable_result"
    
    # Test with a lambda default value
    field_with_lambda_default = Field(default=lambda: [1, 2, 3])
    assert field_with_lambda_default.get_default_value() == [1, 2, 3]
    
    # Test with None as explicit default
    field_with_none_default = Field(default=None)
    assert field_with_none_default.get_default_value() is None
    
    # Test with allow_null=True and no explicit default (should default to None)
    field_allow_null = Field(allow_null=True)
    assert field_allow_null.get_default_value() is None
    
    # Test with empty string as default
    field_with_empty_string = Field(default="")
    assert field_with_empty_string.get_default_value() == ""
    
    # Test with zero as default
    field_with_zero = Field(default=0)
    assert field_with_zero.get_default_value() == 0
    
    # Test with False as default
    field_with_false = Field(default=False)
    assert field_with_false.get_default_value() is False
    
    # Test with list as default value
    field_with_list = Field(default=[1, 2, 3])
    assert field_with_list.get_default_value() == [1, 2, 3]
    
    # Test with dict as default value
    field_with_dict = Field(default={"key": "value"})
    assert field_with_dict.get_default_value() == {"key": "value"}


# LLM-generated content at query #11
#--------------------------

```python
def test_Object_validate():
    # Test with None and allow_null=True
    obj = Object(allow_null=True)
    assert obj.validate(None) is None
    
    # Test with None and allow_null=False
    obj = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with non-dict/mapping type
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate("not a dict")
    assert exc_info.value.code == "type"
    
    # Test with valid dict
    obj = Object()
    result = obj.validate({"key": "value"})
    assert result == {"key": "value"}
    
    # Test with non-string keys
    obj = Object()
    with pytest.raises(ValidationError):
        obj.validate({1: "value"})
    
    # Test with required properties
    obj = Object(required=["name"])
    with pytest.raises(ValidationError):
        obj.validate({})
    
    result = obj.validate({"name": "John"})
    assert result == {"name": "John"}
    
    # Test with properties and default values
    obj = Object(properties={"name": String(), "age": Integer(default=0)})
    result = obj.validate({"name": "John"})
    assert result == {"name": "John", "age": 0}
    
    # Test with min_properties
    obj = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"key": "value"})
    assert exc_info.value.code == "empty"
    
    obj = Object(min_properties=3)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"key1": "value1", "key2": "value2"})
    assert exc_info.value.code == "min_properties"
    
    # Test with max_properties
    obj = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"key1": "value1", "key2": "value2"})
    assert exc_info.value.code == "max_properties"
    
    # Test with additional_properties=False
    obj = Object(properties={"name": String()}, additional_properties=False)
    with pytest.raises(ValidationError):
        obj.validate({"name": "John", "extra": "field"})
    
    result = obj.validate({"name": "John"})
    assert result == {"name": "John"}
    
    # Test with additional_properties=True (default)
    obj = Object(properties={"name": String()}, additional_properties=True)
    result = obj.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John", "extra": "field"}
    
    # Test with additional_properties as Field
    obj = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = obj.validate({"name": "John", "count": 42})
    assert result == {"name": "John", "count": 42}
    
    # Test with pattern_properties
    obj = Object(pattern_properties={"^num_": Integer()})
    result = obj.validate({"num_one": 1, "num_two": 2})
    assert result == {"num_one": 1, "num_two": 2}
    
    # Test with property_names validation
    obj = Object(property_names=String(pattern="^[a-z]+$"))
    with pytest.raises(ValidationError):
        obj.validate({"Invalid": "value"})
    
    result = obj.validate({"valid": "value"})
    assert result == {"valid": "value"}
    
    # Test with nested properties and validation errors
    obj = Object(properties={"age": Integer(minimum=0)})
    with pytest.raises(ValidationError):
        obj.validate({"age": -5})
    
    # Test with complex nested structure
    obj = Object(
        properties={
            "user": Object(
                properties={"name": String(), "age": Integer()},
                required=["name"]
            )
        }
    )
    result = obj.validate({"user": {"name": "John", "age": 30}})
    assert result == {"user": {"name": "John", "age": 30}}
    
    # Test serialization with Mapping type
    from collections import OrderedDict
    obj = Object()
    result = obj.validate(OrderedDict([("key", "value")]))
    assert result == {"key": "value"}


# LLM-generated content at query #12
#--------------------------

```python
def test_Array_serialize():
    # Test serialize with None value
    field = Array(items=String())
    assert field.serialize(None) is None
    
    # Test serialize with list of items and list of validators
    field = Array(items=[String(), Integer()])
    result = field.serialize(["hello", 42])
    assert result == ["hello", 42]
    
    # Test serialize with single item validator
    field = Array(items=String())
    result = field.serialize(["a", "b", "c"])
    assert result == ["a", "b", "c"]
    
    # Test serialize with None items validator
    field = Array(items=None)
    result = field.serialize([1, "two", 3.0])
    assert result == [1, "two", 3.0]
    
    # Test serialize with Decimal items
    field = Array(items=Decimal())
    result = field.serialize([decimal.Decimal("1.5"), decimal.Decimal("2.5")])
    assert result == [1.5, 2.5]
    
    # Test serialize with nested Array
    inner_array = Array(items=Integer())
    field = Array(items=inner_array)
    result = field.serialize([[1, 2], [3, 4]])
    assert result == [[1, 2], [3, 4]]
    
    # Test serialize with Object items
    field = Array(items=Object(properties={"name": String(), "age": Integer()}))
    result = field.serialize([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
    assert result == [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    
    # Test serialize with mixed types and no validator
    field = Array(items=None)
    result = field.serialize([1, "two", 3.0, None])
    assert result == [1, "two", 3.0, None]
    
    # Test serialize with Boolean items
    field = Array(items=Boolean())
    result = field.serialize([True, False, True])
    assert result == [True, False, True]
    
    # Test serialize with Choice items
    field = Array(items=Choice(choices=[("a", "A"), ("b", "B")]))
    result = field.serialize(["a", "b", "a"])
    assert result == ["a", "b", "a"]
    
    # Test serialize with empty array
    field = Array(items=String())
    result = field.serialize([])
    assert result == []


# LLM-generated content at query #13
#--------------------------

def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling
    field_null = String(allow_null=True)
    assert field_null.validate(None) is None
    
    field_not_null = String(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field_not_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test blank handling
    field_blank = String(allow_blank=True)
    assert field_blank.validate("") == ""
    
    field_not_blank = String(allow_blank=False)
    with pytest.raises(ValidationError) as exc_info:
        field_not_blank.validate("")
    assert exc_info.value.code == "blank"
    
    # Test type validation
    field = String()
    with pytest.raises(ValidationError) as exc_info:
        field.validate(123)
    assert exc_info.value.code == "type"
    
    # Test whitespace trimming
    field_trim = String(trim_whitespace=True)
    assert field_trim.validate("  hello  ") == "hello"
    
    field_no_trim = String(trim_whitespace=False)
    assert field_no_trim.validate("  hello  ") == "  hello  "
    
    # Test min_length
    field_min = String(min_length=3)
    assert field_min.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_min.validate("hi")
    assert exc_info.value.code == "min_length"
    
    # Test max_length
    field_max = String(max_length=5)
    assert field_max.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_max.validate("hello world")
    assert exc_info.value.code == "max_length"
    
    # Test pattern validation
    field_pattern = String(pattern=r"^\d+$")
    assert field_pattern.validate("12345") == "12345"
    with pytest.raises(ValidationError) as exc_info:
        field_pattern.validate("hello")
    assert exc_info.value.code == "pattern"
    
    # Test null character removal
    field = String()
    assert field.validate("hel\0lo") == "hello"
    
    # Test coerce_types with null and allow_blank
    field_coerce = String(allow_blank=True, coerce_types=True)
    assert field_coerce.validate(None) == ""
    
    field_no_coerce = String(allow_blank=True, coerce_types=False)
    with pytest.raises(ValidationError):
        field_no_coerce.validate(None)
    
    # Test coerce_types with empty string and allow_null
    field_coerce_null = String(allow_null=True, coerce_types=True)
    assert field_coerce_null.validate("") is None
    
    field_no_coerce_null = String(allow_null=True, coerce_types=False)
    assert field_no_coerce_null.validate("") == ""
    
    # Test format validation (email example)
    field_email = String(format="email")
    assert field_email.validate("test@example.com") == "test@example.com"
    
    # Test multiple constraints together
    field_complex = String(min_length=2, max_length=10, allow_blank=False, trim_whitespace=True)
    assert field_complex.validate("  hello  ") == "hello"
    with pytest.raises(ValidationError):
        field_complex.validate("a")
    with pytest.raises(ValidationError):
        field_complex.validate("a" * 20)


# LLM-generated content at query #14
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling
    field_allow_null = String(allow_null=True)
    assert field_allow_null.validate(None) is None
    
    field_no_null = String()
    with pytest.raises(ValidationError) as exc_info:
        field_no_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test blank handling
    field_blank = String(allow_blank=True)
    assert field_blank.validate("") == ""
    
    field_no_blank = String(allow_blank=False)
    with pytest.raises(ValidationError) as exc_info:
        field_no_blank.validate("")
    assert exc_info.value.code == "blank"
    
    # Test whitespace trimming
    field_trim = String(trim_whitespace=True)
    assert field_trim.validate("  hello  ") == "hello"
    
    field_no_trim = String(trim_whitespace=False)
    assert field_no_trim.validate("  hello  ") == "  hello  "
    
    # Test type validation
    field = String()
    with pytest.raises(ValidationError) as exc_info:
        field.validate(123)
    assert exc_info.value.code == "type"
    
    # Test max_length
    field_max = String(max_length=5)
    assert field_max.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_max.validate("toolong")
    assert exc_info.value.code == "max_length"
    
    # Test min_length
    field_min = String(min_length=3)
    assert field_min.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_min.validate("hi")
    assert exc_info.value.code == "min_length"
    
    # Test pattern
    field_pattern = String(pattern=r"^\d+$")
    assert field_pattern.validate("123") == "123"
    with pytest.raises(ValidationError) as exc_info:
        field_pattern.validate("abc")
    assert exc_info.value.code == "pattern"
    
    # Test null character removal
    field = String()
    assert field.validate("hel\0lo") == "hello"
    
    # Test coerce_types with allow_blank
    field_coerce = String(allow_blank=True, coerce_types=True)
    assert field_coerce.validate(None) == ""
    
    field_no_coerce = String(allow_blank=True, coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field_no_coerce.validate(None)
    assert exc_info.value.code == "null"
    
    # Test coerce_types with allow_null
    field_coerce_null = String(allow_null=True, coerce_types=True)
    assert field_coerce_null.validate("") is None
    
    field_no_coerce_null = String(allow_null=True, coerce_types=False)
    assert field_no_coerce_null.validate("") == ""
    
    # Test format validation
    field_email = String(format="email")
    assert field_email.validate("test@example.com") == "test@example.com"
    
    # Test combined constraints
    field_combined = String(min_length=2, max_length=10, allow_blank=False)
    assert field_combined.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field_combined.validate("a")
    assert exc_info.value.code == "min_length"


# LLM-generated content at query #15
#--------------------------

```python
def test_Array_validate():
    # Test with None and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None

    # Test with None and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"

    # Test exact_items validation
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "exact_items"

    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "exact_items"

    # Test min_items validation
    field = Array(min_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "min_items"

    # Test min_items=1 returns "empty" error
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"

    # Test max_items validation
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test valid array with min_items
    field = Array(min_items=2)
    result = field.validate([1, 2])
    assert result == [1, 2]

    # Test valid array with max_items
    field = Array(max_items=2)
    result = field.validate([1, 2])
    assert result == [1, 2]

    # Test with single item validator
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test with single item validator - invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError):
        field.validate([1, "not an int", 3])

    # Test with tuple of validators
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]

    # Test with tuple of validators - exact length enforced
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]

    # Test additional_items=False with tuple validators
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "hello", 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test additional_items=True with tuple validators
    field = Array(items=[Integer(), String()], additional_items=True)
    result = field.validate([1, "hello", 3])
    assert result == [1, "hello", 3]

    # Test additional_items with Field validator
    field = Array(items=[Integer()], additional_items=String())
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]

    # Test unique_items=True with unique values
    field = Array(unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test unique_items=True with duplicate values
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert exc_info.value.messages()[0].code == "unique_items"

    # Test empty array
    field = Array()
    result = field.validate([])
    assert result == []

    # Test array with nested validation errors
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "invalid"])
    errors = exc_info.value.messages()
    assert len(errors) > 0

    # Test with None items (no validation)
    field = Array(items=None)
    result = field.validate([1, "hello", None])
    assert result == [1, "hello", None]

    # Test unique_items with string values
    field = Array(unique_items=True)
    result = field.validate(["a", "b", "c"])
    assert result == ["a", "b", "c"]

    # Test unique_items with duplicate strings
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(["a", "b", "a"])
    assert exc_info.value.messages()[0].code == "unique_items"


# LLM-generated content at query #16
#--------------------------

```python
def test_Array():
    # Test basic initialization with no arguments
    field = Array()
    assert field.items is None
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False

    # Test initialization with a single Field as items
    string_field = String()
    field = Array(items=string_field)
    assert field.items is string_field
    assert field.additional_items is False

    # Test initialization with a list of Fields as items
    int_field = Integer()
    bool_field = Boolean()
    field = Array(items=[int_field, bool_field])
    assert field.items == [int_field, bool_field]
    assert field.min_items == 2
    assert field.max_items == 2
    assert field.additional_items is False

    # Test initialization with list of Fields and additional_items=True
    field = Array(items=[int_field, bool_field], additional_items=True)
    assert field.items == [int_field, bool_field]
    assert field.min_items == 2
    assert field.max_items is None
    assert field.additional_items is True

    # Test initialization with list of Fields and additional_items as Field
    string_field = String()
    field = Array(items=[int_field], additional_items=string_field)
    assert field.items == [int_field]
    assert field.additional_items is string_field

    # Test initialization with min_items and max_items
    field = Array(items=string_field, min_items=1, max_items=10)
    assert field.min_items == 1
    assert field.max_items == 10

    # Test initialization with exact_items
    field = Array(items=string_field, exact_items=5)
    assert field.min_items == 5
    assert field.max_items == 5

    # Test initialization with exact_items overriding min_items and max_items
    field = Array(items=string_field, min_items=2, max_items=8, exact_items=5)
    assert field.min_items == 5
    assert field.max_items == 5

    # Test initialization with unique_items
    field = Array(items=string_field, unique_items=True)
    assert field.unique_items is True

    # Test initialization with tuple of items (should be converted to list)
    field = Array(items=(int_field, bool_field))
    assert isinstance(field.items, list)
    assert field.items == [int_field, bool_field]

    # Test initialization with allow_null
    field = Array(items=string_field, allow_null=True)
    assert field.allow_null is True

    # Test initialization with default value
    field = Array(items=string_field, default=[])
    assert field.has_default()
    assert field.get_default_value() == []

    # Test initialization with all parameters
    field = Array(
        items=[int_field, bool_field],
        additional_items=string_field,
        min_items=1,
        max_items=10,
        unique_items=True,
        allow_null=False,
    )
    assert field.items == [int_field, bool_field]
    assert field.additional_items is string_field
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items is True
    assert field.allow_null is False

    # Test that min_items defaults to len(items) when items is a list
    field = Array(items=[int_field, bool_field, string_field])
    assert field.min_items == 3
    assert field.max_items == 3

    # Test that max_items is not set when items is a list but additional_items is True
    field = Array(items=[int_field, bool_field], additional_items=True)
    assert field.min_items == 2
    assert field.max_items is None


# LLM-generated content at query #17
#--------------------------

```python
def test_Union():
    # Test basic initialization with multiple fields
    field1 = String()
    field2 = Integer()
    union = Union(any_of=[field1, field2])
    assert union.any_of == [field1, field2]
    assert union.allow_null is False

    # Test with allow_null child field
    field3 = String(allow_null=True)
    field4 = Integer()
    union = Union(any_of=[field3, field4])
    assert union.allow_null is True

    # Test with multiple allow_null child fields
    field5 = String(allow_null=True)
    field6 = Integer(allow_null=True)
    union = Union(any_of=[field5, field6])
    assert union.allow_null is True

    # Test with single field
    field7 = Float()
    union = Union(any_of=[field7])
    assert len(union.any_of) == 1
    assert union.any_of[0] == field7

    # Test with empty list
    union = Union(any_of=[])
    assert union.any_of == []
    assert union.allow_null is False

    # Test with kwargs passed to parent
    field8 = String()
    field9 = Integer()
    union = Union(any_of=[field8, field9], allow_null=False)
    assert union.allow_null is False

    # Test with allow_null override in kwargs
    field10 = String(allow_null=True)
    field11 = Integer()
    union = Union(any_of=[field10, field11], allow_null=False)
    # Child has allow_null=True, so parent should be set to True
    assert union.allow_null is True

    # Test with complex field types
    object_field = Object(properties={"key": String()})
    array_field = Array(items=Integer())
    union = Union(any_of=[object_field, array_field])
    assert union.any_of == [object_field, array_field]

    # Test with nested Union fields
    inner_union = Union(any_of=[String(), Integer()])
    outer_union = Union(any_of=[inner_union, Float()])
    assert len(outer_union.any_of) == 2
    assert outer_union.any_of[0] == inner_union

    # Test errors attribute
    assert "null" in Union.errors
    assert "union" in Union.errors


# LLM-generated content at query #18
#--------------------------

```python
def test_Const():
    # Test basic construction with a constant value
    const_field = Const(const=42)
    assert const_field.const == 42
    
    # Test construction with None
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    
    # Test construction with string
    const_field_str = Const(const="test")
    assert const_field_str.const == "test"
    
    # Test construction with list
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    # Test construction with dict
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    # Test construction with boolean
    const_field_bool = Const(const=True)
    assert const_field_bool.const is True
    
    # Test that allow_null cannot be passed in kwargs
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)
    
    # Test that allow_null cannot be passed even when const is None
    with pytest.raises(AssertionError):
        Const(const=None, allow_null=False)
    
    # Test construction with other keyword arguments
    const_field_with_kwargs = Const(const=100, default=50)
    assert const_field_with_kwargs.const == 100
    assert const_field_with_kwargs.default == 50
    
    # Test construction with error_messages
    const_field_custom_errors = Const(const=5, error_messages={"const": "Custom error"})
    assert const_field_custom_errors.const == 5


# LLM-generated content at query #19
#--------------------------

```python
def test_Boolean_validate():
    # Test with None and allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None

    # Test with None and allow_null=False
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "null"

    # Test with boolean True
    field = Boolean()
    assert field.validate(True) is True

    # Test with boolean False
    field = Boolean()
    assert field.validate(False) is False

    # Test with string "true" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True

    # Test with string "false" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("false") is False

    # Test with string "on" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("on") is True

    # Test with string "off" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("off") is False

    # Test with string "1" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("1") is True

    # Test with string "0" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("0") is False

    # Test with string "" (empty) and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("") is False

    # Test with integer 1 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(1) is True

    # Test with integer 0 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(0) is False

    # Test with uppercase string "TRUE" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("TRUE") is True

    # Test with uppercase string "FALSE" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("FALSE") is False

    # Test with allow_null=True and empty string
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("") is False

    # Test with allow_null=True and "null" string
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("null") is None

    # Test with allow_null=True and "none" string
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("none") is None

    # Test with non-boolean type and coerce_types=False
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"

    # Test with invalid string and coerce_types=True
    field = Boolean(coerce_types=True)
    try:
        field.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"

    # Test with invalid type (list) and coerce_types=True
    field = Boolean(coerce_types=True)
    try:
        field.validate([])
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"


# LLM-generated content at query #20
#--------------------------

```python
def test_Array_validate():
    # Test with None and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test with non-list value
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"
    
    # Test with dict (not a list)
    with pytest.raises(ValidationError) as exc_info:
        field.validate({})
    assert exc_info.value.messages()[0].code == "type"
    
    # Test empty list with min_items=1
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"
    
    # Test list length less than min_items
    field = Array(min_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.messages()[0].code == "min_items"
    
    # Test list length greater than max_items
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"
    
    # Test exact_items mismatch
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "exact_items"
    
    # Test exact_items match
    field = Array(exact_items=2)
    result = field.validate([1, 2])
    assert result == [1, 2]
    
    # Test basic validation with simple list
    field = Array()
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test with items schema (single Field)
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test with items schema validation failure
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "not an integer", 3])
    assert len(exc_info.value.messages()) > 0
    
    # Test with tuple of item schemas
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]
    
    # Test with tuple of schemas and additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError):
        field.validate([1, "hello", 3])
    
    # Test with tuple of schemas and additional_items as Field
    field = Array(items=[Integer(), String()], additional_items=Boolean())
    result = field.validate([1, "hello", True])
    assert result == [1, "hello", True]
    
    # Test unique_items=True with duplicate values
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert any(msg.code == "unique_items" for msg in exc_info.value.messages())
    
    # Test unique_items=True with unique values
    field = Array(unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test unique_items with string duplicates
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(["a", "b", "a"])
    assert any(msg.code == "unique_items" for msg in exc_info.value.messages())
    
    # Test min_items and max_items both set
    field = Array(min_items=2, max_items=4)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test with nested Array
    field = Array(items=Array(items=Integer()))
    result = field.validate([[1, 2], [3, 4]])
    assert result == [[1, 2], [3, 4]]
    
    # Test with nested Array validation error
    field = Array(items=Array(items=Integer()))
    with pytest.raises(ValidationError):
        field.validate([[1, "invalid"], [3, 4]])
    
    # Test with Object items
    field = Array(items=Object(properties={"name": String()}))
    result = field.validate([{"name": "test"}])
    assert result == [{"name": "test"}]
    
    # Test empty list with allow_null=False and min_items=None
    field = Array()
    result = field.validate([])
    assert result == []
    
    # Test with additional_items=True
    field = Array(items=[Integer()], additional_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #21
#--------------------------

def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling
    field_allow_null = String(allow_null=True)
    assert field_allow_null.validate(None) is None
    
    field_no_null = String()
    with pytest.raises(ValidationError) as exc_info:
        field_no_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test blank string handling
    field_allow_blank = String(allow_blank=True)
    assert field_allow_blank.validate("") == ""
    
    field_no_blank = String(allow_blank=False)
    with pytest.raises(ValidationError) as exc_info:
        field_no_blank.validate("")
    assert exc_info.value.code == "blank"
    
    # Test type validation
    field = String()
    with pytest.raises(ValidationError) as exc_info:
        field.validate(123)
    assert exc_info.value.code == "type"
    
    # Test whitespace trimming
    field = String(trim_whitespace=True)
    assert field.validate("  hello  ") == "hello"
    
    field_no_trim = String(trim_whitespace=False)
    assert field_no_trim.validate("  hello  ") == "  hello  "
    
    # Test max_length
    field = String(max_length=5)
    assert field.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("toolong")
    assert exc_info.value.code == "max_length"
    
    # Test min_length
    field = String(min_length=3)
    assert field.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("hi")
    assert exc_info.value.code == "min_length"
    
    # Test pattern validation
    field = String(pattern=r"^\d+$")
    assert field.validate("123") == "123"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("abc")
    assert exc_info.value.code == "pattern"
    
    # Test null character removal
    field = String()
    assert field.validate("hello\0world") == "helloworld"
    
    # Test null coercion to empty string with allow_blank
    field = String(allow_blank=True, coerce_types=True)
    assert field.validate(None) == ""
    
    # Test empty string coercion to null with allow_null
    field = String(allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test format validation (email example)
    field = String(format="email")
    result = field.validate("test@example.com")
    assert result is not None


# LLM-generated content at query #22
#--------------------------

```python
def test_Object_validate():
    # Test with None and allow_null=True
    obj = Object(allow_null=True)
    assert obj.validate(None) is None

    # Test with None and allow_null=False
    obj = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict type
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with valid dict
    obj = Object()
    result = obj.validate({})
    assert result == {}

    # Test with non-string keys
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.error_messages)

    # Test with properties
    obj = Object(properties={"name": String()})
    result = obj.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with default values
    obj = Object(properties={"name": String(default="Unknown")})
    result = obj.validate({})
    assert result == {"name": "Unknown"}

    # Test with required properties
    obj = Object(required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.error_messages)

    # Test with required property present
    obj = Object(required=["name"], properties={"name": String()})
    result = obj.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with min_properties
    obj = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"key": "value"})
    assert exc_info.value.code == "min_properties"

    # Test with min_properties=1 (empty case)
    obj = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({})
    assert exc_info.value.code == "empty"

    # Test with max_properties
    obj = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"key1": "value1", "key2": "value2"})
    assert exc_info.value.code == "max_properties"

    # Test with additional_properties=True
    obj = Object(additional_properties=True)
    result = obj.validate({"extra": "value"})
    assert result == {"extra": "value"}

    # Test with additional_properties=False
    obj = Object(additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"extra": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.error_messages)

    # Test with additional_properties as Field
    obj = Object(additional_properties=Integer())
    result = obj.validate({"extra": 42})
    assert result == {"extra": 42}

    # Test with pattern_properties
    obj = Object(pattern_properties={"^num": Integer()})
    result = obj.validate({"num_field": 123})
    assert result == {"num_field": 123}

    # Test with pattern_properties validation error
    obj = Object(pattern_properties={"^num": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"num_field": "not_an_int"})
    assert any(msg.code == "type" for msg in exc_info.value.error_messages)

    # Test with property_names validation
    obj = Object(property_names=String(min_length=3))
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"ab": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.error_messages)

    # Test with nested object validation
    obj = Object(properties={"nested": Object(properties={"id": Integer()})})
    result = obj.validate({"nested": {"id": 42}})
    assert result == {"nested": {"id": 42}}

    # Test with nested object validation error
    obj = Object(properties={"nested": Object(properties={"id": Integer()})})
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"nested": {"id": "invalid"}})
    assert any(msg.code == "type" for msg in exc_info.value.error_messages)

    # Test with mixed properties and additional_properties
    obj = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = obj.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with Mapping type (not just dict)
    from collections import UserDict
    obj = Object()
    mapping = UserDict({"key": "value"})
    result = obj.validate(mapping)
    assert "key" in result


# LLM-generated content at query #23
#--------------------------

```python
def test_Union_validate():
    # Test: Union with null value when allow_null is True
    union_field = Union(any_of=[String(allow_null=True), Integer()])
    assert union_field.validate(None) is None

    # Test: Union with null value when allow_null is False
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test: Union validates successfully with first matching child
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate("hello") == "hello"

    # Test: Union validates successfully with second matching child
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate(42) == 42

    # Test: Union validates float as Integer (coercion)
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate(3.14) == 3

    # Test: Union fails when value doesn't match any child
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "union"

    # Test: Union with multiple fields, returns child error when exactly one child matches type
    union_field = Union(any_of=[String(max_length=2), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("toolong")
    assert exc_info.value.messages()[0].code == "max_length"

    # Test: Union with null child allows null
    union_field = Union(any_of=[String(allow_null=True), Integer(allow_null=False)])
    assert union_field.allow_null is True
    assert union_field.validate(None) is None

    # Test: Union with multiple null-allowing children
    union_field = Union(any_of=[String(allow_null=True), Integer(allow_null=True)])
    assert union_field.allow_null is True
    assert union_field.validate(None) is None

    # Test: Union with Boolean field
    union_field = Union(any_of=[Boolean(), Integer()])
    assert union_field.validate(True) is True
    assert union_field.validate(False) is False

    # Test: Union with nested fields
    union_field = Union(any_of=[Array(items=String()), Object(properties={"name": String()})])
    assert union_field.validate(["a", "b"]) == ["a", "b"]
    assert union_field.validate({"name": "John"}) == {"name": "John"}

    # Test: Union returns union error when no children match
    union_field = Union(any_of=[String(min_length=5), Integer(minimum=10)])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(2)
    assert exc_info.value.messages()[0].code == "union"

    # Test: Union with candidate errors from type-correct children
    union_field = Union(any_of=[String(max_length=2), Integer(maximum=5)])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("toolong")
    assert exc_info.value.messages()[0].code == "max_length"

    # Test: Union validates with coercion in child
    union_field = Union(any_of=[String(), Integer(coerce_types=True)])
    assert union_field.validate("123") == "123"

    # Test: Union with Boolean and String
    union_field = Union(any_of=[Boolean(coerce_types=True), String()])
    assert union_field.validate("true") is True
    assert union_field.validate("hello") == "hello"

    # Test: Union empty list of children (edge case)
    union_field = Union(any_of=[])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("value")
    assert exc_info.value.messages()[0].code == "union"

    # Test: Union with Float and Integer
    union_field = Union(any_of=[Float(), Integer()])
    assert union_field.validate(3.14) == 3.14
    assert union_field.validate(42) == 42


# LLM-generated content at query #24
#--------------------------

```python
def test_Object_validate():
    # Test with None and allow_null=True
    obj_field = Object(allow_null=True)
    assert obj_field.validate(None) is None

    # Test with None and allow_null=False
    obj_field = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/mapping type
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with valid empty dict
    obj_field = Object()
    assert obj_field.validate({}) == {}

    # Test with non-string keys
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())

    # Test with required properties
    obj_field = Object(required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test with valid required properties
    obj_field = Object(required=["name"])
    assert obj_field.validate({"name": "John"}) == {"name": "John"}

    # Test with properties schema
    obj_field = Object(properties={"name": String(), "age": Integer()})
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with property validation error
    obj_field = Object(properties={"age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"age": "not an integer"})
    assert any(msg.code == "type" for msg in exc_info.value.messages())

    # Test with default values
    obj_field = Object(properties={"name": String(default="Unknown")})
    result = obj_field.validate({})
    assert result == {"name": "Unknown"}

    # Test with additional_properties=True
    obj_field = Object(additional_properties=True)
    result = obj_field.validate({"extra": "value"})
    assert result == {"extra": "value"}

    # Test with additional_properties=False
    obj_field = Object(additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"extra": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test with additional_properties as Field
    obj_field = Object(additional_properties=Integer())
    result = obj_field.validate({"extra": 42})
    assert result == {"extra": 42}

    # Test with pattern_properties
    obj_field = Object(pattern_properties={"^S_": String(), "^I_": Integer()})
    result = obj_field.validate({"S_name": "John", "I_age": 30})
    assert result == {"S_name": "John", "I_age": 30}

    # Test with min_properties
    obj_field = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert exc_info.value.code == "empty"

    # Test with min_properties > 1
    obj_field = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1})
    assert exc_info.value.code == "min_properties"

    # Test with max_properties
    obj_field = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1, "b": 2})
    assert exc_info.value.code == "max_properties"

    # Test with property_names validation
    obj_field = Object(property_names=String(pattern="^[a-z]+$"))
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"123": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test complex nested structure
    obj_field = Object(
        properties={
            "user": Object(
                properties={"name": String(), "age": Integer()},
                required=["name"]
            )
        },
        required=["user"]
    )
    result = obj_field.validate({"user": {"name": "John", "age": 30}})
    assert result == {"user": {"name": "John", "age": 30}}

    # Test complex nested structure with error
    obj_field = Object(
        properties={
            "user": Object(
                properties={"name": String(), "age": Integer()},
                required=["name"]
            )
        },
        required=["user"]
    )
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"user": {"age": 30}})
    assert any(msg.code == "required" for msg in exc_info.value.messages())


# LLM-generated content at query #25
#--------------------------

```python
def test_Number_validate():
    # Test with None and allow_null=True
    field = Number(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test with empty string and allow_null=True with coerce_types=True
    field = Number(allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test with boolean (should always fail)
    field = Number()
    try:
        field.validate(True)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test with float when numeric_type is int (non-integer)
    field = Number(numeric_type=int)
    try:
        field.validate(3.14)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "integer"
    
    # Test with float when numeric_type is int (integer value)
    field = Number(numeric_type=int)
    assert field.validate(3.0) == 3
    
    # Test with string coercion when coerce_types=False
    field = Number(coerce_types=False)
    try:
        field.validate("123")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test with string coercion when coerce_types=True
    field = Number(coerce_types=True)
    assert field.validate("123") == 123
    
    # Test with invalid string
    field = Number()
    try:
        field.validate("not_a_number")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test with infinity
    field = Number()
    try:
        field.validate(float('inf'))
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "finite"
    
    # Test with NaN
    field = Number()
    try:
        field.validate(float('nan'))
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "finite"
    
    # Test minimum constraint
    field = Number(minimum=10)
    try:
        field.validate(5)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "minimum"
    
    assert field.validate(10) == 10
    assert field.validate(15) == 15
    
    # Test exclusive_minimum constraint
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "exclusive_minimum"
    
    assert field.validate(11) == 11
    
    # Test maximum constraint
    field = Number(maximum=100)
    try:
        field.validate(150)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "maximum"
    
    assert field.validate(100) == 100
    assert field.validate(50) == 50
    
    # Test exclusive_maximum constraint
    field = Number(exclusive_maximum=100)
    try:
        field.validate(100)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "exclusive_maximum"
    
    assert field.validate(99) == 99
    
    # Test precision
    field = Number(precision="0.01", numeric_type=float)
    result = field.validate(3.14159)
    assert abs(result - 3.14) < 0.001
    
    # Test multiple_of with integer
    field = Number(multiple_of=5)
    assert field.validate(10) == 10
    try:
        field.validate(12)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"
    
    # Test multiple_of with float
    field = Number(multiple_of=0.5)
    assert field.validate(2.5) == 2.5
    try:
        field.validate(2.3)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"
    
    # Test with numeric_type=int
    field = Number(numeric_type=int)
    assert field.validate(42) == 42
    assert field.validate("42") == 42
    assert isinstance(field.validate("42"), int)
    
    # Test with numeric_type=float
    field = Number(numeric_type=float)
    assert field.validate(3.14) == 3.14
    assert field.validate("3.14") == 3.14
    assert isinstance(field.validate("3.14"), float)
    
    # Test combined constraints
    field = Number(minimum=0, maximum=100, multiple_of=10)
    assert field.validate(50) == 50
    assert field.validate(0) == 0
    assert field.validate(100) == 100
    try:
        field.validate(55)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"


# LLM-generated content at query #26
#--------------------------

```python
def test_Object_validate():
    # Test basic valid object
    obj_field = Object(properties={"name": String(), "age": Integer()})
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test null value with allow_null=False
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate(None)
    assert exc_info.value.code == "null"

    # Test null value with allow_null=True
    obj_field = Object(allow_null=True)
    result = obj_field.validate(None)
    assert result is None

    # Test non-dict value raises type error
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test non-string keys raise error
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value._messages)

    # Test required properties
    obj_field = Object(properties={"name": String()}, required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert any(msg.code == "required" for msg in exc_info.value._messages)

    # Test min_properties
    obj_field = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1})
    assert exc_info.value.code == "min_properties"

    # Test min_properties=1 raises empty error
    obj_field = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert exc_info.value.code == "empty"

    # Test max_properties
    obj_field = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1, "b": 2})
    assert exc_info.value.code == "max_properties"

    # Test default values for missing properties
    obj_field = Object(properties={"name": String(default="Unknown")})
    result = obj_field.validate({})
    assert result == {"name": "Unknown"}

    # Test additional_properties=True
    obj_field = Object(properties={"name": String()}, additional_properties=True)
    result = obj_field.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John", "extra": "value"}

    # Test additional_properties=False
    obj_field = Object(properties={"name": String()}, additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"name": "John", "extra": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value._messages)

    # Test additional_properties with Field schema
    obj_field = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test pattern_properties
    obj_field = Object(pattern_properties={"^S_": String(), "^I_": Integer()})
    result = obj_field.validate({"S_name": "John", "I_age": 30})
    assert result == {"S_name": "John", "I_age": 30}

    # Test property_names validation
    property_names_field = String(pattern="^[a-z]+$")
    obj_field = Object(property_names=property_names_field)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"Invalid": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value._messages)

    # Test nested object validation with errors
    nested_field = Object(properties={"name": String(min_length=5)})
    obj_field = Object(properties={"user": nested_field})
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"user": {"name": "ab"}})
    assert len(exc_info.value._messages) > 0

    # Test empty object
    obj_field = Object()
    result = obj_field.validate({})
    assert result == {}

    # Test mapping type instead of dict
    obj_field = Object()
    from collections import OrderedDict
    result = obj_field.validate(OrderedDict([("a", 1), ("b", 2)]))
    assert "a" in result and "b" in result


# LLM-generated content at query #27
#--------------------------

```python
def test_Union():
    # Test basic initialization with multiple fields
    string_field = String()
    integer_field = Integer()
    union_field = Union(any_of=[string_field, integer_field])
    
    assert union_field.any_of == [string_field, integer_field]
    assert union_field.allow_null is False

    # Test initialization with allow_null field in any_of
    nullable_string = String(allow_null=True)
    non_nullable_int = Integer()
    union_with_null = Union(any_of=[nullable_string, non_nullable_int])
    
    assert union_with_null.allow_null is True
    assert len(union_with_null.any_of) == 2

    # Test initialization with multiple nullable fields
    nullable_string2 = String(allow_null=True)
    nullable_int = Integer(allow_null=True)
    union_all_nullable = Union(any_of=[nullable_string2, nullable_int])
    
    assert union_all_nullable.allow_null is True

    # Test initialization with empty any_of list
    union_empty = Union(any_of=[])
    
    assert union_empty.any_of == []
    assert union_empty.allow_null is False

    # Test initialization with single field
    single_field = Boolean()
    union_single = Union(any_of=[single_field])
    
    assert len(union_single.any_of) == 1
    assert union_single.any_of[0] == single_field

    # Test initialization with kwargs passed to parent
    union_with_kwargs = Union(any_of=[String()], allow_null=False)
    
    assert union_with_kwargs.allow_null is False

    # Test that allow_null is overridden when any child has allow_null=True
    string_nullable = String(allow_null=True)
    int_non_nullable = Integer(allow_null=False)
    union_override = Union(any_of=[int_non_nullable, string_nullable], allow_null=False)
    
    assert union_override.allow_null is True

    # Test initialization with multiple nullable and non-nullable fields
    fields = [
        String(),
        Integer(allow_null=True),
        Boolean(),
        Float(allow_null=True),
    ]
    union_mixed = Union(any_of=fields)
    
    assert union_mixed.allow_null is True
    assert len(union_mixed.any_of) == 4

    # Test that errors dict is inherited
    assert "null" in Union.errors
    assert "union" in Union.errors


# LLM-generated content at query #28
#--------------------------

```python
def test_Boolean_validate():
    # Test with None and allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Boolean(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with actual boolean True
    field = Boolean()
    assert field.validate(True) is True
    
    # Test with actual boolean False
    field = Boolean()
    assert field.validate(False) is False
    
    # Test string coercion with coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    assert field.validate("True") is True
    assert field.validate("false") is False
    assert field.validate("False") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    
    # Test empty string coercion
    field = Boolean(coerce_types=True)
    assert field.validate("") is False
    
    # Test integer coercion
    field = Boolean(coerce_types=True)
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test null coercion with allow_null=True
    field = Boolean(coerce_types=True, allow_null=True)
    assert field.validate("") is False
    assert field.validate("null") is None
    assert field.validate("none") is None
    
    # Test with coerce_types=False and non-boolean value
    field = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("true")
    assert exc_info.value.code == "type"
    
    # Test with coerce_types=False and boolean value
    field = Boolean(coerce_types=False)
    assert field.validate(True) is True
    
    # Test invalid string value with coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("invalid")
    assert exc_info.value.code == "type"
    
    # Test invalid type with coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.code == "type"


# LLM-generated content at query #29
#--------------------------

```python
def test_Object_validate():
    # Test 1: None value with allow_null=True
    obj = Object(allow_null=True)
    assert obj.validate(None) is None

    # Test 2: None value without allow_null raises error
    obj = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate(None)
    assert exc_info.value.code == "null"

    # Test 3: Non-dict/mapping type raises error
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test 4: Simple dict validation with properties
    obj = Object(properties={"name": String(), "age": Integer()})
    result = obj.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test 5: Non-string keys raise error
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({123: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.error_list)

    # Test 6: Required properties missing
    obj = Object(required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.error_list)

    # Test 7: Default values for missing properties
    obj = Object(properties={"name": String(default="Unknown")})
    result = obj.validate({})
    assert result == {"name": "Unknown"}

    # Test 8: Min properties constraint
    obj = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"name": "John"})
    assert exc_info.value.code == "empty" or exc_info.value.code == "min_properties"

    # Test 9: Max properties constraint
    obj = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"name": "John", "age": 30})
    assert exc_info.value.code == "max_properties"

    # Test 10: Additional properties True (default)
    obj = Object(properties={"name": String()}, additional_properties=True)
    result = obj.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John", "extra": "value"}

    # Test 11: Additional properties False
    obj = Object(properties={"name": String()}, additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"name": "John", "extra": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.error_list)

    # Test 12: Additional properties with Field schema
    obj = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = obj.validate({"name": "John", "count": 42})
    assert result == {"name": "John", "count": 42}

    # Test 13: Pattern properties
    obj = Object(pattern_properties={"^num_": Integer()})
    result = obj.validate({"num_1": 10, "num_2": 20})
    assert result == {"num_1": 10, "num_2": 20}

    # Test 14: Property names validation
    obj = Object(property_names=String(pattern="^[a-z]+$"))
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"Invalid": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.error_list)

    # Test 15: Nested object validation
    obj = Object(
        properties={
            "user": Object(properties={"name": String()})
        }
    )
    result = obj.validate({"user": {"name": "John"}})
    assert result == {"user": {"name": "John"}}

    # Test 16: Nested validation error propagation
    obj = Object(
        properties={
            "user": Object(properties={"age": Integer()})
        }
    )
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"user": {"age": "not_an_int"}})
    assert len(exc_info.value.error_list) > 0

    # Test 17: Empty dict with no constraints
    obj = Object()
    result = obj.validate({})
    assert result == {}

    # Test 18: Mapping type (not just dict)
    obj = Object()
    from collections import OrderedDict
    result = obj.validate(OrderedDict([("key", "value")]))
    assert result == {"key": "value"}

    # Test 19: Multiple errors collected
    obj = Object(required=["name", "age"], properties={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"name": "John", "age": "invalid"})
    assert len(exc_info.value.error_list) >= 1

    # Test 20: Child field validation error
    obj = Object(properties={"name": String(max_length=5)})
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"name": "VeryLongName"})
    assert len(exc_info.value.error_list) > 0


# LLM-generated content at query #30
#--------------------------

```python
def test_Const():
    # Test basic instantiation with a simple value
    const_field = Const(const=42)
    assert const_field.const == 42
    
    # Test instantiation with None
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    
    # Test instantiation with string
    const_field_str = Const(const="test_value")
    assert const_field_str.const == "test_value"
    
    # Test instantiation with list
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    # Test instantiation with dict
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    # Test that allow_null cannot be passed as kwarg
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)
    
    # Test that allow_null cannot be passed even with None const
    with pytest.raises(AssertionError):
        Const(const=None, allow_null=False)
    
    # Test instantiation with other Field kwargs
    const_field_with_kwargs = Const(const="value", default="default_value")
    assert const_field_with_kwargs.const == "value"
    assert const_field_with_kwargs.default == "default_value"
    
    # Test instantiation with error_message kwarg
    const_field_with_error = Const(const=100, error_message="Custom error")
    assert const_field_with_error.const == 100
    assert const_field_with_error.error_message == "Custom error"


# LLM-generated content at query #31
#--------------------------

```python
def test_Union():
    # Test basic Union initialization with multiple fields
    string_field = String()
    integer_field = Integer()
    union_field = Union(any_of=[string_field, integer_field])
    
    assert union_field.any_of == [string_field, integer_field]
    assert union_field.allow_null is False
    
    # Test Union with allow_null=True passed explicitly
    union_field_explicit = Union(any_of=[string_field, integer_field], allow_null=True)
    assert union_field_explicit.allow_null is True
    
    # Test Union where one child has allow_null=True
    nullable_string = String(allow_null=True)
    non_nullable_integer = Integer()
    union_field_nullable_child = Union(any_of=[nullable_string, non_nullable_integer])
    
    assert union_field_nullable_child.allow_null is True
    
    # Test Union with multiple nullable children
    nullable_string2 = String(allow_null=True)
    nullable_integer = Integer(allow_null=True)
    union_field_all_nullable = Union(any_of=[nullable_string2, nullable_integer])
    
    assert union_field_all_nullable.allow_null is True
    
    # Test Union with empty any_of list
    union_field_empty = Union(any_of=[])
    assert union_field_empty.any_of == []
    assert union_field_empty.allow_null is False
    
    # Test Union with single field
    single_field_union = Union(any_of=[string_field])
    assert len(single_field_union.any_of) == 1
    assert single_field_union.allow_null is False
    
    # Test Union with various field types
    fields = [String(), Integer(), Float(), Boolean(), Array(items=String())]
    union_multi = Union(any_of=fields)
    assert len(union_multi.any_of) == 5
    
    # Test Union with custom kwargs
    union_with_kwargs = Union(any_of=[string_field], allow_null=False)
    assert union_with_kwargs.allow_null is False


# LLM-generated content at query #32
#--------------------------

```python
def test_Number_validate():
    # Test basic integer validation
    field = Number(numeric_type=int)
    assert field.validate(42) == 42
    assert field.validate("42") == 42
    assert field.validate(42.0) == 42
    
    # Test float validation
    field = Number(numeric_type=float)
    assert field.validate(3.14) == 3.14
    assert field.validate("3.14") == 3.14
    assert field.validate(42) == 42.0
    
    # Test null handling
    field = Number(allow_null=True)
    assert field.validate(None) is None
    
    field = Number(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test empty string with allow_null and coerce_types
    field = Number(allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test boolean rejection
    field = Number()
    with pytest.raises(ValidationError) as exc_info:
        field.validate(True)
    assert exc_info.value.code == "type"
    
    # Test non-integer float when numeric_type is int
    field = Number(numeric_type=int)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(3.14)
    assert exc_info.value.code == "integer"
    
    # Test type coercion disabled
    field = Number(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("42")
    assert exc_info.value.code == "type"
    
    # Test infinite and NaN values
    field = Number()
    with pytest.raises(ValidationError) as exc_info:
        field.validate(float('inf'))
    assert exc_info.value.code == "finite"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate(float('-inf'))
    assert exc_info.value.code == "finite"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate(float('nan'))
    assert exc_info.value.code == "finite"
    
    # Test minimum constraint
    field = Number(minimum=10)
    assert field.validate(10) == 10
    assert field.validate(20) == 20
    with pytest.raises(ValidationError) as exc_info:
        field.validate(5)
    assert exc_info.value.code == "minimum"
    
    # Test exclusive_minimum constraint
    field = Number(exclusive_minimum=10)
    assert field.validate(11) == 11
    with pytest.raises(ValidationError) as exc_info:
        field.validate(10)
    assert exc_info.value.code == "exclusive_minimum"
    
    # Test maximum constraint
    field = Number(maximum=100)
    assert field.validate(100) == 100
    assert field.validate(50) == 50
    with pytest.raises(ValidationError) as exc_info:
        field.validate(150)
    assert exc_info.value.code == "maximum"
    
    # Test exclusive_maximum constraint
    field = Number(exclusive_maximum=100)
    assert field.validate(99) == 99
    with pytest.raises(ValidationError) as exc_info:
        field.validate(100)
    assert exc_info.value.code == "exclusive_maximum"
    
    # Test multiple_of with integer
    field = Number(multiple_of=5)
    assert field.validate(10) == 10
    assert field.validate(0) == 0
    with pytest.raises(ValidationError) as exc_info:
        field.validate(7)
    assert exc_info.value.code == "multiple_of"
    
    # Test multiple_of with float
    field = Number(multiple_of=0.5)
    assert field.validate(1.0) == 1.0
    assert field.validate(1.5) == 1.5
    with pytest.raises(ValidationError) as exc_info:
        field.validate(1.3)
    assert exc_info.value.code == "multiple_of"
    
    # Test precision
    field = Number(numeric_type=float, precision="0.01")
    result = field.validate(3.145)
    assert result == 3.15  # Rounded to 2 decimal places
    
    # Test precision with integer type
    field = Number(numeric_type=int, precision="1")
    assert field.validate(42) == 42
    
    # Test decimal.Decimal input
    field = Number()
    result = field.validate(decimal.Decimal("42.5"))
    assert result == decimal.Decimal("42.5")
    
    # Test invalid string input
    field = Number()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not_a_number")
    assert exc_info.value.code == "type"
    
    # Test combined constraints
    field = Number(minimum=0, maximum=100, multiple_of=10)
    assert field.validate(50) == 50
    with pytest.raises(ValidationError):
        field.validate(150)
    with pytest.raises(ValidationError):
        field.validate(55)


# LLM-generated content at query #33
#--------------------------

```python
def test_Number_validate():
    # Test basic integer validation
    field = Number()
    assert field.validate(42) == 42
    assert field.validate(0) == 0
    assert field.validate(-10) == -10
    
    # Test float validation
    assert field.validate(3.14) == 3.14
    assert field.validate(-2.5) == -2.5
    
    # Test string coercion
    assert field.validate("100") == 100
    assert field.validate("3.14") == 3.14
    
    # Test null handling
    field_nullable = Number(allow_null=True)
    assert field_nullable.validate(None) is None
    
    field_not_nullable = Number(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field_not_nullable.validate(None)
    assert exc_info.value.code == "null"
    
    # Test empty string with allow_null and coerce_types
    assert field_nullable.validate("") is None
    
    # Test boolean rejection
    with pytest.raises(ValidationError) as exc_info:
        field.validate(True)
    assert exc_info.value.code == "type"
    
    # Test minimum constraint
    field_min = Number(minimum=10)
    assert field_min.validate(10) == 10
    assert field_min.validate(100) == 100
    with pytest.raises(ValidationError) as exc_info:
        field_min.validate(5)
    assert exc_info.value.code == "minimum"
    
    # Test maximum constraint
    field_max = Number(maximum=50)
    assert field_max.validate(50) == 50
    assert field_max.validate(10) == 10
    with pytest.raises(ValidationError) as exc_info:
        field_max.validate(100)
    assert exc_info.value.code == "maximum"
    
    # Test exclusive_minimum
    field_excl_min = Number(exclusive_minimum=10)
    assert field_excl_min.validate(11) == 11
    with pytest.raises(ValidationError) as exc_info:
        field_excl_min.validate(10)
    assert exc_info.value.code == "exclusive_minimum"
    
    # Test exclusive_maximum
    field_excl_max = Number(exclusive_maximum=50)
    assert field_excl_max.validate(49) == 49
    with pytest.raises(ValidationError) as exc_info:
        field_excl_max.validate(50)
    assert exc_info.value.code == "exclusive_maximum"
    
    # Test finite constraint (inf and nan)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(float('inf'))
    assert exc_info.value.code == "finite"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate(float('-inf'))
    assert exc_info.value.code == "finite"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate(float('nan'))
    assert exc_info.value.code == "finite"
    
    # Test multiple_of with integer
    field_multiple = Number(multiple_of=5)
    assert field_multiple.validate(10) == 10
    assert field_multiple.validate(0) == 0
    with pytest.raises(ValidationError) as exc_info:
        field_multiple.validate(7)
    assert exc_info.value.code == "multiple_of"
    
    # Test multiple_of with decimal
    field_multiple_decimal = Number(multiple_of=0.5)
    assert field_multiple_decimal.validate(1.0) == 1.0
    assert field_multiple_decimal.validate(2.5) == 2.5
    with pytest.raises(ValidationError) as exc_info:
        field_multiple_decimal.validate(1.3)
    assert exc_info.value.code == "multiple_of"
    
    # Test precision
    field_precision = Number(precision="0.01")
    result = field_precision.validate(3.14159)
    assert result == decimal.Decimal("3.14")
    
    # Test type coercion disabled
    field_no_coerce = Number(coerce_types=False)
    assert field_no_coerce.validate(42) == 42
    with pytest.raises(ValidationError) as exc_info:
        field_no_coerce.validate("100")
    assert exc_info.value.code == "type"
    
    # Test integer type validation with float
    from typesystem.fields import Integer
    field_int = Integer()
    with pytest.raises(ValidationError) as exc_info:
        field_int.validate(3.14)
    assert exc_info.value.code == "integer"
    
    # Integer validation should pass for integer-like floats
    assert field_int.validate(3.0) == 3
    
    # Test decimal.Decimal input
    field_decimal = Number()
    result = field_decimal.validate(decimal.Decimal("10.5"))
    assert result == decimal.Decimal("10.5")


# LLM-generated content at query #34
#--------------------------

def test_Boolean_validate():
    # Test with None and allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test with True boolean
    field = Boolean()
    assert field.validate(True) is True
    
    # Test with False boolean
    field = Boolean()
    assert field.validate(False) is False
    
    # Test with string "true" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    
    # Test with string "false" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("false") is False
    
    # Test with string "on" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("on") is True
    
    # Test with string "off" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("off") is False
    
    # Test with string "1" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("1") is True
    
    # Test with string "0" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("0") is False
    
    # Test with empty string and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("") is False
    
    # Test with integer 1 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(1) is True
    
    # Test with integer 0 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(0) is False
    
    # Test with uppercase string "TRUE" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("TRUE") is True
    
    # Test with coerce_null_values "null" and allow_null=True
    field = Boolean(coerce_types=True, allow_null=True)
    assert field.validate("null") is None
    
    # Test with coerce_null_values "none" and allow_null=True
    field = Boolean(coerce_types=True, allow_null=True)
    assert field.validate("none") is None
    
    # Test with invalid string and coerce_types=True
    field = Boolean(coerce_types=True)
    try:
        field.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test with non-boolean type and coerce_types=False
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test with integer and coerce_types=False
    field = Boolean(coerce_types=False)
    try:
        field.validate(1)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test with empty string and allow_null=True and coerce_types=True
    field = Boolean(coerce_types=True, allow_null=True)
    assert field.validate("") is None


# LLM-generated content at query #35
#--------------------------

```python
def test_Boolean_validate():
    # Test valid boolean values
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test coercion of string values
    assert field.validate("true") is True
    assert field.validate("false") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    assert field.validate("") is False
    
    # Test coercion of numeric values
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test case insensitivity for string values
    assert field.validate("TRUE") is True
    assert field.validate("FALSE") is False
    assert field.validate("On") is True
    assert field.validate("OFF") is False
    
    # Test null handling with allow_null=False
    field_no_null = Boolean(allow_null=False)
    try:
        field_no_null.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test null handling with allow_null=True
    field_allow_null = Boolean(allow_null=True)
    assert field_allow_null.validate(None) is None
    
    # Test null coercion with allow_null=True and coerce_types=True
    field_coerce_null = Boolean(allow_null=True, coerce_types=True)
    assert field_coerce_null.validate("null") is None
    assert field_coerce_null.validate("none") is None
    
    # Test invalid type with coerce_types=False
    field_no_coerce = Boolean(coerce_types=False)
    try:
        field_no_coerce.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid type with coerce_types=True
    field_coerce = Boolean(coerce_types=True)
    try:
        field_coerce.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid type with non-coercible value
    try:
        field.validate([])
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test null coercion with empty string when allow_null=True
    assert field_allow_null.validate("") is False
    assert field_coerce_null.validate("") is False


# LLM-generated content at query #36
#--------------------------

```python
def test_Object_validate():
    # Test: null value with allow_null=True
    obj_field = Object(allow_null=True)
    assert obj_field.validate(None) is None

    # Test: null value with allow_null=False
    obj_field = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate(None)
    assert exc_info.value.code == "null"

    # Test: non-dict/mapping value
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test: non-string keys
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())

    # Test: simple valid dict
    obj_field = Object()
    result = obj_field.validate({"key": "value"})
    assert result == {"key": "value"}

    # Test: properties validation
    obj_field = Object(properties={"name": String(), "age": Integer()})
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test: required properties missing
    obj_field = Object(required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test: property with default value
    obj_field = Object(properties={"name": String(default="Unknown")})
    result = obj_field.validate({})
    assert result == {"name": "Unknown"}

    # Test: min_properties
    obj_field = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"key": "value"})
    assert exc_info.value.code == "min_properties"

    # Test: min_properties=1 (empty)
    obj_field = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert exc_info.value.code == "empty"

    # Test: max_properties
    obj_field = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"key1": "value1", "key2": "value2"})
    assert exc_info.value.code == "max_properties"

    # Test: additional_properties=True
    obj_field = Object(additional_properties=True)
    result = obj_field.validate({"key1": "value1", "key2": "value2"})
    assert result == {"key1": "value1", "key2": "value2"}

    # Test: additional_properties=False
    obj_field = Object(properties={"name": String()}, additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"name": "John", "extra": "field"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test: additional_properties with Field
    obj_field = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = obj_field.validate({"name": "John", "count": 42})
    assert result == {"name": "John", "count": 42}

    # Test: pattern_properties
    obj_field = Object(pattern_properties={"^num_": Integer()})
    result = obj_field.validate({"num_1": 10, "num_2": 20})
    assert result == {"num_1": 10, "num_2": 20}

    # Test: property_names validation
    name_field = String(pattern="^[a-z]+$")
    obj_field = Object(property_names=name_field)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"Invalid": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test: nested object validation
    inner_obj = Object(properties={"id": Integer()})
    outer_obj = Object(properties={"nested": inner_obj})
    result = outer_obj.validate({"nested": {"id": 123}})
    assert result == {"nested": {"id": 123}}

    # Test: nested object validation with error
    inner_obj = Object(properties={"id": Integer()})
    outer_obj = Object(properties={"nested": inner_obj})
    with pytest.raises(ValidationError):
        outer_obj.validate({"nested": {"id": "not_an_int"}})

    # Test: empty dict
    obj_field = Object()
    result = obj_field.validate({})
    assert result == {}

    # Test: Field as first argument (properties shorthand)
    inner_field = String()
    obj_field = Object(inner_field)
    result = obj_field.validate({"key": "value"})
    assert result == {"key": "value"}

    # Test: multiple validation errors
    obj_field = Object(
        properties={"name": String(), "age": Integer()},
        required=["name", "age"]
    )
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    messages = exc_info.value.messages()
    assert len(messages) >= 2


# LLM-generated content at query #37
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling with allow_null=True
    field = String(allow_null=True)
    assert field.validate(None) is None
    
    # Test null handling with allow_null=False
    field = String(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test blank string with allow_blank=False
    field = String(allow_blank=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("")
    assert exc_info.value.code == "blank"
    
    # Test blank string with allow_blank=True
    field = String(allow_blank=True)
    assert field.validate("") == ""
    
    # Test whitespace trimming
    field = String(trim_whitespace=True)
    assert field.validate("  hello  ") == "hello"
    
    # Test no whitespace trimming
    field = String(trim_whitespace=False)
    assert field.validate("  hello  ") == "  hello  "
    
    # Test max_length validation
    field = String(max_length=5)
    assert field.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("toolong")
    assert exc_info.value.code == "max_length"
    
    # Test min_length validation
    field = String(min_length=3)
    assert field.validate("hello") == "hello"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("hi")
    assert exc_info.value.code == "min_length"
    
    # Test pattern validation with string pattern
    field = String(pattern=r"^\d+$")
    assert field.validate("123") == "123"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("abc")
    assert exc_info.value.code == "pattern"
    
    # Test pattern validation with compiled regex
    import re
    pattern = re.compile(r"^[a-z]+$")
    field = String(pattern=pattern)
    assert field.validate("abc") == "abc"
    with pytest.raises(ValidationError) as exc_info:
        field.validate("ABC")
    assert exc_info.value.code == "pattern"
    
    # Test non-string type error
    field = String()
    with pytest.raises(ValidationError) as exc_info:
        field.validate(123)
    assert exc_info.value.code == "type"
    
    # Test null character removal
    field = String()
    assert field.validate("hel\x00lo") == "hello"
    
    # Test null to empty string coercion with allow_blank and coerce_types
    field = String(allow_blank=True, coerce_types=True)
    assert field.validate(None) == ""
    
    # Test empty string to null coercion with allow_null and coerce_types
    field = String(allow_null=True, coerce_types=True, trim_whitespace=True)
    assert field.validate("") is None
    
    # Test coerce_types disabled
    field = String(allow_blank=True, coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test email format validation
    field = String(format="email")
    result = field.validate("test@example.com")
    assert result is not None
    
    # Test uuid format validation
    field = String(format="uuid")
    result = field.validate("550e8400-e29b-41d4-a716-446655440000")
    assert result is not None
    
    # Test combined constraints
    field = String(min_length=2, max_length=10, pattern=r"^[a-z]+$")
    assert field.validate("hello") == "hello"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate("a")
    assert exc_info.value.code == "min_length"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate("verylongstring")
    assert exc_info.value.code == "max_length"
    
    with pytest.raises(ValidationError) as exc_info:
        field.validate("HELLO")
    assert exc_info.value.code == "pattern"


# LLM-generated content at query #38
#--------------------------

```python
def test_Boolean_validate():
    # Test valid boolean values
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test None with allow_null=False (default)
    field = Boolean()
    try:
        field.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test None with allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    # Test string coercion with coerce_types=True (default)
    field = Boolean()
    assert field.validate("true") is True
    assert field.validate("True") is True
    assert field.validate("TRUE") is True
    assert field.validate("false") is False
    assert field.validate("False") is False
    assert field.validate("FALSE") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    
    # Test integer coercion
    field = Boolean()
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test empty string with coerce_types=True
    field = Boolean()
    assert field.validate("") is False
    
    # Test empty string with allow_null=True and coerce_types=True
    field = Boolean(allow_null=True)
    assert field.validate("") is None
    
    # Test null string values with allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate("null") is None
    assert field.validate("none") is None
    
    # Test invalid string with coerce_types=True
    field = Boolean()
    try:
        field.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid type with coerce_types=True
    field = Boolean()
    try:
        field.validate([])
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid type with coerce_types=False
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test non-boolean type with coerce_types=False
    field = Boolean(coerce_types=False)
    try:
        field.validate(1)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test None with coerce_types=False and allow_null=False
    field = Boolean(coerce_types=False)
    try:
        field.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "null"


# LLM-generated content at query #39
#--------------------------

```python
def test_Union():
    # Test basic Union initialization with multiple fields
    field1 = String()
    field2 = Integer()
    union = Union(any_of=[field1, field2])
    assert union.any_of == [field1, field2]
    assert union.allow_null is False

    # Test Union with allow_null child field
    field1_nullable = String(allow_null=True)
    field2 = Integer()
    union = Union(any_of=[field1_nullable, field2])
    assert union.allow_null is True

    # Test Union with multiple nullable children
    field1_nullable = String(allow_null=True)
    field2_nullable = Integer(allow_null=True)
    union = Union(any_of=[field1_nullable, field2_nullable])
    assert union.allow_null is True

    # Test Union with no nullable children
    field1 = String()
    field2 = Integer()
    field3 = Float()
    union = Union(any_of=[field1, field2, field3])
    assert union.allow_null is False

    # Test Union with single field
    field1 = String()
    union = Union(any_of=[field1])
    assert union.any_of == [field1]
    assert len(union.any_of) == 1

    # Test Union with many fields
    fields = [String(), Integer(), Float(), Boolean(), Array()]
    union = Union(any_of=fields)
    assert union.any_of == fields
    assert len(union.any_of) == 5

    # Test Union with kwargs passed to parent
    field1 = String()
    field2 = Integer()
    union = Union(any_of=[field1, field2], allow_null=False)
    assert union.allow_null is False

    # Test Union with nullable child overrides explicit allow_null
    field1_nullable = String(allow_null=True)
    field2 = Integer()
    union = Union(any_of=[field1_nullable, field2], allow_null=False)
    assert union.allow_null is True

    # Test Union with complex nested fields
    nested_object = Object(properties={"name": String()})
    nested_array = Array(items=Integer())
    union = Union(any_of=[nested_object, nested_array])
    assert union.any_of == [nested_object, nested_array]
    assert union.allow_null is False

    # Test Union error messages are defined
    assert "null" in union.errors
    assert "union" in union.errors
    assert union.errors["null"] == "May not be null."
    assert union.errors["union"] == "Did not match any valid type."


# LLM-generated content at query #40
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test whitespace trimming (default behavior)
    assert field.validate("  hello  ") == "hello"
    
    # Test with trim_whitespace=False
    field_no_trim = String(trim_whitespace=False)
    assert field_no_trim.validate("  hello  ") == "  hello  "
    
    # Test null handling
    field_allow_null = String(allow_null=True)
    assert field_allow_null.validate(None) is None
    
    field_no_null = String(allow_null=False)
    with_error = field_no_null.validate_or_error(None)
    assert with_error.error is not None
    assert with_error.error.code == "null"
    
    # Test blank string handling
    field_allow_blank = String(allow_blank=True)
    assert field_allow_blank.validate("") == ""
    
    field_no_blank = String(allow_blank=False)
    with_error = field_no_blank.validate_or_error("")
    assert with_error.error is not None
    assert with_error.error.code == "blank"
    
    # Test type validation
    field = String()
    with_error = field.validate_or_error(123)
    assert with_error.error is not None
    assert with_error.error.code == "type"
    
    # Test max_length
    field_max = String(max_length=5)
    assert field_max.validate("hello") == "hello"
    with_error = field_max.validate_or_error("toolong")
    assert with_error.error is not None
    assert with_error.error.code == "max_length"
    
    # Test min_length
    field_min = String(min_length=3)
    assert field_min.validate("hello") == "hello"
    with_error = field_min.validate_or_error("hi")
    assert with_error.error is not None
    assert with_error.error.code == "min_length"
    
    # Test pattern validation
    field_pattern = String(pattern=r"^\d+$")
    assert field_pattern.validate("123") == "123"
    with_error = field_pattern.validate_or_error("abc")
    assert with_error.error is not None
    assert with_error.error.code == "pattern"
    
    # Test pattern with compiled regex
    import re
    compiled_pattern = re.compile(r"^[a-z]+$")
    field_compiled = String(pattern=compiled_pattern)
    assert field_compiled.validate("abc") == "abc"
    with_error = field_compiled.validate_or_error("123")
    assert with_error.error is not None
    assert with_error.error.code == "pattern"
    
    # Test null character removal
    field = String()
    assert field.validate("hel\x00lo") == "hello"
    
    # Test coerce_types with allow_blank
    field_coerce = String(allow_blank=True, coerce_types=True)
    assert field_coerce.validate(None) == ""
    
    # Test coerce_types with allow_null
    field_coerce_null = String(allow_null=True, coerce_types=True)
    assert field_coerce_null.validate("") is None
    
    # Test format validation (email example)
    field_email = String(format="email")
    assert field_email.validate("test@example.com") == "test@example.com"
    with_error = field_email.validate_or_error("invalid-email")
    assert with_error.error is not None
    assert with_error.error.code == "format"
    
    # Test combination of constraints
    field_complex = String(min_length=2, max_length=10, allow_blank=False)
    assert field_complex.validate("hello") == "hello"
    with_error = field_complex.validate_or_error("a")
    assert with_error.error is not None
    assert with_error.error.code == "min_length"


# LLM-generated content at query #41
#--------------------------

```python
def test_Boolean_validate():
    # Test with None and allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Boolean(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with True
    field = Boolean()
    assert field.validate(True) is True
    
    # Test with False
    field = Boolean()
    assert field.validate(False) is False
    
    # Test coerce string "true" to True
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    
    # Test coerce string "false" to False
    field = Boolean(coerce_types=True)
    assert field.validate("false") is False
    
    # Test coerce string "on" to True
    field = Boolean(coerce_types=True)
    assert field.validate("on") is True
    
    # Test coerce string "off" to False
    field = Boolean(coerce_types=True)
    assert field.validate("off") is False
    
    # Test coerce string "1" to True
    field = Boolean(coerce_types=True)
    assert field.validate("1") is True
    
    # Test coerce string "0" to False
    field = Boolean(coerce_types=True)
    assert field.validate("0") is False
    
    # Test coerce empty string to False
    field = Boolean(coerce_types=True)
    assert field.validate("") is False
    
    # Test coerce integer 1 to True
    field = Boolean(coerce_types=True)
    assert field.validate(1) is True
    
    # Test coerce integer 0 to False
    field = Boolean(coerce_types=True)
    assert field.validate(0) is False
    
    # Test coerce uppercase strings
    field = Boolean(coerce_types=True)
    assert field.validate("TRUE") is True
    assert field.validate("FALSE") is False
    
    # Test coerce null values with allow_null=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("null") is None
    assert field.validate("none") is None
    assert field.validate("") is None
    
    # Test invalid string with coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("invalid")
    assert exc_info.value.code == "type"
    
    # Test invalid type with coerce_types=False
    field = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("true")
    assert exc_info.value.code == "type"
    
    # Test invalid type with coerce_types=False and non-boolean
    field = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(1)
    assert exc_info.value.code == "type"
    
    # Test coerce_types=False with None and allow_null=False
    field = Boolean(coerce_types=False, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test mixed case strings
    field = Boolean(coerce_types=True)
    assert field.validate("TrUe") is True
    assert field.validate("FaLsE") is False
    
    # Test invalid integer with coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(2)
    assert exc_info.value.code == "type"
    
    # Test invalid type like list with coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.code == "type"


# LLM-generated content at query #42
#--------------------------

```python
def test_Object_validate():
    # Test basic valid object
    field = Object(properties={"name": String(), "age": Integer()})
    result = field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test null value with allow_null=True
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

    # Test null value with allow_null=False
    field = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"

    # Test non-dict/mapping type
    field = Object()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test required properties
    field = Object(properties={"name": String()}, required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        field.validate({})
    error = exc_info.value
    assert any(msg.code == "required" for msg in error.messages())

    # Test invalid property key (non-string)
    field = Object()
    with pytest.raises(ValidationError) as exc_info:
        field.validate({123: "value"})
    error = exc_info.value
    assert any(msg.code == "invalid_key" for msg in error.messages())

    # Test min_properties
    field = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate({"a": 1})
    assert exc_info.value.code == "empty" or exc_info.value.code == "min_properties"

    # Test max_properties
    field = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate({"a": 1, "b": 2})
    assert exc_info.value.code == "max_properties"

    # Test with default values
    field = Object(properties={"name": String(default="Unknown")})
    result = field.validate({})
    assert result == {"name": "Unknown"}

    # Test additional_properties=True
    field = Object(properties={"name": String()}, additional_properties=True)
    result = field.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John", "extra": "value"}

    # Test additional_properties=False
    field = Object(properties={"name": String()}, additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate({"name": "John", "extra": "value"})
    error = exc_info.value
    assert any(msg.code == "invalid_property" for msg in error.messages())

    # Test additional_properties with Field validator
    field = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test pattern_properties
    field = Object(pattern_properties={"^num_": Integer()})
    result = field.validate({"num_1": 10, "num_2": 20})
    assert result == {"num_1": 10, "num_2": 20}

    # Test property_names validation
    property_names_field = String(pattern="^[a-z]+$")
    field = Object(property_names=property_names_field)
    with pytest.raises(ValidationError) as exc_info:
        field.validate({"Name": "value"})
    error = exc_info.value
    assert any(msg.code == "invalid_property" for msg in error.messages())

    # Test nested object validation
    field = Object(properties={
        "user": Object(properties={"name": String()})
    })
    result = field.validate({"user": {"name": "John"}})
    assert result == {"user": {"name": "John"}}

    # Test nested object with validation error
    field = Object(properties={
        "user": Object(properties={"age": Integer()})
    })
    with pytest.raises(ValidationError):
        field.validate({"user": {"age": "not_an_int"}})

    # Test empty object
    field = Object()
    result = field.validate({})
    assert result == {}

    # Test with Mapping type
    from collections import OrderedDict
    field = Object(properties={"name": String()})
    result = field.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}


# LLM-generated content at query #43
#--------------------------

```python
def test_Union():
    # Test basic initialization with single field
    field1 = String()
    union1 = Union(any_of=[field1])
    assert union1.any_of == [field1]
    assert union1.allow_null is False

    # Test initialization with multiple fields
    field2 = Integer()
    field3 = Float()
    union2 = Union(any_of=[field2, field3])
    assert union2.any_of == [field2, field3]
    assert union2.allow_null is False

    # Test initialization with allow_null child field
    field_nullable = String(allow_null=True)
    union3 = Union(any_of=[field_nullable])
    assert union3.any_of == [field_nullable]
    assert union3.allow_null is True

    # Test initialization with mixed nullable and non-nullable fields
    field_non_null = String()
    field_nullable2 = Integer(allow_null=True)
    union4 = Union(any_of=[field_non_null, field_nullable2])
    assert union4.any_of == [field_non_null, field_nullable2]
    assert union4.allow_null is True

    # Test initialization with multiple nullable fields
    field_nullable3 = Float(allow_null=True)
    field_nullable4 = Boolean(allow_null=True)
    union5 = Union(any_of=[field_nullable3, field_nullable4])
    assert union5.any_of == [field_nullable3, field_nullable4]
    assert union5.allow_null is True

    # Test initialization with kwargs passed to parent
    union6 = Union(any_of=[String()], allow_null=True)
    assert union6.allow_null is True

    # Test initialization with empty any_of list
    union7 = Union(any_of=[])
    assert union7.any_of == []
    assert union7.allow_null is False

    # Test initialization with complex field types
    object_field = Object(properties={"name": String()})
    array_field = Array(items=Integer())
    union8 = Union(any_of=[object_field, array_field])
    assert union8.any_of == [object_field, array_field]
    assert union8.allow_null is False

    # Test that allow_null is set correctly when at least one child has allow_null
    field_null_true = String(allow_null=True)
    field_null_false = Integer(allow_null=False)
    union9 = Union(any_of=[field_null_false, field_null_true])
    assert union9.allow_null is True

    # Test initialization with all non-nullable fields
    field_a = String(allow_null=False)
    field_b = Integer(allow_null=False)
    union10 = Union(any_of=[field_a, field_b])
    assert union10.allow_null is False


# LLM-generated content at query #44
#--------------------------

```python
def test_Boolean_validate():
    # Test with None and allow_null=True
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None and allow_null=False
    field = Boolean(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with boolean True
    field = Boolean()
    assert field.validate(True) is True
    
    # Test with boolean False
    field = Boolean()
    assert field.validate(False) is False
    
    # Test with string "true" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    
    # Test with string "false" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("false") is False
    
    # Test with string "on" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("on") is True
    
    # Test with string "off" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("off") is False
    
    # Test with string "1" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("1") is True
    
    # Test with string "0" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("0") is False
    
    # Test with empty string and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("") is False
    
    # Test with integer 1 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(1) is True
    
    # Test with integer 0 and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate(0) is False
    
    # Test with uppercase "TRUE" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("TRUE") is True
    
    # Test with uppercase "FALSE" and coerce_types=True
    field = Boolean(coerce_types=True)
    assert field.validate("FALSE") is False
    
    # Test with invalid string and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("invalid")
    assert exc_info.value.code == "type"
    
    # Test with non-boolean type and coerce_types=False
    field = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate("true")
    assert exc_info.value.code == "type"
    
    # Test with "null" string and allow_null=True and coerce_types=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("null") is None
    
    # Test with "none" string and allow_null=True and coerce_types=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("none") is None
    
    # Test with empty string and allow_null=True and coerce_types=True
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test with list type and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.code == "type"
    
    # Test with dict type and coerce_types=True
    field = Boolean(coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate({})
    assert exc_info.value.code == "type"


# LLM-generated content at query #45
#--------------------------

```python
def test_Union_validate():
    # Test: Union with null value when allow_null is True
    union_field = Union(any_of=[String(allow_null=True), Integer()])
    assert union_field.validate(None) is None
    
    # Test: Union with null value when allow_null is False
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test: Union matches first type (String)
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate("hello")
    assert result == "hello"
    
    # Test: Union matches second type (Integer)
    union_field = Union(any_of=[String(), Integer()])
    result = union_field.validate(42)
    assert result == 42
    
    # Test: Union with multiple types, matches Boolean
    union_field = Union(any_of=[Boolean(), String(), Integer()])
    result = union_field.validate(True)
    assert result is True
    
    # Test: Union with validation constraints, passes constraint
    union_field = Union(any_of=[Integer(minimum=0, maximum=100), String()])
    result = union_field.validate(50)
    assert result == 50
    
    # Test: Union with validation constraints, fails constraint but matches other type
    union_field = Union(any_of=[Integer(minimum=0, maximum=100), String()])
    result = union_field.validate("valid_string")
    assert result == "valid_string"
    
    # Test: Union with validation constraints, fails all constraints with type error
    union_field = Union(any_of=[Integer(), Float()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate("not_a_number")
    assert exc_info.value.messages()[0].code == "union"
    
    # Test: Union with one child having non-type error, returns that error
    union_field = Union(any_of=[Integer(minimum=10), String()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(5)
    error = exc_info.value
    assert error.messages()[0].code == "minimum"
    
    # Test: Union with multiple non-type errors, returns union error
    union_field = Union(any_of=[Integer(minimum=10), Integer(minimum=20)])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(5)
    assert exc_info.value.messages()[0].code == "union"
    
    # Test: Union with child allowing null
    union_field = Union(any_of=[String(allow_null=True), Integer()])
    assert union_field.allow_null is True
    assert union_field.validate(None) is None
    
    # Test: Union with empty string coercion to null
    union_field = Union(any_of=[String(allow_null=True, coerce_types=True), Integer()])
    result = union_field.validate("")
    assert result == ""
    
    # Test: Union with Float matching integer value
    union_field = Union(any_of=[Integer(), Float()])
    result = union_field.validate(3.14)
    assert result == 3.14
    
    # Test: Union with Boolean coercion
    union_field = Union(any_of=[Boolean(coerce_types=True), String()])
    result = union_field.validate("true")
    assert result is True
    
    # Test: Union with Array and Object
    union_field = Union(any_of=[Array(items=Integer()), Object()])
    result = union_field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test: Union with Object matching
    union_field = Union(any_of=[Array(items=Integer()), Object()])
    result = union_field.validate({"key": "value"})
    assert result == {"key": "value"}


# LLM-generated content at query #46
#--------------------------

def test_String():
    # Test basic initialization
    field = String()
    assert field.allow_blank is False
    assert field.trim_whitespace is True
    assert field.max_length is None
    assert field.min_length is None
    assert field.pattern is None
    assert field.pattern_regex is None
    assert field.format is None
    assert field.coerce_types is True
    
    # Test with allow_blank=True and no default
    field = String(allow_blank=True)
    assert field.allow_blank is True
    assert field.has_default() is True
    assert field.get_default_value() == ""
    
    # Test with allow_blank=True and explicit default
    field = String(allow_blank=True, default="custom")
    assert field.get_default_value() == "custom"
    
    # Test with max_length
    field = String(max_length=10)
    assert field.max_length == 10
    
    # Test with min_length
    field = String(min_length=5)
    assert field.min_length == 5
    
    # Test with pattern as string
    field = String(pattern=r"^\d+$")
    assert field.pattern == r"^\d+$"
    assert field.pattern_regex is not None
    assert field.pattern_regex.pattern == r"^\d+$"
    
    # Test with pattern as compiled regex
    import re
    compiled_pattern = re.compile(r"^[a-z]+$")
    field = String(pattern=compiled_pattern)
    assert field.pattern == r"^[a-z]+$"
    assert field.pattern_regex is compiled_pattern
    
    # Test with format
    field = String(format="email")
    assert field.format == "email"
    
    # Test with coerce_types
    field = String(coerce_types=False)
    assert field.coerce_types is False
    
    # Test with all parameters
    field = String(
        title="Username",
        description="User's login name",
        default="guest",
        allow_null=False,
        read_only=False,
        allow_blank=False,
        trim_whitespace=True,
        max_length=50,
        min_length=3,
        pattern=r"^[a-zA-Z0-9_]+$",
        format="email",
        coerce_types=True
    )
    assert field.title == "Username"
    assert field.description == "User's login name"
    assert field.get_default_value() == "guest"
    assert field.allow_null is False
    assert field.read_only is False
    assert field.max_length == 50
    assert field.min_length == 3
    assert field.pattern == r"^[a-zA-Z0-9_]+$"
    assert field.format == "email"
    
    # Test allow_null with no default sets default to None
    field = String(allow_null=True)
    assert field.has_default() is True
    assert field.get_default_value() is None
    
    # Test trim_whitespace option
    field = String(trim_whitespace=False)
    assert field.trim_whitespace is False


# LLM-generated content at query #47
#--------------------------

```python
def test_Const():
    # Test basic initialization with a constant value
    const_field = Const(const=42)
    assert const_field.const == 42
    assert const_field.allow_null is False
    
    # Test initialization with None as constant
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    assert const_field_none.allow_null is False
    
    # Test initialization with string constant
    const_field_str = Const(const="test_value")
    assert const_field_str.const == "test_value"
    
    # Test initialization with list constant
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    # Test initialization with dict constant
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    # Test that allow_null cannot be passed as kwarg
    with pytest.raises(AssertionError):
        Const(const=42, allow_null=True)
    
    # Test initialization with other Field kwargs
    const_field_with_description = Const(const=100, description="A constant field")
    assert const_field_with_description.const == 100
    assert const_field_with_description.description == "A constant field"
    
    # Test initialization with boolean constant
    const_field_bool = Const(const=True)
    assert const_field_bool.const is True
    
    # Test initialization with float constant
    const_field_float = Const(const=3.14)
    assert const_field_float.const == 3.14


# LLM-generated content at query #48
#--------------------------

```python
def test_Array_validate():
    # Test with None value and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    # Test with None value and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with non-list value
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.code == "type"
    
    # Test with non-list value (dict)
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate({"key": "value"})
    assert exc_info.value.code == "type"
    
    # Test exact_items validation - too few items
    field = Array(exact_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.code == "exact_items"
    
    # Test exact_items validation - too many items
    field = Array(exact_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3, 4])
    assert exc_info.value.code == "exact_items"
    
    # Test exact_items validation - correct items
    field = Array(exact_items=3)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test min_items validation - empty list with min_items=1
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.code == "empty"
    
    # Test min_items validation - too few items
    field = Array(min_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.code == "min_items"
    
    # Test max_items validation - too many items
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.code == "max_items"
    
    # Test with single Field items validator
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test with single Field items validator - invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError):
        field.validate([1, "invalid", 3])
    
    # Test with list of Field items validators
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]
    
    # Test with list of Field items validators - too many items with additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError):
        field.validate([1, "hello", "extra"])
    
    # Test with list of Field items validators - additional items allowed
    field = Array(items=[Integer(), String()], additional_items=True)
    result = field.validate([1, "hello", "extra"])
    assert result == [1, "hello", "extra"]
    
    # Test with list of Field items validators - additional items validated
    field = Array(items=[Integer()], additional_items=String())
    result = field.validate([1, "hello", "world"])
    assert result == [1, "hello", "world"]
    
    # Test unique_items=True - all unique
    field = Array(unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Test unique_items=True - duplicate items
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 2, 3])
    assert exc_info.value.code == "unique_items"
    
    # Test empty list with no constraints
    field = Array()
    result = field.validate([])
    assert result == []
    
    # Test with nested validators
    field = Array(items=Object(properties={"name": String(), "age": Integer()}))
    result = field.validate([
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ])
    assert result == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ]
    
    # Test with nested validators - invalid nested item
    field = Array(items=Object(properties={"age": Integer()}))
    with pytest.raises(ValidationError):
        field.validate([{"age": "not an integer"}])
    
    # Test with no items validator - returns as is
    field = Array()
    result = field.validate([1, "string", 3.14, True])
    assert result == [1, "string", 3.14, True]
    
    # Test min_items validation - exact minimum
    field = Array(min_items=2)
    result = field.validate([1, 2])
    assert result == [1, 2]
    
    # Test max_items validation - exact maximum
    field = Array(max_items=2)
    result = field.validate([1, 2])
    assert result == [1, 2]


# LLM-generated content at query #49
#--------------------------

```python
def test_Union_validate():
    # Test with null value when allow_null is True
    union_field = Union(any_of=[String(), Integer()], allow_null=True)
    assert union_field.validate(None) is None
    
    # Test with null value when allow_null is False
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test with valid string value
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate("hello") == "hello"
    
    # Test with valid integer value
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.validate(42) == 42
    
    # Test with invalid value that doesn't match any type
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "union"
    
    # Test with child containing null when child allows null
    union_field = Union(any_of=[String(allow_null=True), Integer()])
    assert union_field.allow_null is True
    assert union_field.validate(None) is None
    
    # Test with candidate error from child (non-type error)
    union_field = Union(any_of=[Integer(minimum=10), String()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(5)
    # Should raise the candidate error from Integer validator
    assert exc_info.value.messages()[0].code == "minimum"
    
    # Test with multiple children, one matches
    union_field = Union(any_of=[Integer(minimum=100), Integer(maximum=50)])
    assert union_field.validate(25) == 25
    
    # Test with string matching first in union
    union_field = Union(any_of=[String(max_length=5), String()])
    assert union_field.validate("hi") == "hi"
    
    # Test with boolean (should not match string or integer by default)
    union_field = Union(any_of=[String(), Integer()])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(True)
    assert exc_info.value.messages()[0].code == "union"
    
    # Test with float matching integer type
    union_field = Union(any_of=[Integer(), String()])
    assert union_field.validate(3.0) == 3
    
    # Test with validation error that has non-type code
    union_field = Union(any_of=[Integer(minimum=50), Integer(maximum=10)])
    with pytest.raises(ValidationError) as exc_info:
        union_field.validate(25)
    # Should raise candidate error (minimum constraint from first child)
    assert exc_info.value.messages()[0].code == "minimum"


# LLM-generated content at query #50
#--------------------------

```python
def test_Array_validate():
    # Test with None and allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None

    # Test with None and allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"

    # Test with empty list and min_items=1
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"

    # Test with list shorter than min_items
    field = Array(min_items=3)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2])
    assert exc_info.value.messages()[0].code == "min_items"

    # Test with list longer than max_items
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test exact_items mismatch
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "exact_items"

    # Test exact_items match
    field = Array(exact_items=2)
    assert field.validate([1, 2]) == [1, 2]

    # Test with single Field validator
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]

    # Test with single Field validator and invalid item
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "invalid", 3])
    assert len(exc_info.value.messages()) == 1

    # Test with list of Field validators
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "hello"]) == [1, "hello"]

    # Test with list of Field validators and additional_items
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "hello", "world"]) == [1, "hello", "world"]

    # Test with list of Field validators and additional_items=False
    field = Array(items=[Integer()], additional_items=False)
    result = field.validate([1])
    assert result == [1]

    # Test unique_items with duplicates
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert any(msg.code == "unique_items" for msg in exc_info.value.messages())

    # Test unique_items without duplicates
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

    # Test with nested validators and errors
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "not_an_int", 3])
    errors = exc_info.value.messages()
    assert len(errors) == 1
    assert errors[0].index == [1]

    # Test with list of validators and errors
    field = Array(items=[Integer(), String()])
    with pytest.raises(ValidationError) as exc_info:
        field.validate(["invalid_int", 123])
    errors = exc_info.value.messages()
    assert len(errors) == 2

    # Test valid empty array
    field = Array()
    assert field.validate([]) == []

    # Test valid array with mixed types when no validator
    field = Array()
    result = field.validate([1, "string", 3.14, None])
    assert result == [1, "string", 3.14, None]

    # Test min_items and max_items both set
    field = Array(min_items=2, max_items=4)
    assert field.validate([1, 2]) == [1, 2]
    assert field.validate([1, 2, 3, 4]) == [1, 2, 3, 4]

    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "min_items"

    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3, 4, 5])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test with coercible types in items
    field = Array(items=Integer())
    assert field.validate(["1", "2", "3"]) == [1, 2, 3]

    # Test unique_items with string duplicates
    field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(["a", "b", "a"])
    assert any(msg.code == "unique_items" for msg in exc_info.value.messages())


# LLM-generated content at query #51
#--------------------------

```python
def test_Number_validate():
    # Test with None and allow_null=True
    field = Number(allow_null=True)
    assert field.validate(None) is None

    # Test with None and allow_null=False
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "null"

    # Test with empty string and allow_null=True with coerce_types=True
    field = Number(allow_null=True, coerce_types=True)
    assert field.validate("") is None

    # Test with boolean value
    field = Number()
    try:
        field.validate(True)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"

    # Test with float when numeric_type is int and value is not integer
    field = Number()
    field.numeric_type = int
    try:
        field.validate(3.14)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "integer"

    # Test with non-numeric string and coerce_types=False
    field = Number(coerce_types=False)
    try:
        field.validate("not_a_number")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"

    # Test with valid integer
    field = Number()
    assert field.validate(42) == 42

    # Test with valid float
    field = Number()
    assert field.validate(3.14) == 3.14

    # Test with string representation of number
    field = Number()
    result = field.validate("123")
    assert result == 123

    # Test with infinity
    field = Number()
    try:
        field.validate(float('inf'))
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "finite"

    # Test with NaN
    field = Number()
    try:
        field.validate(float('nan'))
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "finite"

    # Test minimum constraint
    field = Number(minimum=10)
    try:
        field.validate(5)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "minimum"

    # Test minimum constraint passes
    field = Number(minimum=10)
    assert field.validate(10) == 10

    # Test exclusive_minimum constraint
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "exclusive_minimum"

    # Test exclusive_minimum constraint passes
    field = Number(exclusive_minimum=10)
    assert field.validate(11) == 11

    # Test maximum constraint
    field = Number(maximum=100)
    try:
        field.validate(150)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "maximum"

    # Test maximum constraint passes
    field = Number(maximum=100)
    assert field.validate(100) == 100

    # Test exclusive_maximum constraint
    field = Number(exclusive_maximum=100)
    try:
        field.validate(100)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "exclusive_maximum"

    # Test exclusive_maximum constraint passes
    field = Number(exclusive_maximum=100)
    assert field.validate(99) == 99

    # Test multiple_of with integer
    field = Number(multiple_of=5)
    try:
        field.validate(7)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"

    # Test multiple_of with integer passes
    field = Number(multiple_of=5)
    assert field.validate(15) == 15

    # Test multiple_of with float
    field = Number(multiple_of=0.5)
    try:
        field.validate(1.3)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"

    # Test multiple_of with float passes
    field = Number(multiple_of=0.5)
    assert field.validate(1.5) == 1.5

    # Test precision rounding
    field = Number(precision="0.01")
    result = field.validate(3.146)
    assert result == 3.15

    # Test numeric_type conversion to int
    field = Number()
    field.numeric_type = int
    result = field.validate(42.7)
    assert result == 42
    assert isinstance(result, int)

    # Test numeric_type conversion to float
    field = Number()
    field.numeric_type = float
    result = field.validate(42)
    assert result == 42.0
    assert isinstance(result, float)

    # Test with Decimal input
    field = Number()
    decimal_value = decimal.Decimal("123.45")
    result = field.validate(decimal_value)
    assert result == decimal.Decimal("123.45")

    # Test coerce_types=False with non-numeric type
    field = Number(coerce_types=False)
    try:
        field.validate([1, 2, 3])
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code == "type"


# LLM-generated content at query #52
#--------------------------

```python
def test_Object_validate():
    # Test basic valid object
    obj_field = Object(properties={"name": String(), "age": Integer()})
    result = obj_field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test null value when allow_null is False
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate(None)
    assert exc_info.value.code == "null"

    # Test null value when allow_null is True
    obj_field = Object(allow_null=True)
    result = obj_field.validate(None)
    assert result is None

    # Test non-dict value
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test non-string keys
    obj_field = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value._messages)

    # Test required properties
    obj_field = Object(properties={"name": String()}, required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert any(msg.code == "required" for msg in exc_info.value._messages)

    # Test required property present
    obj_field = Object(properties={"name": String()}, required=["name"])
    result = obj_field.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test property with default value
    obj_field = Object(properties={"name": String(default="Unknown")})
    result = obj_field.validate({})
    assert result == {"name": "Unknown"}

    # Test min_properties
    obj_field = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1})
    assert exc_info.value.code == "min_properties"

    # Test min_properties with value 1 (empty)
    obj_field = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({})
    assert exc_info.value.code == "empty"

    # Test max_properties
    obj_field = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"a": 1, "b": 2})
    assert exc_info.value.code == "max_properties"

    # Test additional_properties=True (default)
    obj_field = Object(properties={"name": String()}, additional_properties=True)
    result = obj_field.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John", "extra": "value"}

    # Test additional_properties=False
    obj_field = Object(properties={"name": String()}, additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"name": "John", "extra": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value._messages)

    # Test additional_properties with Field schema
    obj_field = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = obj_field.validate({"name": "John", "count": 5})
    assert result == {"name": "John", "count": 5}

    # Test pattern_properties
    obj_field = Object(pattern_properties={"^S_": String()})
    result = obj_field.validate({"S_name": "John"})
    assert result == {"S_name": "John"}

    # Test property_names validation
    obj_field = Object(property_names=String(pattern="^[a-z]+$"))
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"Invalid": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value._messages)

    # Test nested object validation error
    obj_field = Object(properties={"age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        obj_field.validate({"age": "not_a_number"})
    assert exc_info.value._messages

    # Test mapping type (not just dict)
    from collections import OrderedDict
    obj_field = Object(properties={"name": String()})
    mapping = OrderedDict([("name", "John")])
    result = obj_field.validate(mapping)
    assert result == {"name": "John"}

    # Test combination of required and default
    obj_field = Object(
        properties={"name": String(), "status": String(default="active")},
        required=["name"]
    )
    result = obj_field.validate({"name": "John"})
    assert result == {"name": "John", "status": "active"}

    # Test invalid nested property
    obj_field = Object(properties={"user": Object(properties={"age": Integer()})})
    with pytest.raises(ValidationError):
        obj_field.validate({"user": {"age": "invalid"}})


# LLM-generated content at query #53
#--------------------------

```python
def test_Array_validate():
    # Test null value with allow_null=True
    field = Array(allow_null=True)
    assert field.validate(None) is None

    # Test null value with allow_null=False
    field = Array(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test non-list type
    field = Array()
    with pytest.raises(ValidationError) as exc_info:
        field.validate("not a list")
    assert exc_info.value.messages()[0].code == "type"

    # Test exact_items validation
    field = Array(exact_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "exact_items"

    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "exact_items"

    # Test min_items validation
    field = Array(min_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1])
    assert exc_info.value.messages()[0].code == "min_items"

    # Test min_items=1 shows "empty" error
    field = Array(min_items=1)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([])
    assert exc_info.value.messages()[0].code == "empty"

    # Test max_items validation
    field = Array(max_items=2)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 3])
    assert exc_info.value.messages()[0].code == "max_items"

    # Test valid array with single item schema
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test array with invalid item type
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "not an int", 3])
    assert len(exc_info.value.messages()) == 1
    assert exc_info.value.messages()[0].index == [1]

    # Test array with tuple of schemas
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]

    # Test array with additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, "hello", "extra"])
    assert exc_info.value.messages()[0].code == "additional_items"

    # Test array with additional_items as Field
    field = Array(items=[Integer()], additional_items=String())
    result = field.validate([1, "hello", "world"])
    assert result == [1, "hello", "world"]

    # Test unique_items validation
    field = Array(items=Integer(), unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        field.validate([1, 2, 1])
    assert exc_info.value.messages()[0].code == "unique_items"

    # Test unique_items with valid unique items
    field = Array(items=Integer(), unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test empty array
    field = Array()
    result = field.validate([])
    assert result == []

    # Test array without schema validation
    field = Array()
    result = field.validate([1, "string", {"key": "value"}])
    assert result == [1, "string", {"key": "value"}]

    # Test array with multiple validation errors
    field = Array(items=Integer())
    with pytest.raises(ValidationError) as exc_info:
        field.validate(["not int", "also not int"])
    assert len(exc_info.value.messages()) == 2

    # Test min_items with valid count
    field = Array(min_items=2)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

    # Test max_items with valid count
    field = Array(max_items=3)
    result = field.validate([1, 2])
    assert result == [1, 2]

    # Test array with nested schemas
    field = Array(items=Array(items=Integer()))
    result = field.validate([[1, 2], [3, 4]])
    assert result == [[1, 2], [3, 4]]

    # Test array with exact_items equal to list length
    field = Array(exact_items=3)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #54
#--------------------------

```python
def test_Object_validate():
    # Test with None and allow_null=True
    obj = Object(allow_null=True)
    assert obj.validate(None) is None

    # Test with None and allow_null=False
    obj = Object(allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict value
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with valid empty dict
    obj = Object()
    assert obj.validate({}) == {}

    # Test with non-string keys
    obj = Object()
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({1: "value"})
    assert exc_info.value.code == "invalid_key"

    # Test with required properties
    obj = Object(required=["name"])
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test with properties validation
    obj = Object(properties={"name": String()})
    result = obj.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with properties having defaults
    obj = Object(properties={"name": String(default="John")})
    result = obj.validate({})
    assert result == {"name": "John"}

    # Test with invalid property value
    obj = Object(properties={"age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"age": "not an int"})
    assert exc_info.value.code == "type"

    # Test with min_properties
    obj = Object(min_properties=2)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"a": 1})
    assert exc_info.value.code == "min_properties"

    # Test with min_properties=1 (empty error)
    obj = Object(min_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({})
    assert exc_info.value.code == "empty"

    # Test with max_properties
    obj = Object(max_properties=1)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"a": 1, "b": 2})
    assert exc_info.value.code == "max_properties"

    # Test with additional_properties=True (default)
    obj = Object(properties={"name": String()})
    result = obj.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John", "extra": "field"}

    # Test with additional_properties=False
    obj = Object(properties={"name": String()}, additional_properties=False)
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"name": "John", "extra": "field"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test with additional_properties as Field
    obj = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = obj.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with pattern_properties
    obj = Object(pattern_properties={"^num_": Integer()})
    result = obj.validate({"num_1": 10, "num_2": 20})
    assert result == {"num_1": 10, "num_2": 20}

    # Test with invalid pattern_properties value
    obj = Object(pattern_properties={"^num_": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"num_1": "not an int"})
    assert exc_info.value.code == "type"

    # Test with property_names validation
    obj = Object(property_names=String(pattern="^[a-z]+$"))
    with pytest.raises(ValidationError) as exc_info:
        obj.validate({"Name": "value"})
    assert any(msg.code == "invalid_property" for msg in exc_info.value.messages())

    # Test with valid property_names
    obj = Object(property_names=String(pattern="^[a-z]+$"))
    result = obj.validate({"name": "value"})
    assert result == {"name": "value"}

    # Test complex nested structure
    obj = Object(
        properties={
            "user": Object(
                properties={"name": String(), "age": Integer()},
                required=["name"]
            )
        }
    )
    result = obj.validate({"user": {"name": "John", "age": 30}})
    assert result == {"user": {"name": "John", "age": 30}}

    # Test with Mapping (not just dict)
    from collections import UserDict
    mapping = UserDict({"key": "value"})
    obj = Object()
    result = obj.validate(mapping)
    assert result == {"key": "value"}


# LLM-generated content at query #55
#--------------------------

```python
def test_Union():
    # Test basic Union initialization with multiple fields
    field1 = String()
    field2 = Integer()
    union = Union(any_of=[field1, field2])
    assert union.any_of == [field1, field2]
    assert union.allow_null is False

    # Test Union with allow_null child field
    field1 = String(allow_null=True)
    field2 = Integer()
    union = Union(any_of=[field1, field2])
    assert union.allow_null is True

    # Test Union with multiple allow_null child fields
    field1 = String(allow_null=True)
    field2 = Integer(allow_null=True)
    union = Union(any_of=[field1, field2])
    assert union.allow_null is True

    # Test Union with no allow_null child fields
    field1 = String()
    field2 = Integer()
    field3 = Boolean()
    union = Union(any_of=[field1, field2, field3])
    assert union.allow_null is False

    # Test Union with kwargs passed to parent
    field1 = String()
    field2 = Integer()
    union = Union(any_of=[field1, field2], allow_null=True)
    assert union.allow_null is True
    assert union.any_of == [field1, field2]

    # Test Union with single field
    field1 = String()
    union = Union(any_of=[field1])
    assert union.any_of == [field1]
    assert len(union.any_of) == 1

    # Test Union with many fields
    fields = [String(), Integer(), Boolean(), Float(), Array()]
    union = Union(any_of=fields)
    assert union.any_of == fields
    assert len(union.any_of) == 5

    # Test Union with empty any_of list
    union = Union(any_of=[])
    assert union.any_of == []
    assert union.allow_null is False

    # Test Union with mixed allow_null values
    field1 = String(allow_null=False)
    field2 = Integer(allow_null=True)
    field3 = Boolean(allow_null=False)
    union = Union(any_of=[field1, field2, field3])
    assert union.allow_null is True

    # Test Union preserves field order
    field1 = String()
    field2 = Integer()
    field3 = Boolean()
    union = Union(any_of=[field1, field2, field3])
    assert union.any_of[0] is field1
    assert union.any_of[1] is field2
    assert union.any_of[2] is field3


