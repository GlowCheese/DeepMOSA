####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test with single field
    all_of_single = AllOf(all_of=[string_field])
    assert all_of_single.all_of == [string_field]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test with additional valid kwargs
    all_of_with_kwargs = AllOf(all_of=[string_field, integer_field], description="Test")
    assert all_of_with_kwargs.all_of == [string_field, integer_field]


# LLM-generated content at query #2
#--------------------------

```python
def test_IfThenElse_validate():
    """Test IfThenElse.validate() method"""
    from typesystem.fields import String, Integer, Boolean
    
    # Test case 1: if_clause matches, then_clause is executed
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=String()
    )
    result = if_then_else.validate("test")
    assert result == "test"
    
    # Test case 2: if_clause doesn't match, else_clause is executed
    if_then_else = IfThenElse(
        if_clause=Integer(),
        then_clause=String(),
        else_clause=Boolean()
    )
    result = if_then_else.validate(True)
    assert result is True
    
    # Test case 3: then_clause is None, defaults to Any()
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=None,
        else_clause=Integer()
    )
    result = if_then_else.validate("test")
    assert result == "test"
    
    # Test case 4: else_clause is None, defaults to Any()
    if_then_else = IfThenElse(
        if_clause=Integer(),
        then_clause=String(),
        else_clause=None
    )
    result = if_then_else.validate("test")
    assert result == "test"
    
    # Test case 5: both then_clause and else_clause are None
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=None,
        else_clause=None
    )
    result = if_then_else.validate("test")
    assert result == "test"
    
    # Test case 6: if_clause matches, then_clause validates value
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=String()
    )
    result = if_then_else.validate(42)
    assert result == 42
    
    # Test case 7: if_clause doesn't match, else_clause validates value
    if_then_else = IfThenElse(
        if_clause=Integer(),
        then_clause=Integer(),
        else_clause=String()
    )
    result = if_then_else.validate("test")
    assert result == "test"


# LLM-generated content at query #3
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Field()
    field2 = Field()
    all_of = AllOf(all_of=[field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test with single field
    single_field = Field()
    all_of_single = AllOf(all_of=[single_field])
    assert all_of_single.all_of == [single_field]
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        AllOf(all_of=[field1], allow_null=True)
    
    # Test with kwargs passed to parent
    all_of_with_kwargs = AllOf(all_of=[field1, field2], default=None)
    assert all_of_with_kwargs.all_of == [field1, field2]
    assert all_of_with_kwargs.default is None
    
    # Test with multiple fields
    field3 = Field()
    field4 = Field()
    field5 = Field()
    all_of_multiple = AllOf(all_of=[field1, field2, field3, field4, field5])
    assert len(all_of_multiple.all_of) == 5
    assert all_of_multiple.all_of == [field1, field2, field3, field4, field5]


# LLM-generated content at query #4
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with a list of fields
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]

    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []

    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]

    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)

    # Test that allow_null=False is not allowed
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=False)

    # Test other kwargs are accepted
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="test description")
    assert one_of_with_kwargs.one_of == [field1, field2]

    # Test that one_of parameter is required
    with pytest.raises(TypeError):
        OneOf()


# LLM-generated content at query #5
#--------------------------

```python
def test_IfThenElse():
    # Test basic constructor with all parameters
    if_field = Any()
    then_field = Any()
    else_field = Any()
    
    field = IfThenElse(if_clause=if_field, then_clause=then_field, else_clause=else_field)
    
    assert field.if_clause is if_field
    assert field.then_clause is then_field
    assert field.else_clause is else_field
    
    # Test constructor with only if_clause (then and else should default to Any())
    field2 = IfThenElse(if_clause=if_field)
    
    assert field2.if_clause is if_field
    assert isinstance(field2.then_clause, Any)
    assert isinstance(field2.else_clause, Any)
    
    # Test constructor with if_clause and then_clause (else should default to Any())
    field3 = IfThenElse(if_clause=if_field, then_clause=then_field)
    
    assert field3.if_clause is if_field
    assert field3.then_clause is then_field
    assert isinstance(field3.else_clause, Any)
    
    # Test constructor with if_clause and else_clause (then should default to Any())
    field4 = IfThenElse(if_clause=if_field, else_clause=else_field)
    
    assert field4.if_clause is if_field
    assert isinstance(field4.then_clause, Any)
    assert field4.else_clause is else_field
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        IfThenElse(if_clause=if_field, allow_null=True)


# LLM-generated content at query #6
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        Not(negated=Any(), allow_null=True)
    
    # Test with additional kwargs
    not_field_with_kwargs = Not(negated=Any(), description="Test description")
    assert not_field_with_kwargs.negated is not None
    
    # Test that negated parameter is required
    with pytest.raises(TypeError):
        Not()


# LLM-generated content at query #7
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are accepted
    field_with_kwargs = NeverMatch(description="test field")
    assert isinstance(field_with_kwargs, Field)
    
    # Test that validation always fails
    with pytest.raises(Exception):  # validation_error exception
        field.validate("any_value")
    
    with pytest.raises(Exception):
        field.validate(None)
    
    with pytest.raises(Exception):
        field.validate(123)
    
    # Test error message
    assert field.errors["never"] == "This never validates."


# LLM-generated content at query #8
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    field1 = String()
    field2 = Integer()
    all_of = AllOf(all_of=[field1, field2])
    
    assert all_of.all_of == [field1, field2]
    assert len(all_of.all_of) == 2
    
    # Test initialization with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test initialization with single field
    all_of_single = AllOf(all_of=[field1])
    assert all_of_single.all_of == [field1]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        AllOf(all_of=[field1], allow_null=True)
    
    # Test initialization with multiple fields
    field3 = String()
    field4 = Integer()
    field5 = String()
    all_of_multiple = AllOf(all_of=[field3, field4, field5])
    assert len(all_of_multiple.all_of) == 3
    assert all_of_multiple.all_of[0] is field3
    assert all_of_multiple.all_of[1] is field4
    assert all_of_multiple.all_of[2] is field5
    
    # Test that other kwargs are accepted
    all_of_with_kwargs = AllOf(all_of=[field1], description="test description")
    assert all_of_with_kwargs.all_of == [field1]


# LLM-generated content at query #9
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with valid fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test that other kwargs are accepted
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="Test field")
    assert one_of_with_kwargs.one_of == [field1, field2]


# LLM-generated content at query #10
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are accepted
    field_with_kwargs = NeverMatch(description="test field")
    assert isinstance(field_with_kwargs, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(Exception) as exc_info:
        field.validate("any_value")
    assert "never" in str(exc_info.value).lower()
    
    with pytest.raises(Exception) as exc_info:
        field.validate(None)
    assert "never" in str(exc_info.value).lower()
    
    with pytest.raises(Exception) as exc_info:
        field.validate(123)
    assert "never" in str(exc_info.value).lower()


# LLM-generated content at query #11
#--------------------------

```python
def test_AllOf():
    """Test AllOf field constructor."""
    from typesystem.fields import String, Integer
    
    # Test basic initialization with list of fields
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test with single field
    all_of_single = AllOf(all_of=[string_field])
    assert all_of_single.all_of == [string_field]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed in kwargs
    try:
        AllOf(all_of=[string_field], allow_null=True)
        assert False, "Should raise AssertionError"
    except AssertionError:
        pass
    
    # Test with additional kwargs (non-allow_null)
    all_of_with_kwargs = AllOf(all_of=[string_field], description="Test field")
    assert all_of_with_kwargs.all_of == [string_field]


# LLM-generated content at query #12
#--------------------------

```python
def test_Not():
    # Test basic instantiation
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test with kwargs
    not_field_with_kwargs = Not(negated=negated_field, description="test description")
    assert not_field_with_kwargs.negated is negated_field
    assert not_field_with_kwargs.description == "test description"
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        Not(negated=negated_field, allow_null=True)
    
    # Test that allow_null=False in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        Not(negated=negated_field, allow_null=False)


# LLM-generated content at query #13
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test that allow_null=False still raises assertion
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=False)
    
    # Test initialization with other kwargs
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="Test field")
    assert one_of_with_kwargs.one_of == [field1, field2]
    assert one_of_with_kwargs.description == "Test field"
    
    # Test that errors dictionary is present
    assert "no_match" in OneOf.errors
    assert "multiple_matches" in OneOf.errors
    assert OneOf.errors["no_match"] == "Did not match any valid type."
    assert OneOf.errors["multiple_matches"] == "Matched more than one type."


# LLM-generated content at query #14
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    single_field = Any()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test with additional kwargs
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="test description")
    assert one_of_with_kwargs.one_of == [field1, field2]
    assert one_of_with_kwargs.description == "test description"


# LLM-generated content at query #15
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        Not(negated=Field(), allow_null=True)
    
    # Test initialization with various field types
    from typesystem.fields import String, Integer
    
    not_string = Not(negated=String())
    assert isinstance(not_string.negated, String)
    
    not_integer = Not(negated=Integer())
    assert isinstance(not_integer.negated, Integer)
    
    # Test that other kwargs are passed to parent
    not_field_with_kwargs = Not(negated=Field(), description="test description")
    assert not_field_with_kwargs.negated is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    # Test that allow_null cannot be passed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are accepted
    field_with_kwargs = NeverMatch(description="test field")
    assert isinstance(field_with_kwargs, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(Exception):  # validation_error raises an exception
        field.validate("anything")
    
    with pytest.raises(Exception):
        field.validate(None)
    
    with pytest.raises(Exception):
        field.validate(123)
    
    with pytest.raises(Exception):
        field.validate([])


# LLM-generated content at query #17
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test with single field
    all_of_single = AllOf(all_of=[string_field])
    assert all_of_single.all_of == [string_field]
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test with kwargs
    all_of_with_kwargs = AllOf(all_of=[string_field, integer_field], description="test")
    assert all_of_with_kwargs.all_of == [string_field, integer_field]


# LLM-generated content at query #18
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are accepted
    field_with_kwargs = NeverMatch(description="test field")
    assert isinstance(field_with_kwargs, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(Exception) as exc_info:
        field.validate("any value")
    assert "never" in str(exc_info.value).lower()
    
    with pytest.raises(Exception) as exc_info:
        field.validate(None)
    assert "never" in str(exc_info.value).lower()
    
    with pytest.raises(Exception) as exc_info:
        field.validate(42)
    assert "never" in str(exc_info.value).lower()


# LLM-generated content at query #19
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization with a list of fields
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test initialization with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test initialization with single field
    all_of_single = AllOf(all_of=[string_field])
    assert all_of_single.all_of == [string_field]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test that other kwargs are accepted
    all_of_with_kwargs = AllOf(all_of=[string_field], description="test description")
    assert all_of_with_kwargs.all_of == [string_field]


# LLM-generated content at query #20
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch can be instantiated
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that NeverMatch raises AssertionError when allow_null is passed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch raises AssertionError when allow_null is False
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that NeverMatch can be instantiated with other valid kwargs
    field_with_kwargs = NeverMatch(description="test field")
    assert isinstance(field_with_kwargs, Field)
    
    # Test that the errors dict is properly set
    assert NeverMatch.errors == {"never": "This never validates."}


# LLM-generated content at query #21
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test with single field
    all_of_single = AllOf(all_of=[string_field])
    assert all_of_single.all_of == [string_field]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test with additional kwargs that are allowed
    all_of_with_kwargs = AllOf(all_of=[string_field, integer_field], default=None)
    assert all_of_with_kwargs.all_of == [string_field, integer_field]
    assert all_of_with_kwargs.default is None


# LLM-generated content at query #22
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with valid fields
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test that other kwargs are accepted
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="test")
    assert one_of_with_kwargs.one_of == [field1, field2]
    
    # Test initialization with multiple fields
    field3 = Field()
    field4 = Field()
    one_of_multiple = OneOf(one_of=[field1, field2, field3, field4])
    assert len(one_of_multiple.one_of) == 4
    assert one_of_multiple.one_of == [field1, field2, field3, field4]


# LLM-generated content at query #23
#--------------------------

```python
def test_Not():
    # Test basic instantiation
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        Not(negated=Any(), allow_null=True)
    
    # Test with different field types
    from typesystem.fields import String, Integer
    not_string = Not(negated=String())
    assert isinstance(not_string.negated, String)
    
    not_integer = Not(negated=Integer())
    assert isinstance(not_integer.negated, Integer)
    
    # Test that errors are properly defined
    assert "negated" in Not.errors
    assert Not.errors["negated"] == "Must not match."
    
    # Test with additional kwargs
    not_field_with_kwargs = Not(negated=Any(), description="test description")
    assert not_field_with_kwargs.negated is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_Not():
    # Test basic instantiation
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        Not(negated=Field(), allow_null=True)
    
    # Test with additional kwargs
    not_field_with_kwargs = Not(negated=Field(), description="test description")
    assert not_field_with_kwargs.negated is not None
    
    # Test that negated parameter is required
    with pytest.raises(TypeError):
        Not()


# LLM-generated content at query #25
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with a list of fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test that allow_null=False in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=False)
    
    # Test initialization with other kwargs
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="test")
    assert one_of_with_kwargs.one_of == [field1, field2]
    
    # Test that errors dict is properly defined
    assert "no_match" in OneOf.errors
    assert "multiple_matches" in OneOf.errors
    assert OneOf.errors["no_match"] == "Did not match any valid type."
    assert OneOf.errors["multiple_matches"] == "Matched more than one type."


# LLM-generated content at query #26
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that validate always raises validation error
    with pytest.raises(Exception) as exc_info:
        field.validate("any_value")
    assert "never" in str(exc_info.value).lower()
    
    # Test with different value types
    with pytest.raises(Exception):
        field.validate(None)
    
    with pytest.raises(Exception):
        field.validate(123)
    
    with pytest.raises(Exception):
        field.validate({})
    
    with pytest.raises(Exception):
        field.validate([])
    
    # Test that other kwargs work
    field_with_kwargs = NeverMatch(description="test field")
    assert field_with_kwargs is not None


# LLM-generated content at query #27
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with multiple fields
    field3 = Any()
    field4 = Any()
    field5 = Any()
    one_of_multiple = OneOf(one_of=[field1, field2, field3, field4, field5])
    assert len(one_of_multiple.one_of) == 5
    assert one_of_multiple.one_of == [field1, field2, field3, field4, field5]
    
    # Test that allow_null kwarg raises AssertionError
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test initialization with additional kwargs (non-allow_null)
    one_of_with_kwargs = OneOf(one_of=[field1], description="test description")
    assert one_of_with_kwargs.one_of == [field1]


# LLM-generated content at query #28
#--------------------------

```python
def test_IfThenElse():
    """Test IfThenElse constructor."""
    from typesystem.fields import String, Integer
    
    # Test basic construction with all parameters
    if_field = String()
    then_field = Integer()
    else_field = String()
    
    field = IfThenElse(if_clause=if_field, then_clause=then_field, else_clause=else_field)
    
    assert field.if_clause is if_field
    assert field.then_clause is then_field
    assert field.else_clause is else_field
    
    # Test construction with only if_clause (then and else should default to Any)
    field2 = IfThenElse(if_clause=if_field)
    
    assert field2.if_clause is if_field
    assert isinstance(field2.then_clause, Any)
    assert isinstance(field2.else_clause, Any)
    
    # Test construction with if_clause and then_clause (else should default to Any)
    field3 = IfThenElse(if_clause=if_field, then_clause=then_field)
    
    assert field3.if_clause is if_field
    assert field3.then_clause is then_field
    assert isinstance(field3.else_clause, Any)
    
    # Test construction with if_clause and else_clause (then should default to Any)
    field4 = IfThenElse(if_clause=if_field, else_clause=else_field)
    
    assert field4.if_clause is if_field
    assert isinstance(field4.then_clause, Any)
    assert field4.else_clause is else_field
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        IfThenElse(if_clause=if_field, allow_null=True)


# LLM-generated content at query #29
#--------------------------

```python
def test_Not():
    # Test basic instantiation
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        Not(negated=Field(), allow_null=True)
    
    # Test with additional kwargs
    not_field_with_kwargs = Not(negated=Field(), description="Test not field")
    assert not_field_with_kwargs.negated is not None
    
    # Test that negated parameter is required
    with pytest.raises(TypeError):
        Not()


# LLM-generated content at query #30
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that allow_null=False also raises assertion error
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs work fine
    field_with_kwargs = NeverMatch(description="test field")
    assert field_with_kwargs.description == "test field"
    
    # Test that validate always raises validation_error
    with pytest.raises(Exception):  # validation_error
        field.validate("any_value")
    
    with pytest.raises(Exception):  # validation_error
        field.validate(None)
    
    with pytest.raises(Exception):  # validation_error
        field.validate(123)
    
    # Test errors dictionary
    assert "never" in NeverMatch.errors
    assert NeverMatch.errors["never"] == "This never validates."


# LLM-generated content at query #31
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test that allow_null=False in kwargs also raises AssertionError
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=False)
    
    # Test initialization with other kwargs (non-allow_null)
    one_of_with_kwargs = OneOf(one_of=[field1, field2])
    assert one_of_with_kwargs.one_of == [field1, field2]
    
    # Test that one_of parameter is required
    with pytest.raises(TypeError):
        OneOf()


# LLM-generated content at query #32
#--------------------------

```python
def test_Not():
    # Test basic instantiation
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test with kwargs
    not_field_with_kwargs = Not(negated=negated_field, description="test description")
    assert not_field_with_kwargs.negated is negated_field
    assert not_field_with_kwargs.description == "test description"
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        Not(negated=negated_field, allow_null=True)
    
    # Test that allow_null=False in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        Not(negated=negated_field, allow_null=False)
    
    # Test errors attribute
    assert hasattr(not_field, 'errors')
    assert not_field.errors == {"negated": "Must not match."}


# LLM-generated content at query #33
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        Not(negated=Any(), allow_null=True)
    
    # Test with kwargs
    not_field_with_kwargs = Not(negated=Any(), description="Test not field")
    assert not_field_with_kwargs.negated is not None
    
    # Test error messages are set
    assert "negated" in Not.errors
    assert Not.errors["negated"] == "Must not match."


# LLM-generated content at query #34
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test with kwargs
    not_field_with_kwargs = Not(negated=negated_field, allow_null=False)
    assert not_field_with_kwargs.negated is negated_field
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        Not(negated=negated_field, allow_null=True)
    
    # Test error messages are set correctly
    assert "negated" in Not.errors
    assert Not.errors["negated"] == "Must not match."


# LLM-generated content at query #35
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test initialization with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test initialization with single field
    all_of_single = AllOf(all_of=[string_field])
    assert all_of_single.all_of == [string_field]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed
    try:
        AllOf(all_of=[string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with multiple fields
    all_of_multi = AllOf(all_of=[string_field, integer_field, String()])
    assert len(all_of_multi.all_of) == 3
    
    # Test with additional kwargs
    all_of_kwargs = AllOf(all_of=[string_field], description="Test description")
    assert all_of_kwargs.all_of == [string_field]


# LLM-generated content at query #36
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = Any()
    one_of_field = OneOf(one_of=[field1, field2])
    
    assert one_of_field.one_of == [field1, field2]
    assert len(one_of_field.one_of) == 2
    
    # Test initialization with single field
    single_field = Any()
    one_of_single = OneOf(one_of=[single_field])
    
    assert one_of_single.one_of == [single_field]
    assert len(one_of_single.one_of) == 1
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    
    assert one_of_empty.one_of == []
    assert len(one_of_empty.one_of) == 0
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        OneOf(one_of=[Any()], allow_null=True)
    
    # Test initialization with additional kwargs (non-allow_null)
    one_of_with_kwargs = OneOf(one_of=[Any()], description="test description")
    
    assert one_of_with_kwargs.one_of is not None
    assert hasattr(one_of_with_kwargs, 'description')


# LLM-generated content at query #37
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization with a list of fields
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test initialization with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test initialization with single field
    all_of_single = AllOf(all_of=[string_field])
    assert all_of_single.all_of == [string_field]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test initialization with additional kwargs (non-allow_null)
    all_of_with_kwargs = AllOf(all_of=[string_field], description="test description")
    assert all_of_with_kwargs.all_of == [string_field]
    
    # Test initialization with multiple fields
    fields_list = [String(), Integer(), String()]
    all_of_multiple = AllOf(all_of=fields_list)
    assert all_of_multiple.all_of == fields_list
    assert len(all_of_multiple.all_of) == 3


# LLM-generated content at query #38
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test with single field
    single_field = Field()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test with multiple fields
    field3 = Field()
    field4 = Field()
    field5 = Field()
    one_of_multi = OneOf(one_of=[field1, field2, field3, field4, field5])
    assert len(one_of_multi.one_of) == 5
    assert one_of_multi.one_of == [field1, field2, field3, field4, field5]
    
    # Test that other kwargs are passed to parent
    one_of_with_kwargs = OneOf(one_of=[field1], description="test description")
    assert one_of_with_kwargs.one_of == [field1]


# LLM-generated content at query #39
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that allow_null=False raises AssertionError
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are allowed
    field = NeverMatch(description="test field")
    assert field.description == "test field"
    
    # Test multiple valid kwargs
    field = NeverMatch(description="test", title="Test Field")
    assert field.description == "test"
    assert field.title == "Test Field"


# LLM-generated content at query #40
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with single field
    single_field = Any()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        OneOf(one_of=[Any()], allow_null=True)
    
    # Test that other kwargs are accepted
    one_of_with_kwargs = OneOf(one_of=[Any()], description="test field")
    assert one_of_with_kwargs.one_of == [Any()]
    
    # Test errors attribute is set correctly
    assert "no_match" in OneOf.errors
    assert "multiple_matches" in OneOf.errors
    assert OneOf.errors["no_match"] == "Did not match any valid type."
    assert OneOf.errors["multiple_matches"] == "Matched more than one type."


####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Not():
    # Test basic instantiation
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        Not(negated=Field(), allow_null=True)
    
    # Test with different field types
    from typesystem.fields import String, Integer
    not_string = Not(negated=String())
    assert isinstance(not_string.negated, String)
    
    not_integer = Not(negated=Integer())
    assert isinstance(not_integer.negated, Integer)
    
    # Test that other kwargs are accepted
    not_field_with_kwargs = Not(negated=Field(), description="test description")
    assert not_field_with_kwargs.negated is not None
    
    # Test multiple instantiations are independent
    field1 = Not(negated=String())
    field2 = Not(negated=Integer())
    assert not isinstance(field1.negated, Integer)
    assert not isinstance(field2.negated, String)


# LLM-generated content at query #2
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that allow_null=False is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are allowed
    field_with_kwargs = NeverMatch(description="test field")
    assert isinstance(field_with_kwargs, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(Exception) as exc_info:
        field.validate("any_value")
    assert "never" in str(exc_info.value).lower()
    
    # Test with different value types
    with pytest.raises(Exception):
        field.validate(None)
    
    with pytest.raises(Exception):
        field.validate(123)
    
    with pytest.raises(Exception):
        field.validate([])
    
    with pytest.raises(Exception):
        field.validate({})


# LLM-generated content at query #3
#--------------------------

```python
def test_AllOf():
    """Test AllOf field constructor."""
    from typesystem.fields import String, Integer
    
    # Test basic initialization with multiple fields
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test initialization with single field
    single_field = String()
    all_of_single = AllOf(all_of=[single_field])
    
    assert all_of_single.all_of == [single_field]
    assert len(all_of_single.all_of) == 1
    
    # Test initialization with empty list
    all_of_empty = AllOf(all_of=[])
    
    assert all_of_empty.all_of == []
    assert len(all_of_empty.all_of) == 0
    
    # Test that allow_null is not allowed in kwargs
    try:
        AllOf(all_of=[string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test initialization with additional kwargs (other than allow_null)
    all_of_with_kwargs = AllOf(all_of=[string_field], description="Test description")
    
    assert all_of_with_kwargs.all_of == [string_field]


# LLM-generated content at query #4
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        Not(negated=Field(), allow_null=True)
    
    # Test with different field types
    from typesystem.fields import String, Integer
    not_string = Not(negated=String())
    assert isinstance(not_string.negated, String)
    
    not_integer = Not(negated=Integer())
    assert isinstance(not_integer.negated, Integer)
    
    # Test that other kwargs are passed to parent
    not_field_with_description = Not(negated=Field(), description="test description")
    assert not_field_with_description.description == "test description"
    
    # Test error message exists
    assert "negated" in Not.errors
    assert Not.errors["negated"] == "Must not match."


# LLM-generated content at query #5
#--------------------------

```python
def test_AllOf():
    """Test AllOf constructor"""
    from typesystem.fields import String, Integer
    
    # Test basic construction with valid fields
    field1 = String()
    field2 = Integer()
    all_of = AllOf(all_of=[field1, field2])
    
    assert all_of.all_of == [field1, field2]
    assert len(all_of.all_of) == 2
    
    # Test construction with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test construction with single field
    all_of_single = AllOf(all_of=[field1])
    assert all_of_single.all_of == [field1]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        AllOf(all_of=[field1], allow_null=True)
    
    # Test construction with multiple fields
    field3 = String()
    all_of_multi = AllOf(all_of=[field1, field2, field3])
    assert len(all_of_multi.all_of) == 3
    
    # Test that allow_null=False doesn't raise
    all_of_no_null = AllOf(all_of=[field1], allow_null=False)
    assert all_of_no_null.all_of == [field1]


# LLM-generated content at query #6
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that allow_null=False raises AssertionError
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are accepted
    field_with_kwargs = NeverMatch(description="test field")
    assert field_with_kwargs is not None
    
    # Test validation always fails
    with pytest.raises(Exception):  # validation_error
        field.validate("anything")
    
    with pytest.raises(Exception):
        field.validate(None)
    
    with pytest.raises(Exception):
        field.validate(123)
    
    with pytest.raises(Exception):
        field.validate({})


# LLM-generated content at query #7
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test that other kwargs are accepted
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="test description")
    assert one_of_with_kwargs.one_of == [field1, field2]
    
    # Test that errors dict is properly set
    assert "no_match" in OneOf.errors
    assert "multiple_matches" in OneOf.errors
    assert OneOf.errors["no_match"] == "Did not match any valid type."
    assert OneOf.errors["multiple_matches"] == "Matched more than one type."


# LLM-generated content at query #8
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with a list of fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    single_field = Any()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test that other kwargs are accepted
    one_of_with_kwargs = OneOf(one_of=[field1, field2])
    assert hasattr(one_of_with_kwargs, 'one_of')
    
    # Test errors dictionary is properly set
    assert "no_match" in OneOf.errors
    assert "multiple_matches" in OneOf.errors
    assert OneOf.errors["no_match"] == "Did not match any valid type."
    assert OneOf.errors["multiple_matches"] == "Matched more than one type."


# LLM-generated content at query #9
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with valid fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test with multiple fields of different types
    from typesystem.fields import String, Integer
    one_of_mixed = OneOf(one_of=[String(), Integer(), Any()])
    assert len(one_of_mixed.one_of) == 3
    
    # Test that other kwargs are passed to parent
    one_of_with_kwargs = OneOf(one_of=[field1], description="test description")
    assert one_of_with_kwargs.one_of == [field1]


# LLM-generated content at query #10
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are accepted
    field_with_kwargs = NeverMatch(description="test")
    assert field_with_kwargs.description == "test"
    
    # Test that validate always raises validation error
    with pytest.raises(Exception):  # validation_error raises an exception
        field.validate("any_value")
    
    with pytest.raises(Exception):
        field.validate(None)
    
    with pytest.raises(Exception):
        field.validate(123)
    
    # Test error message
    assert "never" in field.errors
    assert field.errors["never"] == "This never validates."


# LLM-generated content at query #11
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert field is not None
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are accepted
    field_with_description = NeverMatch(description="test description")
    assert field_with_description is not None
    
    # Test that validate always raises validation_error
    with pytest.raises(Exception):  # validation_error
        field.validate("any value")
    
    with pytest.raises(Exception):  # validation_error
        field.validate(None)
    
    with pytest.raises(Exception):  # validation_error
        field.validate(123)
    
    # Test that errors dict is properly set
    assert "never" in NeverMatch.errors
    assert NeverMatch.errors["never"] == "This never validates."


# LLM-generated content at query #12
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with valid fields
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test that other kwargs are accepted
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="test description")
    assert one_of_with_kwargs.one_of == [field1, field2]
    
    # Test that allow_null=False raises AssertionError
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=False)


# LLM-generated content at query #13
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with single field
    single_field = Field()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test with multiple fields
    fields = [Field() for _ in range(5)]
    one_of_multiple = OneOf(one_of=fields)
    assert one_of_multiple.one_of == fields
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        OneOf(one_of=[Field()], allow_null=True)
    
    # Test with additional kwargs (other than allow_null)
    one_of_with_kwargs = OneOf(one_of=[Field()], description="test")
    assert one_of_with_kwargs.one_of
    
    # Test that one_of parameter is stored correctly
    test_fields = [Field(), Field(), Field()]
    one_of_test = OneOf(one_of=test_fields)
    assert len(one_of_test.one_of) == 3
    assert all(isinstance(f, Field) for f in one_of_test.one_of)


# LLM-generated content at query #14
#--------------------------

```python
def test_Not():
    # Test basic instantiation
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test with keyword arguments
    not_field_with_kwargs = Not(negated=negated_field, description="Test description")
    assert not_field_with_kwargs.negated is negated_field
    assert not_field_with_kwargs.description == "Test description"
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        Not(negated=negated_field, allow_null=True)
    
    # Test that allow_null=False is also not allowed
    with pytest.raises(AssertionError):
        Not(negated=negated_field, allow_null=False)
    
    # Test with different field types
    string_field = Field()
    not_string = Not(negated=string_field)
    assert not_string.negated is string_field
    
    # Test error messages are set correctly
    assert "negated" in Not.errors
    assert Not.errors["negated"] == "Must not match."


# LLM-generated content at query #15
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are accepted
    field_with_kwargs = NeverMatch(description="test field")
    assert field_with_kwargs is not None
    
    # Test that validate always raises validation_error
    with pytest.raises(Exception):  # validation_error
        field.validate("any_value")
    
    with pytest.raises(Exception):  # validation_error
        field.validate(None)
    
    with pytest.raises(Exception):  # validation_error
        field.validate(123)
    
    # Test errors attribute
    assert "never" in NeverMatch.errors
    assert NeverMatch.errors["never"] == "This never validates."


# LLM-generated content at query #16
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with a list of fields
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    
    assert one_of.one_of == [field1, field2]
    assert len(one_of.one_of) == 2

    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []

    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    assert len(one_of_single.one_of) == 1

    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)

    # Test that other kwargs are passed to parent
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="test")
    assert one_of_with_kwargs.one_of == [field1, field2]


# LLM-generated content at query #17
#--------------------------

```python
def test_OneOf():
    """Test OneOf field constructor."""
    # Test basic instantiation with list of fields
    field1 = Field()
    field2 = Field()
    one_of_field = OneOf(one_of=[field1, field2])
    
    assert one_of_field.one_of == [field1, field2]
    assert len(one_of_field.one_of) == 2
    
    # Test with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    assert len(one_of_single.one_of) == 1
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test that allow_null kwarg raises AssertionError
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test with other kwargs (should not raise)
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="test")
    assert one_of_with_kwargs.one_of == [field1, field2]
    
    # Test errors dictionary exists
    assert "no_match" in OneOf.errors
    assert "multiple_matches" in OneOf.errors
    assert OneOf.errors["no_match"] == "Did not match any valid type."
    assert OneOf.errors["multiple_matches"] == "Matched more than one type."


# LLM-generated content at query #18
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with multiple fields
    field3 = Any()
    field4 = Any()
    field5 = Any()
    one_of_multi = OneOf(one_of=[field1, field2, field3, field4, field5])
    assert len(one_of_multi.one_of) == 5
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1], allow_null=True)
    
    # Test that other kwargs are accepted
    one_of_with_kwargs = OneOf(one_of=[field1], description="test description")
    assert one_of_with_kwargs.one_of == [field1]
    
    # Test initialization with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    assert len(one_of_single.one_of) == 1


# LLM-generated content at query #19
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with single field
    one_of_single = OneOf(one_of=[field1])
    assert one_of_single.one_of == [field1]
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        OneOf(one_of=[field1, field2], allow_null=True)
    
    # Test with additional kwargs
    one_of_with_kwargs = OneOf(one_of=[field1, field2], description="test description")
    assert one_of_with_kwargs.one_of == [field1, field2]


# LLM-generated content at query #20
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Field()
    field2 = Field()
    all_of = AllOf(all_of=[field1, field2])
    assert all_of.all_of == [field1, field2]

    # Test with single field
    single_field = Field()
    all_of_single = AllOf(all_of=[single_field])
    assert all_of_single.all_of == [single_field]

    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []

    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        AllOf(all_of=[field1], allow_null=True)

    # Test with additional kwargs (other than allow_null)
    all_of_with_kwargs = AllOf(all_of=[field1, field2])
    assert all_of_with_kwargs.all_of == [field1, field2]

    # Test with multiple fields
    fields = [Field() for _ in range(5)]
    all_of_multiple = AllOf(all_of=fields)
    assert all_of_multiple.all_of == fields
    assert len(all_of_multiple.all_of) == 5


# LLM-generated content at query #21
#--------------------------

```python
def test_IfThenElse():
    # Test basic initialization with all parameters
    if_field = Field()
    then_field = Field()
    else_field = Field()
    
    ite = IfThenElse(if_clause=if_field, then_clause=then_field, else_clause=else_field)
    
    assert ite.if_clause is if_field
    assert ite.then_clause is then_field
    assert ite.else_clause is else_field
    
    # Test initialization with only if_clause
    ite2 = IfThenElse(if_clause=if_field)
    
    assert ite2.if_clause is if_field
    assert isinstance(ite2.then_clause, Any)
    assert isinstance(ite2.else_clause, Any)
    
    # Test initialization with if_clause and then_clause
    ite3 = IfThenElse(if_clause=if_field, then_clause=then_field)
    
    assert ite3.if_clause is if_field
    assert ite3.then_clause is then_field
    assert isinstance(ite3.else_clause, Any)
    
    # Test initialization with if_clause and else_clause
    ite4 = IfThenElse(if_clause=if_field, else_clause=else_field)
    
    assert ite4.if_clause is if_field
    assert isinstance(ite4.then_clause, Any)
    assert ite4.else_clause is else_field
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        IfThenElse(if_clause=if_field, allow_null=True)
    
    # Test initialization with other kwargs
    ite5 = IfThenElse(if_clause=if_field, description="test description")
    
    assert ite5.if_clause is if_field
    assert isinstance(ite5.then_clause, Any)
    assert isinstance(ite5.else_clause, Any)


# LLM-generated content at query #22
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with a list of fields
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test initialization with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test initialization with single field
    single_field = Field()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test initialization with multiple fields
    fields = [Field(), Field(), Field()]
    one_of_multiple = OneOf(one_of=fields)
    assert one_of_multiple.one_of == fields
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        OneOf(one_of=[Field()], allow_null=True)
    
    # Test that allow_null=False in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        OneOf(one_of=[Field()], allow_null=False)
    
    # Test initialization with other valid kwargs
    one_of_with_kwargs = OneOf(one_of=[Field()], description="test")
    assert one_of_with_kwargs.one_of == [Field()]


# LLM-generated content at query #23
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    field1 = String()
    field2 = Integer()
    all_of = AllOf(all_of=[field1, field2])
    
    assert all_of.all_of == [field1, field2]
    assert len(all_of.all_of) == 2
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test with single field
    all_of_single = AllOf(all_of=[field1])
    assert all_of_single.all_of == [field1]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        AllOf(all_of=[field1], allow_null=True)
    
    # Test with kwargs (other than allow_null)
    all_of_kwargs = AllOf(all_of=[field1, field2], description="test")
    assert all_of_kwargs.all_of == [field1, field2]


# LLM-generated content at query #24
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    # Test that other kwargs are accepted
    field_with_kwargs = NeverMatch(description="test field")
    assert isinstance(field_with_kwargs, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(Exception):  # validation_error
        field.validate("anything")
    
    with pytest.raises(Exception):  # validation_error
        field.validate(None)
    
    with pytest.raises(Exception):  # validation_error
        field.validate(123)
    
    # Test error message
    assert field.errors["never"] == "This never validates."


# LLM-generated content at query #25
#--------------------------

```python
def test_NeverMatch():
    # Test basic instantiation
    field = NeverMatch()
    assert isinstance(field, Field)
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that validation always fails
    with pytest.raises(Exception):
        field.validate("any_value")
    
    with pytest.raises(Exception):
        field.validate(None)
    
    with pytest.raises(Exception):
        field.validate(123)
    
    with pytest.raises(Exception):
        field.validate([])
    
    # Test error message
    assert field.errors["never"] == "This never validates."


# LLM-generated content at query #26
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test with single field
    single_field = String()
    all_of_single = AllOf(all_of=[single_field])
    
    assert all_of_single.all_of == [single_field]
    assert len(all_of_single.all_of) == 1
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    
    assert all_of_empty.all_of == []
    assert len(all_of_empty.all_of) == 0
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test with additional kwargs
    all_of_with_kwargs = AllOf(all_of=[string_field], description="Test description")
    
    assert all_of_with_kwargs.all_of == [string_field]
    assert all_of_with_kwargs.description == "Test description"


# LLM-generated content at query #27
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic construction with list of fields
    field1 = String()
    field2 = Integer()
    all_of = AllOf(all_of=[field1, field2])
    
    assert all_of.all_of == [field1, field2]
    assert len(all_of.all_of) == 2
    
    # Test construction with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test construction with single field
    all_of_single = AllOf(all_of=[field1])
    assert all_of_single.all_of == [field1]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        AllOf(all_of=[field1], allow_null=True)
    
    # Test with additional valid kwargs
    all_of_with_kwargs = AllOf(all_of=[field1, field2], description="test description")
    assert all_of_with_kwargs.all_of == [field1, field2]
    assert all_of_with_kwargs.description == "test description"


# LLM-generated content at query #28
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic construction with list of fields
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test construction with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test construction with single field
    all_of_single = AllOf(all_of=[string_field])
    assert all_of_single.all_of == [string_field]
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test construction with additional kwargs (non-allow_null)
    all_of_with_kwargs = AllOf(all_of=[string_field], description="test description")
    assert all_of_with_kwargs.all_of == [string_field]
    
    # Test that it's a Field instance
    assert isinstance(all_of, Field)


# LLM-generated content at query #29
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test initialization with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test initialization with single field
    all_of_single = AllOf(all_of=[string_field])
    assert len(all_of_single.all_of) == 1
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test that other kwargs are accepted
    all_of_with_kwargs = AllOf(all_of=[string_field], description="test description")
    assert all_of_with_kwargs.all_of == [string_field]


# LLM-generated content at query #30
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Field()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        Not(negated=Field(), allow_null=True)
    
    # Test with various field types
    from typesystem.fields import String, Integer
    not_string = Not(negated=String())
    assert isinstance(not_string.negated, String)
    
    not_integer = Not(negated=Integer())
    assert isinstance(not_integer.negated, Integer)
    
    # Test that errors dictionary is properly set
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test with additional kwargs (excluding allow_null)
    not_field_with_kwargs = Not(negated=Field(), description="test description")
    assert not_field_with_kwargs.negated is not None


# LLM-generated content at query #31
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic instantiation
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2
    
    # Test with single field
    single_field = String()
    all_of_single = AllOf(all_of=[single_field])
    assert all_of_single.all_of == [single_field]
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test with additional kwargs
    all_of_with_kwargs = AllOf(all_of=[string_field, integer_field], description="Test field")
    assert all_of_with_kwargs.all_of == [string_field, integer_field]


# LLM-generated content at query #32
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field
    
    # Test that allow_null is not allowed in kwargs
    with pytest.raises(AssertionError):
        Not(negated=Any(), allow_null=True)
    
    # Test with different field types
    from typesystem.fields import Integer, String
    
    int_field = Integer()
    not_int = Not(negated=int_field)
    assert not_int.negated is int_field
    
    string_field = String()
    not_string = Not(negated=string_field)
    assert not_string.negated is string_field
    
    # Test that errors dictionary is properly set
    assert "negated" in Not.errors
    assert Not.errors["negated"] == "Must not match."
    
    # Test with additional kwargs (other than allow_null)
    not_field_with_kwargs = Not(negated=Any())
    assert not_field_with_kwargs.negated is not None


# LLM-generated content at query #33
#--------------------------

```python
def test_IfThenElse():
    # Test basic initialization with all parameters
    if_field = Any()
    then_field = Any()
    else_field = Any()
    
    field = IfThenElse(if_clause=if_field, then_clause=then_field, else_clause=else_field)
    
    assert field.if_clause is if_field
    assert field.then_clause is then_field
    assert field.else_clause is else_field
    
    # Test initialization with only if_clause (then and else default to Any())
    field2 = IfThenElse(if_clause=if_field)
    
    assert field2.if_clause is if_field
    assert isinstance(field2.then_clause, Any)
    assert isinstance(field2.else_clause, Any)
    
    # Test initialization with if_clause and then_clause (else defaults to Any())
    field3 = IfThenElse(if_clause=if_field, then_clause=then_field)
    
    assert field3.if_clause is if_field
    assert field3.then_clause is then_field
    assert isinstance(field3.else_clause, Any)
    
    # Test initialization with if_clause and else_clause (then defaults to Any())
    field4 = IfThenElse(if_clause=if_field, else_clause=else_field)
    
    assert field4.if_clause is if_field
    assert isinstance(field4.then_clause, Any)
    assert field4.else_clause is else_field
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        IfThenElse(if_clause=if_field, allow_null=True)
    
    # Test that other kwargs are accepted
    field5 = IfThenElse(if_clause=if_field, description="test description")
    assert field5.if_clause is if_field


# LLM-generated content at query #34
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf(all_of=[string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test with single field
    all_of_single = AllOf(all_of=[string_field])
    assert all_of_single.all_of == [string_field]
    
    # Test that allow_null is not allowed
    with pytest.raises(AssertionError):
        AllOf(all_of=[string_field], allow_null=True)
    
    # Test with additional kwargs
    all_of_with_kwargs = AllOf(all_of=[string_field, integer_field], description="Test field")
    assert all_of_with_kwargs.all_of == [string_field, integer_field]
    assert all_of_with_kwargs.description == "Test field"


# LLM-generated content at query #35
#--------------------------

```python
def test_IfThenElse():
    # Test basic initialization with all parameters
    if_field = Field()
    then_field = Field()
    else_field = Field()
    
    if_then_else = IfThenElse(if_clause=if_field, then_clause=then_field, else_clause=else_field)
    
    assert if_then_else.if_clause is if_field
    assert if_then_else.then_clause is then_field
    assert if_then_else.else_clause is else_field
    
    # Test initialization with only if_clause
    if_then_else_minimal = IfThenElse(if_clause=if_field)
    
    assert if_then_else_minimal.if_clause is if_field
    assert isinstance(if_then_else_minimal.then_clause, Any)
    assert isinstance(if_then_else_minimal.else_clause, Any)
    
    # Test initialization with if_clause and then_clause
    if_then_else_partial = IfThenElse(if_clause=if_field, then_clause=then_field)
    
    assert if_then_else_partial.if_clause is if_field
    assert if_then_else_partial.then_clause is then_field
    assert isinstance(if_then_else_partial.else_clause, Any)
    
    # Test initialization with if_clause and else_clause
    if_then_else_else = IfThenElse(if_clause=if_field, else_clause=else_field)
    
    assert if_then_else_else.if_clause is if_field
    assert isinstance(if_then_else_else.then_clause, Any)
    assert if_then_else_else.else_clause is else_field
    
    # Test that allow_null in kwargs raises AssertionError
    with pytest.raises(AssertionError):
        IfThenElse(if_clause=if_field, allow_null=True)
    
    # Test that extra kwargs are passed to parent
    if_then_else_with_kwargs = IfThenElse(if_clause=if_field, description="test")
    assert if_then_else_with_kwargs.if_clause is if_field


