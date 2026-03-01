####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Form_validate():
    # Mock schema with a simple field
    class MockSchema(Schema):
        name = String(max_length=10, required=True)
        age = Integer(minimum=0, maximum=150)
    
    # Mock jinja2 environment
    class MockTemplate:
        def render(self, context):
            return f"rendered: {context}"
    
    class MockEnv:
        def get_template(self, name):
            return MockTemplate()
    
    # Test 1: Valid data
    schema = MockSchema()
    env = MockEnv()
    form = Form(env=env, schema=schema, values=None)
    
    valid_data = {"name": "John", "age": 25}
    form.validate(valid_data)
    
    assert form.is_valid is True
    assert form.errors is None
    assert form.validated_data == {"name": "John", "age": 25}
    assert form._validate_called is True
    
    # Test 2: Invalid data
    form2 = Form(env=env, schema=schema, values=None)
    invalid_data = {"name": "VeryLongNameExceedsLimit", "age": 200}
    form2.validate(invalid_data)
    
    assert form2.is_valid is False
    assert form2.errors is not None
    assert "name" in form2.errors
    assert "age" in form2.errors
    assert form2._validate_called is True
    
    # Test 3: Empty data for required field
    form3 = Form(env=env, schema=schema, values=None)
    empty_data = {"age": 25}  # missing required 'name'
    form3.validate(empty_data)
    
    assert form3.is_valid is False
    assert form3.errors is not None
    assert "name" in form3.errors
    
    # Test 4: validate() called twice should raise AssertionError
    form4 = Form(env=env, schema=schema, values=None)
    form4.validate({"name": "Test", "age": 30})
    
    try:
        form4.validate({"name": "Another", "age": 40})
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert str(e) == "validate() has already been called."
    
    # Test 5: None data
    form5 = Form(env=env, schema=schema, values=None)
    form5.validate(None)
    
    assert form5.is_valid is False
    assert form5.errors is not None
    
    # Test 6: Initial values preserved when no validation errors
    form6 = Form(env=env, schema=schema, values={"name": "Initial", "age": 99})
    form6.validate({"name": "Updated", "age": 100})
    
    assert form6.is_valid is True
    assert form6.validated_data == {"name": "Updated", "age": 100}
    
    # Test 7: Check that data attribute is set
    form7 = Form(env=env, schema=schema, values=None)
    test_data = {"name": "Test", "age": 42}
    form7.validate(test_data)
    
    assert hasattr(form7, 'data')
    assert form7.data == test_data


# LLM-generated content at query #2
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms1 = Jinja2Forms(directory="/test/templates")
    env1 = forms1.env
    assert isinstance(env1, jinja2.Environment)
    assert isinstance(env1.loader, jinja2.FileSystemLoader)
    assert env1.loader.searchpath == ["/test/templates"]
    assert env1.autoescape is True

    # Test with package only
    forms2 = Jinja2Forms(package="test_package")
    env2 = forms2.env
    assert isinstance(env2, jinja2.Environment)
    assert isinstance(env2.loader, jinja2.PackageLoader)
    assert env2.loader.package_name == "test_package"
    assert env2.loader.package_path == "templates"
    assert env2.autoescape is True

    # Test with both directory and package
    forms3 = Jinja2Forms(directory="/test/templates", package="test_package")
    env3 = forms3.env
    assert isinstance(env3, jinja2.Environment)
    assert isinstance(env3.loader, jinja2.ChoiceLoader)
    assert len(env3.loader.loaders) == 2
    assert isinstance(env3.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env3.loader.loaders[1], jinja2.PackageLoader)
    assert env3.autoescape is True


# LLM-generated content at query #3
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user"])
        bio = fields.String(format="text")
        read_only_field = fields.String(read_only=True)
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>',
        })
    )
    
    form = Form(env=env, schema=TestSchema, values={
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "active": True,
        "role": "admin",
        "bio": "Software developer",
        "read_only_field": "hidden"
    })
    
    result = str(form)
    
    assert '<input type="text" name="name" value="John Doe">' in result
    assert '<input type="email" name="email" value="john@example.com">' in result
    assert '<input type="text" name="age" value="30">' in result
    assert '<input type="checkbox" name="active" checked>' in result
    assert '<select name="role"><option value="admin">admin</option></select>' in result
    assert '<textarea name="bio">Software developer</textarea>' in result
    assert 'read_only_field' not in result
    
    form2 = Form(env=env, schema=TestSchema, values=None)
    result2 = str(form2)
    
    assert '<input type="text" name="name" value="">' in result2
    assert '<input type="email" name="email" value="">' in result2
    assert '<input type="text" name="age" value="">' in result2
    assert '<input type="checkbox" name="active">' in result2
    assert '<select name="role"><option value=""></option></select>' in result2
    assert '<textarea name="bio"></textarea>' in result2


# LLM-generated content at query #4
#--------------------------

```python
def test_Form_render_fields():
    # Mock jinja2 environment and templates
    class MockTemplate:
        def render(self, context):
            field_name = context["field_name"]
            field_id = context["field_id"]
            input_type = context["input_type"]
            value = context["value"]
            error = context["error"]
            required = context["required"]
            
            if error:
                return f'<div class="error">{field_name}: {error}</div>'
            return f'<input id="{field_id}" name="{field_name}" type="{input_type}" value="{value}" required="{required}">'

    class MockEnv:
        def get_template(self, template_name):
            return MockTemplate()

    # Mock schema with fields
    class MockField:
        def __init__(self, read_only=False, title=None, allow_null=False, allow_blank=False, has_default=False, format=None):
            self.read_only = read_only
            self.title = title
            self.allow_null = allow_null
            self.allow_blank = allow_blank
            self._has_default = has_default
            self.format = format

        def has_default(self):
            return self._has_default

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields

        def serialize(self, values):
            return values

        def validate_or_error(self, data):
            if data == {"field1": "invalid"}:
                return None, {"field1": "Invalid value"}
            return data, None

    # Test 1: Render fields with values and no errors
    fields = {
        "field1": MockField(title="Field One", allow_null=False, has_default=False),
        "field2": MockField(read_only=True),
        "field3": MockField(title="Field Three", allow_null=True, has_default=True),
    }
    schema = MockSchema(fields)
    env = MockEnv()
    form = Form(env=env, schema=schema, values={"field1": "value1", "field3": "value3"})
    
    # Call validate to set data and errors
    form.validate({"field1": "value1", "field3": "value3"})
    
    result = form.render_fields()
    assert '<input id="field1" name="field1" type="text" value="value1" required="True">' in result
    assert 'field2' not in result  # read_only field should not be rendered
    assert '<input id="field3" name="field3" type="text" value="value3" required="False">' in result

    # Test 2: Render fields with errors
    form2 = Form(env=env, schema=schema, values={"field1": "initial"})
    form2.validate({"field1": "invalid"})
    
    result2 = form2.render_fields()
    assert '<div class="error">field1: Invalid value</div>' in result2

    # Test 3: Render fields with no values (fresh form)
    form3 = Form(env=env, schema=schema)
    form3.validate({})
    
    result3 = form3.render_fields()
    assert 'value=""' in result3  # Empty value for field1
    assert 'required="True"' in result3  # field1 is required

    # Test 4: Test with password field (should have empty value)
    password_field = MockField(format="password")
    password_schema = MockSchema({"password": password_field})
    form4 = Form(env=env, schema=password_schema, values={"password": "secret"})
    form4.validate({"password": "secret"})
    
    result4 = form4.render_fields()
    assert 'value=""' in result4  # Password should always render empty value

    # Test 5: Test with different input types based on format
    email_field = MockField(format="email")
    number_field = MockField(format="number")
    mixed_schema = MockSchema({
        "email": email_field,
        "number": number_field,
    })
    form5 = Form(env=env, schema=mixed_schema, values={"email": "test@example.com", "number": 42})
    form5.validate({"email": "test@example.com", "number": 42})
    
    result5 = form5.render_fields()
    assert 'type="email"' in result5
    assert 'type="number"' in result5


# LLM-generated content at query #5
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory parameter
    forms1 = Jinja2Forms(directory="/test/templates")
    assert forms1.env is not None
    assert isinstance(forms1.env.loader, jinja2.FileSystemLoader)
    assert forms1.env.autoescape == True

    # Test with package parameter
    forms2 = Jinja2Forms(package="test_package")
    assert forms2.env is not None
    assert isinstance(forms2.env.loader, jinja2.PackageLoader)
    assert forms2.env.autoescape == True

    # Test with both directory and package parameters
    forms3 = Jinja2Forms(directory="/test/templates", package="test_package")
    assert forms3.env is not None
    assert isinstance(forms3.env.loader, jinja2.ChoiceLoader)
    assert forms3.env.autoescape == True

    # Test that create_form method returns Form instance
    from typesystem.schemas import Schema
    from typesystem.fields import String

    class TestSchema(Schema):
        name = String()

    form = forms1.create_form(TestSchema())
    assert isinstance(form, Form)
    assert form.env == forms1.env
    assert isinstance(form.schema, TestSchema)


# LLM-generated content at query #6
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user"])
        bio = fields.String(format="text")
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
                <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">
            """,
            "forms/checkbox.html": """
                <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>
            """,
            "forms/select.html": """
                <select name="{{ field_name }}" id="{{ field_id }}">
                    <option value="admin" {% if value == "admin" %}selected{% endif %}>admin</option>
                    <option value="user" {% if value == "user" %}selected{% endif %}>user</option>
                </select>
            """,
            "forms/textarea.html": """
                <textarea name="{{ field_name }}" id="{{ field_id }}">{{ value }}</textarea>
            """,
        })
    )
    
    form = Form(env=env, schema=TestSchema, values={
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "active": True,
        "role": "admin",
        "bio": "Software developer"
    })
    
    result = str(form)
    
    assert '<input type="text" name="name" id="name" value="John Doe">' in result
    assert '<input type="email" name="email" id="email" value="john@example.com">' in result
    assert '<input type="text" name="age" id="age" value="30">' in result
    assert '<input type="checkbox" name="active" id="active" checked>' in result
    assert '<option value="admin" selected' in result
    assert '<textarea name="bio" id="bio">Software developer</textarea>' in result
    
    form2 = Form(env=env, schema=TestSchema, values=None)
    result2 = str(form2)
    
    assert 'value=""' in result2
    assert 'checked' not in result2
    assert 'selected' not in result2
    
    class ReadOnlySchema(Schema):
        name = fields.String()
        id = fields.String(read_only=True)
    
    form3 = Form(env=env, schema=ReadOnlySchema, values={"name": "Test", "id": "123"})
    result3 = str(form3)
    
    assert 'name="name"' in result3
    assert 'name="id"' not in result3


# LLM-generated content at query #7
#--------------------------

```python
def test_Form_validate():
    import typesystem
    from unittest.mock import Mock, MagicMock

    # Create a mock jinja2 environment
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render = Mock(return_value="<input>")
    mock_env.get_template = Mock(return_value=mock_template)

    # Test 1: Basic validation with valid data
    class TestSchema(typesystem.Schema):
        name = typesystem.String(max_length=100)
        age = typesystem.Integer(minimum=0)

    schema = TestSchema()
    form = Form(env=mock_env, schema=schema, values=None)
    
    # Initial state checks
    assert form._validate_called == False
    assert form.errors is None
    
    # Call validate with valid data
    valid_data = {"name": "John", "age": 25}
    form.validate(valid_data)
    
    assert form._validate_called == True
    assert form.is_valid == True
    assert form.errors is None
    assert form.validated_data == {"name": "John", "age": 25}
    assert form.data == valid_data

    # Test 2: Validation with invalid data
    form2 = Form(env=mock_env, schema=schema, values=None)
    invalid_data = {"name": "A" * 101, "age": -5}  # Name too long, age negative
    form2.validate(invalid_data)
    
    assert form2._validate_called == True
    assert form2.is_valid == False
    assert form2.errors is not None
    assert "name" in form2.errors
    assert "age" in form2.errors

    # Test 3: Validation with None data
    form3 = Form(env=mock_env, schema=schema, values=None)
    form3.validate(None)
    
    assert form3._validate_called == True
    assert form3.is_valid == False
    assert form3.errors is not None

    # Test 4: Validation with initial values and valid data
    initial_values = {"name": "Initial", "age": 30}
    form4 = Form(env=mock_env, schema=schema, values=initial_values)
    
    # Validate with different data
    new_data = {"name": "Updated", "age": 35}
    form4.validate(new_data)
    
    assert form4._validate_called == True
    assert form4.is_valid == True
    assert form4.validated_data == new_data

    # Test 5: Ensure validate() can only be called once
    form5 = Form(env=mock_env, schema=schema, values=None)
    form5.validate({"name": "Test", "age": 20})
    
    # Try to call validate again - should raise AssertionError
    try:
        form5.validate({"name": "Another", "age": 25})
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert str(e) == "validate() has already been called."

    # Test 6: Validation with read_only field
    class SchemaWithReadOnly(typesystem.Schema):
        name = typesystem.String(max_length=100)
        id = typesystem.String(read_only=True)
        age = typesystem.Integer(minimum=0)

    schema_with_ro = SchemaWithReadOnly()
    form6 = Form(env=mock_env, schema=schema_with_ro, values=None)
    
    # Provide data including read_only field
    data_with_ro = {"name": "Test", "id": "123", "age": 25}
    form6.validate(data_with_ro)
    
    # Read-only field should be ignored in validation
    assert form6.is_valid == True
    assert form6.validated_data == {"name": "Test", "age": 25}

    # Test 7: Check that values are serialized before validation
    class ComplexSchema(typesystem.Schema):
        active = typesystem.Boolean()
        score = typesystem.Float()

    complex_schema = ComplexSchema()
    form7 = Form(env=mock_env, schema=complex_schema, values=None)
    
    # Validate with string values that should be converted
    form7.validate({"active": "true", "score": "42.5"})
    
    assert form7.is_valid == True
    assert form7.validated_data == {"active": True, "score": 42.5}

    # Test 8: Empty validation (no data provided)
    form8 = Form(env=mock_env, schema=schema, values=None)
    form8.validate({})
    
    assert form8._validate_called == True
    assert form8.is_valid == False  # Required fields missing
    assert form8.errors is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_Form___html__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
                <input type="{{ input_type }}" 
                       id="{{ field_id }}" 
                       name="{{ field_name }}" 
                       value="{{ value }}" 
                       {% if required %}required{% endif %}>
            """,
            "forms/textarea.html": """
                <textarea id="{{ field_id }}" 
                          name="{{ field_name }}" 
                          {% if required %}required{% endif %}>{{ value }}</textarea>
            """,
            "forms/select.html": """
                <select id="{{ field_id }}" 
                        name="{{ field_name }}" 
                        {% if required %}required{% endif %}>
                </select>
            """,
            "forms/checkbox.html": """
                <input type="checkbox" 
                       id="{{ field_id }}" 
                       name="{{ field_name }}" 
                       {% if required %}required{% endif %}>
            """
        })
    )
    
    form = Form(env=env, schema=TestSchema, values={"name": "John", "email": "john@example.com"})
    
    html_output = form.__html__()
    
    assert isinstance(html_output, markupsafe.Markup)
    assert '<input type="text"' in str(html_output)
    assert 'id="name"' in str(html_output)
    assert 'value="John"' in str(html_output)
    assert 'type="email"' in str(html_output)
    assert 'id="email"' in str(html_output)
    assert 'value="john@example.com"' in str(html_output)
    
    empty_form = Form(env=env, schema=TestSchema)
    empty_html = empty_form.__html__()
    assert isinstance(empty_html, markupsafe.Markup)
    assert 'value=""' in str(empty_html)
    
    form_with_errors = Form(env=env, schema=TestSchema)
    form_with_errors.validate({"email": "invalid-email"})
    error_html = form_with_errors.__html__()
    assert isinstance(error_html, markupsafe.Markup)


# LLM-generated content at query #9
#--------------------------

```python
def test_Form_render_field():
    import jinja2
    from typesystem.fields import String, Choice, Boolean, Object
    from typesystem.schemas import Schema
    
    # Create a minimal Jinja2 environment with mock templates
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
                <input type="{{ input_type }}" 
                       id="{{ field_id }}" 
                       name="{{ field_name }}" 
                       value="{{ value }}" 
                       {% if required %}required{% endif %}>
            """,
            "forms/textarea.html": """
                <textarea id="{{ field_id }}" 
                          name="{{ field_name }}" 
                          {% if required %}required{% endif %}>{{ value }}</textarea>
            """,
            "forms/select.html": """
                <select id="{{ field_id }}" 
                        name="{{ field_name }}" 
                        {% if required %}required{% endif %}>
                    <option value="">Select</option>
                </select>
            """,
            "forms/checkbox.html": """
                <input type="checkbox" 
                       id="{{ field_id }}" 
                       name="{{ field_name }}" 
                       value="true" 
                       {% if value %}checked{% endif %}>
            """
        })
    )
    
    # Test 1: String field with text format (should use textarea)
    class TestSchema1(Schema):
        description = String(format="text", title="Description")
    
    schema1 = TestSchema1()
    form1 = Form(env=env, schema=schema1)
    
    result1 = form1.render_field(
        field_name="description",
        field=schema1.fields["description"],
        value="Test description"
    )
    assert "textarea" in result1
    assert "Test description" in result1
    
    # Test 2: String field with email format (should use input with type=email)
    class TestSchema2(Schema):
        email = String(format="email", title="Email Address")
    
    schema2 = TestSchema2()
    form2 = Form(env=env, schema=schema2)
    
    result2 = form2.render_field(
        field_name="email",
        field=schema2.fields["email"],
        value="test@example.com"
    )
    assert 'type="email"' in result2
    assert "test@example.com" in result2
    
    # Test 3: Choice field (should use select)
    class TestSchema3(Schema):
        color = Choice(choices=["red", "green", "blue"], title="Favorite Color")
    
    schema3 = TestSchema3()
    form3 = Form(env=env, schema=schema3)
    
    result3 = form3.render_field(
        field_name="color",
        field=schema3.fields["color"]
    )
    assert "select" in result3
    
    # Test 4: Boolean field (should use checkbox)
    class TestSchema4(Schema):
        active = Boolean(title="Active Status")
    
    schema4 = TestSchema4()
    form4 = Form(env=env, schema=schema4)
    
    result4 = form4.render_field(
        field_name="active",
        field=schema4.fields["active"],
        value=True
    )
    assert 'type="checkbox"' in result4
    assert "checked" in result4
    
    # Test 5: Field with error
    class TestSchema5(Schema):
        username = String(title="Username", min_length=3)
    
    schema5 = TestSchema5()
    form5 = Form(env=env, schema=schema5)
    
    result5 = form5.render_field(
        field_name="username",
        field=schema5.fields["username"],
        value="ab",
        error="Must be at least 3 characters"
    )
    assert "ab" in result5
    
    # Test 6: Required field
    class TestSchema6(Schema):
        name = String(title="Full Name")
    
    schema6 = TestSchema6()
    form6 = Form(env=env, schema=schema6)
    
    result6 = form6.render_field(
        field_name="name",
        field=schema6.fields["name"]
    )
    assert "required" in result6
    
    # Test 7: Field with allow_null (not required)
    class TestSchema7(Schema):
        optional = String(title="Optional Field", allow_null=True)
    
    schema7 = TestSchema7()
    form7 = Form(env=env, schema=schema7)
    
    result7 = form7.render_field(
        field_name="optional",
        field=schema7.fields["optional"]
    )
    assert "required" not in result7
    
    # Test 8: Password field (value should be empty string)
    class TestSchema8(Schema):
        password = String(format="password", title="Password")
    
    schema8 = TestSchema8()
    form8 = Form(env=env, schema=schema8)
    
    result8 = form8.render_field(
        field_name="password",
        field=schema8.fields["password"],
        value="secret123"
    )
    assert 'value=""' in result8
    
    # Test 9: Field with custom format not in FORMAT_TO_INPUTTYPE
    class TestSchema9(Schema):
        custom = String(format="unknown", title="Custom Field")
    
    schema9 = TestSchema9()
    form9 = Form(env=env, schema=schema9)
    
    result9 = form9.render_field(
        field_name="custom",
        field=schema9.fields["custom"]
    )
    assert 'type="text"' in result9
    
    # Test 10: Field name with underscores converted to hyphens in field_id
    class TestSchema10(Schema):
        first_name = String(title="First Name")
    
    schema10 = TestSchema10()
    form10 = Form(env=env, schema=schema10)
    
    result10 = form10.render_field(
        field_name="first_name",
        field=schema10.fields["first_name"]
    )
    assert 'id="first-name"' in result10


# LLM-generated content at query #10
#--------------------------

```python
def test_Form_render_fields():
    # Mock jinja2 environment and templates
    class MockTemplate:
        def render(self, context):
            field_id = context["field_id"]
            field_name = context["field_name"]
            value = context["value"]
            error = context["error"]
            required = context["required"]
            input_type = context["input_type"]
            
            if error:
                return f'<div class="error">{field_name}: {error}</div>'
            return f'<input id="{field_id}" name="{field_name}" type="{input_type}" value="{value}" required="{required}">'

    class MockEnv:
        def get_template(self, template_name):
            return MockTemplate()

    # Mock schema with different field types
    class MockField:
        def __init__(self, read_only=False, title=None, allow_null=False, allow_blank=False, has_default=False, format=None):
            self.read_only = read_only
            self.title = title
            self.allow_null = allow_null
            self.allow_blank = allow_blank
            self._has_default = has_default
            self.format = format
        
        def has_default(self):
            return self._has_default

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
        
        def serialize(self, values):
            return values or {}
        
        def validate_or_error(self, data):
            if data == {"error": "test"}:
                return None, {"field1": "Validation error"}
            return data, None

    # Test 1: Basic rendering with values
    env = MockEnv()
    fields = {
        "field1": MockField(title="Field One", allow_null=True),
        "field2": MockField(has_default=True),
    }
    schema = MockSchema(fields)
    form = Form(env=env, schema=schema, values={"field1": "value1", "field2": "value2"})
    
    # Call validate to set data
    form.validate({"field1": "new_value1", "field2": "new_value2"})
    
    result = form.render_fields()
    assert '<input id="field1" name="field1" type="text" value="new_value1" required="False">' in result
    assert '<input id="field2" name="field2" type="text" value="new_value2" required="False">' in result

    # Test 2: Rendering with errors
    form2 = Form(env=env, schema=schema, values={})
    form2.validate({"error": "test"})
    
    result2 = form2.render_fields()
    assert '<div class="error">field1: Validation error</div>' in result2

    # Test 3: Rendering without calling validate first
    form3 = Form(env=env, schema=schema, values={"field1": "initial"})
    result3 = form3.render_fields()
    assert '<input id="field1" name="field1" type="text" value="initial" required="False">' in result3

    # Test 4: Skip read-only fields
    fields_with_readonly = {
        "field1": MockField(title="Field One"),
        "field2": MockField(read_only=True, title="Read Only Field"),
    }
    schema2 = MockSchema(fields_with_readonly)
    form4 = Form(env=env, schema=schema2, values={"field1": "value1", "field2": "value2"})
    form4.validate({"field1": "new_value"})
    
    result4 = form4.render_fields()
    assert "field1" in result4
    assert "field2" not in result4

    # Test 5: Empty values
    form5 = Form(env=env, schema=schema, values=None)
    form5.validate(None)
    
    result5 = form5.render_fields()
    assert 'value=""' in result5 or 'value="None"' in result5

    # Test 6: Field with no title uses field name
    fields_no_title = {
        "test_field": MockField(title=None),
    }
    schema3 = MockSchema(fields_no_title)
    form6 = Form(env=env, schema=schema3, values={})
    form6.validate({"test_field": "test"})
    
    result6 = form6.render_fields()
    assert "test_field" in result6

    # Test 7: Required field detection
    fields_required = {
        "required_field": MockField(allow_null=False, allow_blank=False, has_default=False),
        "optional_field": MockField(allow_null=True, has_default=False),
    }
    schema4 = MockSchema(fields_required)
    form7 = Form(env=env, schema=schema4, values={})
    form7.validate({})
    
    result7 = form7.render_fields()
    assert 'required="True"' in result7
    assert 'required="False"' in result7


# LLM-generated content at query #11
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean(default=True)
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            'forms/select.html': '<select name="{{ field_name }}"></select>'
        })
    )
    
    form = Form(env=env, schema=TestSchema())
    result = str(form)
    
    assert '<input type="text" name="name" value="">' in result
    assert '<input type="email" name="email" value="">' in result
    assert '<input type="text" name="age" value="">' in result
    assert '<input type="checkbox" name="active" checked>' in result
    
    form_with_values = Form(env=env, schema=TestSchema(), values={
        'name': 'John Doe',
        'email': 'john@example.com',
        'age': 30,
        'active': False
    })
    result_with_values = str(form_with_values)
    
    assert '<input type="text" name="name" value="John Doe">' in result_with_values
    assert '<input type="email" name="email" value="john@example.com">' in result_with_values
    assert '<input type="text" name="age" value="30">' in result_with_values
    assert '<input type="checkbox" name="active">' in result_with_values
    
    form.validate({'name': '', 'email': 'invalid', 'age': 'not-a-number'})
    result_with_errors = str(form)
    
    assert '<input type="text" name="name" value="">' in result_with_errors
    assert '<input type="email" name="email" value="invalid">' in result_with_errors
    assert '<input type="text" name="age" value="not-a-number">' in result_with_errors
    
    class ReadOnlySchema(Schema):
        id = fields.Integer(read_only=True)
        name = fields.String()
    
    read_only_form = Form(env=env, schema=ReadOnlySchema())
    read_only_result = str(read_only_form)
    
    assert 'id' not in read_only_result
    assert '<input type="text" name="name" value="">' in read_only_result


# LLM-generated content at query #12
#--------------------------

```python
def test_Form_render_fields():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        read_only_field = fields.String(read_only=True)
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        "forms/select.html": '<select name="{{ field_name }}">{{ value }}</select>'
    }))
    
    schema = TestSchema()
    
    form = Form(env=env, schema=schema, values={"name": "John", "email": "john@test.com", "age": 25, "active": True})
    
    html = form.render_fields()
    
    assert '<input type="text" name="name" value="John">' in html
    assert '<input type="email" name="email" value="john@test.com">' in html
    assert '<input type="text" name="age" value="25">' in html
    assert '<input type="checkbox" name="active" checked>' in html
    assert "read_only_field" not in html
    
    form_with_errors = Form(env=env, schema=schema)
    form_with_errors.validate({"name": "", "email": "invalid"})
    form_with_errors._validate_called = True
    
    html_with_errors = form_with_errors.render_fields()
    
    assert '<input type="text" name="name" value="">' in html_with_errors
    assert '<input type="email" name="email" value="invalid">' in html_with_errors
    
    form_no_validate = Form(env=env, schema=schema)
    html_no_validate = form_no_validate.render_fields()
    
    assert '<input type="text" name="name" value="None">' in html_no_validate


# LLM-generated content at query #13
#--------------------------

```python
def test_Form_render_fields():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user"])
        read_only_field = fields.String(read_only=True)
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
        "forms/select.html": '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    }))
    
    schema = TestSchema()
    
    form = Form(env=env, schema=schema, values={"name": "John", "age": 30, "active": True})
    form.validate({"name": "Jane", "email": "jane@test.com", "age": "invalid", "active": False})
    
    result = form.render_fields()
    
    assert "name" in result
    assert "email" in result
    assert "age" in result
    assert "active" in result
    assert "role" in result
    assert "read_only_field" not in result
    
    assert '<input type="text" name="name" value="Jane">' in result
    assert '<input type="email" name="email" value="jane@test.com">' in result
    assert '<input type="checkbox" name="active">' in result
    assert '<select name="role">' in result
    
    form2 = Form(env=env, schema=schema)
    form2.validate(None)
    
    result2 = form2.render_fields()
    assert 'value=""' in result2 or "value=" not in result2
    
    form3 = Form(env=env, schema=schema, values={"name": "Original"})
    result3 = form3.render_fields()
    assert 'value="Original"' in result3


# LLM-generated content at query #14
#--------------------------

```python
def test_Form___html__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
        'forms/select.html': '<select name="{{ field_name }}"><option>{{ value }}</option></select>'
    }))
    
    form = Form(env=env, schema=TestSchema())
    form.validate({"name": "John", "email": "john@example.com"})
    
    html_output = form.__html__()
    
    assert isinstance(html_output, type(jinja2.utils.htmlsafe_json_dumps("test")))
    assert '<input type="text" name="name" value="John">' in str(html_output)
    assert '<input type="email" name="email" value="john@example.com">' in str(html_output)
    
    form2 = Form(env=env, schema=TestSchema())
    form2.validate({})
    
    html_output2 = form2.__html__()
    assert isinstance(html_output2, type(jinja2.utils.htmlsafe_json_dumps("test")))
    
    class SchemaWithAllFields(Schema):
        text = fields.String(format="text")
        boolean = fields.Boolean()
        choice = fields.Choice(choices=[("a", "A"), ("b", "B")])
    
    form3 = Form(env=env, schema=SchemaWithAllFields())
    form3.validate({"text": "Hello", "boolean": True, "choice": "a"})
    
    html_output3 = form3.__html__()
    assert isinstance(html_output3, type(jinja2.utils.htmlsafe_json_dumps("test")))
    assert '<textarea name="text">Hello</textarea>' in str(html_output3)
    assert '<input type="checkbox" name="boolean" checked>' in str(html_output3)
    assert '<select name="choice"><option>a</option></select>' in str(html_output3)


# LLM-generated content at query #15
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user"])
        bio = fields.String(format="text")
        read_only_field = fields.String(read_only=True)
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": """
        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" 
               value="{{ value }}" {% if required %}required{% endif %}>
        """,
        "forms/checkbox.html": """
        <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" 
               {% if value %}checked{% endif %} {% if required %}required{% endif %}>
        """,
        "forms/select.html": """
        <select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>
            <option value="">Select</option>
        </select>
        """,
        "forms/textarea.html": """
        <textarea name="{{ field_name }}" id="{{ field_id }}" 
                  {% if required %}required{% endif %}>{{ value }}</textarea>
        """
    }))
    
    schema = TestSchema()
    
    form = Form(env=env, schema=schema, values={
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "active": True,
        "role": "admin",
        "bio": "Software developer"
    })
    
    result = str(form)
    
    assert "<input" in result
    assert 'name="name"' in result
    assert 'value="John Doe"' in result
    assert 'id="name"' in result
    
    assert 'type="email"' in result
    assert 'name="email"' in result
    assert 'value="john@example.com"' in result
    
    assert 'type="text"' in result
    assert 'name="age"' in result
    assert 'value="30"' in result
    
    assert 'type="checkbox"' in result
    assert 'name="active"' in result
    assert "checked" in result
    
    assert "<select" in result
    assert 'name="role"' in result
    assert 'id="role"' in result
    
    assert "<textarea" in result
    assert 'name="bio"' in result
    assert "Software developer" in result
    
    assert "read_only_field" not in result
    
    empty_form = Form(env=env, schema=schema, values=None)
    empty_result = str(empty_form)
    
    assert 'value=""' in empty_result
    assert "checked" not in empty_result
    
    form_with_errors = Form(env=env, schema=schema, values={
        "name": "",
        "email": "invalid-email"
    })
    form_with_errors.validate({"name": "", "email": "invalid-email"})
    
    error_result = str(form_with_errors)
    assert 'value=""' in error_result
    assert 'value="invalid-email"' in error_result


# LLM-generated content at query #16
#--------------------------

```python
def test_Form_render_field():
    import jinja2
    from typesystem.fields import String, Choice, Boolean, Object
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(title="Full Name")
        email = String(format="email")
        password = String(format="password")
        country = Choice(choices=[("us", "USA"), ("uk", "UK")])
        active = Boolean()
        description = String(format="text")
        hidden_field = String(format="hidden")
        number_field = String(format="number")
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
                <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" 
                       value="{{ value }}" {% if required %}required{% endif %}>
            """,
            "forms/textarea.html": """
                <textarea id="{{ field_id }}" name="{{ field_name }}" 
                          {% if required %}required{% endif %}>{{ value }}</textarea>
            """,
            "forms/select.html": """
                <select id="{{ field_id }}" name="{{ field_name }}" 
                        {% if required %}required{% endif %}>
                    <option value="">Select</option>
                </select>
            """,
            "forms/checkbox.html": """
                <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" 
                       value="true" {% if value %}checked{% endif %}>
            """
        })
    )
    
    schema = TestSchema()
    form = Form(env=env, schema=schema)
    
    # Test String field with text format (should use textarea)
    result = form.render_field(
        field_name="description",
        field=schema.fields["description"],
        value="Test description"
    )
    assert "textarea" in result
    assert 'name="description"' in result
    assert "Test description" in result
    
    # Test String field with email format
    result = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="test@example.com"
    )
    assert 'type="email"' in result
    assert 'value="test@example.com"' in result
    
    # Test String field with password format (value should be empty)
    result = form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret123"
    )
    assert 'type="password"' in result
    assert 'value=""' in result
    assert "secret123" not in result
    
    # Test String field with hidden format
    result = form.render_field(
        field_name="hidden_field",
        field=schema.fields["hidden_field"],
        value="hidden_value"
    )
    assert 'type="hidden"' in result
    assert 'value="hidden_value"' in result
    
    # Test String field with number format
    result = form.render_field(
        field_name="number_field",
        field=schema.fields["number_field"],
        value="42"
    )
    assert 'type="number"' in result
    assert 'value="42"' in result
    
    # Test Choice field (should use select template)
    result = form.render_field(
        field_name="country",
        field=schema.fields["country"],
        value="us"
    )
    assert "select" in result
    assert 'name="country"' in result
    
    # Test Boolean field (should use checkbox template)
    result = form.render_field(
        field_name="active",
        field=schema.fields["active"],
        value=True
    )
    assert 'type="checkbox"' in result
    assert "checked" in result
    
    # Test Boolean field with False value
    result = form.render_field(
        field_name="active",
        field=schema.fields["active"],
        value=False
    )
    assert 'type="checkbox"' in result
    assert "checked" not in result
    
    # Test field with error
    result = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="invalid-email",
        error="Invalid email format"
    )
    assert 'type="email"' in result
    assert 'value="invalid-email"' in result
    
    # Test field with title from schema
    result = form.render_field(
        field_name="name",
        field=schema.fields["name"],
        value="John Doe"
    )
    assert 'name="name"' in result
    assert 'value="John Doe"' in result
    
    # Test field_id generation (underscores to hyphens)
    result = form.render_field(
        field_name="test_field",
        field=String(),
        value="test"
    )
    assert 'id="test-field"' in result
    
    # Test required attribute for non-nullable field without default
    result = form.render_field(
        field_name="required_field",
        field=String(allow_null=False),
        value="test"
    )
    assert "required" in result
    
    # Test not required for nullable field
    result = form.render_field(
        field_name="nullable_field",
        field=String(allow_null=True),
        value="test"
    )
    assert "required" not in result
    
    # Test field with no value
    result = form.render_field(
        field_name="email",
        field=schema.fields["email"]
    )
    assert 'type="email"' in result
    assert 'value=""' in result or 'value=""' in result


# LLM-generated content at query #17
#--------------------------

```python
def test_Form_render_fields():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        read_only_field = fields.String(read_only=True)
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
        "forms/select.html": '<select name="{{ field_name }}"></select>',
    }))
    
    schema = TestSchema()
    
    form_without_values = Form(env=env, schema=schema)
    form_without_values.validate({})
    
    result = form_without_values.render_fields()
    
    assert '<input type="text" name="name" value="">' in result
    assert '<input type="email" name="email" value="">' in result
    assert '<input type="text" name="age" value="">' in result
    assert '<input type="checkbox" name="active">' in result
    assert "read_only_field" not in result
    
    form_with_values = Form(env=env, schema=schema, values={
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "active": True
    })
    form_with_values.validate({})
    
    result = form_with_values.render_fields()
    
    assert '<input type="text" name="name" value="John Doe">' in result
    assert '<input type="email" name="email" value="john@example.com">' in result
    assert '<input type="text" name="age" value="30">' in result
    assert '<input type="checkbox" name="active" checked>' in result
    
    form_with_errors = Form(env=env, schema=schema)
    form_with_errors.validate({"name": "", "email": "invalid"})
    
    result = form_with_errors.render_fields()
    
    assert '<input type="text" name="name" value="">' in result
    assert '<input type="email" name="email" value="invalid">' in result
    
    class TextAreaSchema(Schema):
        description = fields.String(format="text")
    
    env_with_textarea = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
    }))
    
    textarea_form = Form(env=env_with_textarea, schema=TextAreaSchema())
    textarea_form.validate({})
    
    result = textarea_form.render_fields()
    
    assert '<textarea name="description">' in result
    
    class ChoiceSchema(Schema):
        color = fields.Choice(choices=[("red", "Red"), ("blue", "Blue")])
    
    env_with_select = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/select.html": '<select name="{{ field_name }}"></select>',
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
    }))
    
    choice_form = Form(env=env_with_select, schema=ChoiceSchema())
    choice_form.validate({})
    
    result = choice_form.render_fields()
    
    assert '<select name="color">' in result


# LLM-generated content at query #18
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
                <input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">
            """,
            "forms/textarea.html": """
                <textarea name="{{ field_name }}">{{ value }}</textarea>
            """,
            "forms/checkbox.html": """
                <input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>
            """,
            "forms/select.html": """
                <select name="{{ field_name }}">
                    <option value="{{ value }}">{{ value }}</option>
                </select>
            """,
        })
    )
    
    form = Form(env=env, schema=TestSchema())
    result = str(form)
    
    assert "<input" in result
    assert "name=\"name\"" in result
    assert "name=\"email\"" in result
    assert "type=\"email\"" in result
    assert "name=\"age\"" in result
    assert "type=\"text\"" in result
    assert "name=\"active\"" in result
    assert "type=\"checkbox\"" in result
    
    form_with_values = Form(env=env, schema=TestSchema(), values={"name": "John", "active": True})
    result_with_values = str(form_with_values)
    
    assert "value=\"John\"" in result_with_values
    assert "checked" in result_with_values
    
    form_with_errors = Form(env=env, schema=TestSchema())
    form_with_errors.validate({"email": "invalid"})
    result_with_errors = str(form_with_errors)
    
    assert "<input" in result_with_errors
    assert "name=\"email\"" in result_with_errors
    assert "value=\"invalid\"" in result_with_errors


# LLM-generated content at query #19
#--------------------------

```python
def test_Form___html__():
    import jinja2
    import markupsafe
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
                <input type="{{ input_type }}" 
                       id="{{ field_id }}" 
                       name="{{ field_name }}" 
                       value="{{ value }}">
            """,
            "forms/textarea.html": """
                <textarea id="{{ field_id }}" 
                          name="{{ field_name }}">{{ value }}</textarea>
            """,
            "forms/checkbox.html": """
                <input type="checkbox" 
                       id="{{ field_id }}" 
                       name="{{ field_name }}" 
                       {% if value %}checked{% endif %}>
            """,
            "forms/select.html": """
                <select id="{{ field_id }}" name="{{ field_name }}">
                    <option value="{{ value }}">{{ value }}</option>
                </select>
            """
        }),
        autoescape=True
    )
    
    form = Form(env=env, schema=TestSchema)
    html_output = form.__html__()
    
    assert isinstance(html_output, markupsafe.Markup)
    assert "name" in str(html_output)
    assert "email" in str(html_output)
    
    form2 = Form(env=env, schema=TestSchema, values={"name": "John", "email": "john@example.com"})
    html_output2 = form2.__html__()
    
    assert isinstance(html_output2, markupsafe.Markup)
    assert "John" in str(html_output2)
    assert "john@example.com" in str(html_output2)
    
    form3 = Form(env=env, schema=TestSchema)
    form3.validate({"name": "", "email": "invalid"})
    html_output3 = form3.__html__()
    
    assert isinstance(html_output3, markupsafe.Markup)
    
    assert form.__html__() == markupsafe.Markup(form.render_fields())
    assert form2.__html__() == markupsafe.Markup(form2.render_fields())
    assert form3.__html__() == markupsafe.Markup(form3.render_fields())


# LLM-generated content at query #20
#--------------------------

```python
def test_Form___html__():
    import jinja2
    import markupsafe
    
    # Create a simple schema for testing
    class TestSchema(Schema):
        name = String(title="Full Name")
        email = String(format="email")
    
    # Create a jinja2 environment with a simple template
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            'forms/select.html': '<select name="{{ field_name }}"></select>'
        }),
        autoescape=True
    )
    
    # Test 1: Basic __html__ method returns Markup object
    schema = TestSchema()
    form = Form(env=env, schema=schema, values={"name": "John", "email": "john@example.com"})
    html_output = form.__html__()
    
    assert isinstance(html_output, markupsafe.Markup)
    assert "name" in str(html_output)
    assert "email" in str(html_output)
    
    # Test 2: __html__ returns escaped HTML
    schema2 = TestSchema()
    form2 = Form(env=env, schema=schema2, values={"name": "<script>alert('xss')</script>", "email": "test@example.com"})
    html_output2 = form2.__html__()
    
    assert isinstance(html_output2, markupsafe.Markup)
    html_str = str(html_output2)
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in html_str or "&lt;script&gt;alert('xss')&lt;/script&gt;" in html_str
    
    # Test 3: __html__ works with empty values
    schema3 = TestSchema()
    form3 = Form(env=env, schema=schema3, values={})
    html_output3 = form3.__html__()
    
    assert isinstance(html_output3, markupsafe.Markup)
    assert "name" in str(html_output3)
    assert "email" in str(html_output3)
    
    # Test 4: __html__ returns same as render_fields()
    schema4 = TestSchema()
    form4 = Form(env=env, schema=schema4, values={"name": "Alice", "email": "alice@example.com"})
    render_fields_output = form4.render_fields()
    html_output4 = form4.__html__()
    
    assert str(html_output4) == render_fields_output
    
    # Test 5: __html__ works with different field types
    class ComplexSchema(Schema):
        active = Boolean()
        role = Choice(choices=[("admin", "Admin"), ("user", "User")])
        bio = String(format="text")
    
    schema5 = ComplexSchema()
    form5 = Form(env=env, schema=schema5, values={"active": True, "role": "admin", "bio": "Test bio"})
    html_output5 = form5.__html__()
    
    assert isinstance(html_output5, markupsafe.Markup)
    html_str5 = str(html_output5)
    assert "checkbox" in html_str5
    assert "select" in html_str5
    assert "textarea" in html_str5


# LLM-generated content at query #21
#--------------------------

```python
def test_Form_validate():
    # Mock schema and environment
    class MockSchema:
        def __init__(self):
            self.fields = {}
        
        def serialize(self, values):
            return values or {}
        
        def validate_or_error(self, data):
            if data and data.get("error"):
                return None, {"field": "Error message"}
            return data or {}, None
    
    class MockEnv:
        def get_template(self, name):
            return MockTemplate()
    
    class MockTemplate:
        def render(self, context):
            return ""
    
    # Test 1: Successful validation
    schema = MockSchema()
    env = MockEnv()
    form = Form(env=env, schema=schema)
    
    test_data = {"field": "value"}
    form.validate(test_data)
    
    assert form._validate_called == True
    assert form.data == test_data
    assert form.values == test_data
    assert form.errors == None
    assert form.is_valid == True
    assert form.validated_data == test_data
    
    # Test 2: Failed validation
    schema2 = MockSchema()
    form2 = Form(env=env, schema=schema2)
    
    error_data = {"field": "value", "error": True}
    form2.validate(error_data)
    
    assert form2._validate_called == True
    assert form2.data == error_data
    assert form2.values == None
    assert form2.errors == {"field": "Error message"}
    assert form2.is_valid == False
    
    # Test 3: Validate called twice should raise assertion
    schema3 = MockSchema()
    form3 = Form(env=env, schema=schema3)
    form3.validate({})
    
    try:
        form3.validate({})
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert str(e) == "validate() has already been called."
    
    # Test 4: Validate with None data
    schema4 = MockSchema()
    form4 = Form(env=env, schema=schema4)
    form4.validate(None)
    
    assert form4._validate_called == True
    assert form4.data == None
    assert form4.values == {}
    assert form4.errors == None
    
    # Test 5: is_valid called before validate should raise assertion
    schema5 = MockSchema()
    form5 = Form(env=env, schema=schema5)
    
    try:
        _ = form5.is_valid
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert str(e) == "validate() has not been called."


# LLM-generated content at query #22
#--------------------------

```python
def test_Form_input_type_for_field():
    # Mock environment and schema for Form initialization
    mock_env = type("MockEnv", (), {"get_template": lambda x: None})()
    mock_schema = type("MockSchema", (), {"fields": {}, "serialize": lambda x: x, "validate_or_error": lambda x: (x, None)})()

    form = Form(env=mock_env, schema=mock_schema)

    # Test field with no format attribute
    field_without_format = type("MockField", (), {})()
    assert form.input_type_for_field(field_without_format) == "text"

    # Test field with format but not in FORMAT_TO_INPUTTYPE
    field_unknown_format = type("MockField", (), {"format": "unknown"})()
    assert form.input_type_for_field(field_unknown_format) == "text"

    # Test all formats in FORMAT_TO_INPUTTYPE
    for format_str, expected_type in Form.FORMAT_TO_INPUTTYPE.items():
        field = type("MockField", (), {"format": format_str})()
        assert form.input_type_for_field(field) == expected_type

    # Test field with empty format string
    field_empty_format = type("MockField", (), {"format": ""})()
    assert form.input_type_for_field(field_empty_format) == "text"

    # Test field with format=None explicitly
    field_none_format = type("MockField", (), {"format": None})()
    assert form.input_type_for_field(field_none_format) == "text"


# LLM-generated content at query #23
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name", max_length=100)
        email = fields.String(format="email")
        age = fields.Integer(minimum=0, maximum=150)
        active = fields.Boolean(default=True)
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
        'forms/select.html': '<select name="{{ field_name }}"></select>'
    }))
    
    schema = TestSchema()
    form = Form(env=env, schema=schema, values={"name": "John", "email": "john@example.com"})
    
    result = str(form)
    
    assert '<input type="text" name="name" value="John">' in result
    assert '<input type="email" name="email" value="john@example.com">' in result
    assert '<input type="text" name="age" value="">' in result
    assert '<input type="checkbox" name="active" checked>' in result
    
    form2 = Form(env=env, schema=schema, values=None)
    result2 = str(form2)
    
    assert '<input type="text" name="name" value="">' in result2
    assert '<input type="email" name="email" value="">' in result2
    assert '<input type="text" name="age" value="">' in result2
    assert '<input type="checkbox" name="active">' in result2
    
    form3 = Form(env=env, schema=schema, values={})
    result3 = str(form3)
    
    assert '<input type="text" name="name" value="">' in result3
    assert '<input type="email" name="email" value="">' in result3
    assert '<input type="text" name="age" value="">' in result3
    assert '<input type="checkbox" name="active">' in result3


# LLM-generated content at query #24
#--------------------------

```python
def test_Form_render_fields():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        read_only_field = fields.String(read_only=True)
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
        "forms/select.html": '<select name="{{ field_name }}"></select>',
    }))
    
    schema = TestSchema()
    
    # Test with no values and no errors
    form = Form(env=env, schema=schema)
    html = form.render_fields()
    assert "name" in html
    assert "email" in html
    assert "age" in html
    assert "active" in html
    assert "read_only_field" not in html
    
    # Test with values and no errors
    form = Form(env=env, schema=schema, values={"name": "John", "email": "john@example.com"})
    html = form.render_fields()
    assert 'value="John"' in html
    assert 'value="john@example.com"' in html
    
    # Test with errors (should use self.data instead of self.values)
    form = Form(env=env, schema=schema)
    form.validate({"name": "", "email": "invalid"})
    html = form.render_fields()
    assert 'value=""' in html  # Should use empty value from self.data
    
    # Test with errors but no self.data (edge case)
    form = Form(env=env, schema=schema)
    form.validate({})
    html = form.render_fields()
    assert html is not None
    
    # Test with Boolean field
    form = Form(env=env, schema=schema, values={"active": True})
    html = form.render_fields()
    assert "checkbox" in html
    
    # Test with read_only field exclusion
    form = Form(env=env, schema=schema, values={"read_only_field": "should not appear"})
    html = form.render_fields()
    assert "read_only_field" not in html
    
    # Test with String field with text format
    class TextSchema(Schema):
        description = fields.String(format="text")
    
    text_schema = TextSchema()
    form = Form(env=env, schema=text_schema)
    html = form.render_fields()
    assert "textarea" in html
    
    # Test with Choice field
    class ChoiceSchema(Schema):
        color = fields.Choice(choices=[("red", "Red"), ("blue", "Blue")])
    
    choice_schema = ChoiceSchema()
    form = Form(env=env, schema=choice_schema)
    html = form.render_fields()
    assert "select" in html


# LLM-generated content at query #25
#--------------------------

```python
def test_Form_validate():
    import pytest
    from unittest.mock import Mock, MagicMock
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer

    # Mock jinja2 environment
    mock_env = Mock()
    
    # Test 1: Successful validation
    def test_successful_validation():
        mock_schema = Mock(spec=Schema)
        mock_schema.serialize = Mock(return_value={"name": "John", "age": 30})
        mock_schema.validate_or_error = Mock(return_value=({"name": "John", "age": 30}, None))
        mock_schema.fields = {}
        
        form = Form(env=mock_env, schema=mock_schema, values={"name": "John", "age": 30})
        
        test_data = {"name": "John", "age": 30}
        form.validate(data=test_data)
        
        assert form._validate_called == True
        assert form.is_valid == True
        assert form.validated_data == {"name": "John", "age": 30}
        assert form.errors is None
        assert form.data == test_data

    # Test 2: Failed validation
    def test_failed_validation():
        mock_schema = Mock(spec=Schema)
        mock_schema.serialize = Mock(return_value={"name": "", "age": -5})
        mock_schema.validate_or_error = Mock(return_value=(
            {"name": "", "age": -5},
            {"name": "This field is required", "age": "Must be positive"}
        ))
        mock_schema.fields = {}
        
        form = Form(env=mock_env, schema=mock_schema, values={"name": "", "age": -5})
        
        test_data = {"name": "", "age": -5}
        form.validate(data=test_data)
        
        assert form._validate_called == True
        assert form.is_valid == False
        assert form.validated_data == {"name": "", "age": -5}
        assert form.errors == {"name": "This field is required", "age": "Must be positive"}
        assert form.data == test_data

    # Test 3: Validate called twice raises assertion
    def test_double_validate_raises_error():
        mock_schema = Mock(spec=Schema)
        mock_schema.serialize = Mock(return_value={})
        mock_schema.validate_or_error = Mock(return_value=({}, None))
        mock_schema.fields = {}
        
        form = Form(env=mock_env, schema=mock_schema, values={})
        
        form.validate(data={})
        
        with pytest.raises(AssertionError, match="validate\\(\\) has already been called"):
            form.validate(data={})

    # Test 4: Validate with no data
    def test_validate_with_no_data():
        mock_schema = Mock(spec=Schema)
        mock_schema.serialize = Mock(return_value={})
        mock_schema.validate_or_error = Mock(return_value=({}, None))
        mock_schema.fields = {}
        
        form = Form(env=mock_env, schema=mock_schema, values={})
        
        form.validate()
        
        assert form._validate_called == True
        assert form.data is None

    # Test 5: is_valid property called before validate raises assertion
    def test_is_valid_before_validate():
        mock_schema = Mock(spec=Schema)
        mock_schema.serialize = Mock(return_value={})
        mock_schema.fields = {}
        
        form = Form(env=mock_env, schema=mock_schema, values={})
        
        with pytest.raises(AssertionError, match="validate\\(\\) has not been called"):
            _ = form.is_valid

    # Test 6: Initial values are serialized
    def test_initial_values_serialized():
        mock_schema = Mock(spec=Schema)
        mock_schema.serialize = Mock(return_value={"serialized": "value"})
        mock_schema.validate_or_error = Mock(return_value=({"serialized": "value"}, None))
        mock_schema.fields = {}
        
        form = Form(env=mock_env, schema=mock_schema, values={"raw": "data"})
        
        mock_schema.serialize.assert_called_once_with({"raw": "data"})
        assert form.values == {"serialized": "value"}

    # Run all tests
    test_successful_validation()
    test_failed_validation()
    test_double_validate_raises_error()
    test_validate_with_no_data()
    test_is_valid_before_validate()
    test_initial_values_serialized()


# LLM-generated content at query #26
#--------------------------

```python
def test_Form_template_for_field():
    # Mock jinja2 environment
    class MockEnv:
        def get_template(self, name):
            return name

    # Mock schema with fields
    class MockSchema:
        def __init__(self, fields):
            self.fields = fields

    # Test String field with format="text"
    string_text_field = String(format="text")
    form = Form(env=MockEnv(), schema=MockSchema({}))
    assert form.template_for_field(string_text_field) == "forms/textarea.html"

    # Test String field with other format
    string_email_field = String(format="email")
    assert form.template_for_field(string_email_field) == "forms/input.html"

    # Test String field without format
    string_no_format_field = String()
    assert form.template_for_field(string_no_format_field) == "forms/input.html"

    # Test Boolean field
    boolean_field = Boolean()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"

    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    assert form.template_for_field(choice_field) == "forms/select.html"

    # Test generic Field
    generic_field = Field()
    assert form.template_for_field(generic_field) == "forms/input.html"

    # Test that Object field raises assertion
    object_field = Object(properties={})
    try:
        form.template_for_field(object_field)
        assert False, "Should have raised AssertionError for Object field"
    except AssertionError as e:
        assert "Forms do not support rendering Object fields" in str(e)


# LLM-generated content at query #27
#--------------------------

```python
def test_Form_render_field():
    import jinja2
    from typesystem.fields import String, Choice, Boolean, Object
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(title="Full Name")
        email = String(format="email")
        password = String(format="password")
        bio = String(format="text")
        active = Boolean()
        role = Choice(choices=[("admin", "Admin"), ("user", "User")])
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" 
                   value="{{ value }}" {% if required %}required{% endif %}>
            """,
            "forms/textarea.html": """
            <textarea id="{{ field_id }}" name="{{ field_name }}" 
                      {% if required %}required{% endif %}>{{ value }}</textarea>
            """,
            "forms/checkbox.html": """
            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" 
                   value="true" {% if value %}checked{% endif %}>
            """,
            "forms/select.html": """
            <select id="{{ field_id }}" name="{{ field_name }}" 
                    {% if required %}required{% endif %}>
                {% for option_value, option_label in field.choices %}
                <option value="{{ option_value }}" 
                        {% if option_value == value %}selected{% endif %}>
                    {{ option_label }}
                </option>
                {% endfor %}
            </select>
            """
        })
    )
    
    schema = TestSchema()
    form = Form(env=env, schema=schema)
    
    # Test String field with text format (textarea)
    result = form.render_field(
        field_name="bio",
        field=schema.fields["bio"],
        value="Test bio",
        error=None
    )
    assert 'textarea' in result
    assert 'id="bio"' in result
    assert 'name="bio"' in result
    assert 'Test bio' in result
    
    # Test String field with email format
    result = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="test@example.com",
        error="Invalid email"
    )
    assert 'type="email"' in result
    assert 'value="test@example.com"' in result
    
    # Test String field with password format (value should be empty)
    result = form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret123",
        error=None
    )
    assert 'type="password"' in result
    assert 'value=""' in result
    assert 'secret123' not in result
    
    # Test Boolean field (checkbox)
    result = form.render_field(
        field_name="active",
        field=schema.fields["active"],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in result
    assert 'checked' in result
    
    # Test Choice field (select)
    result = form.render_field(
        field_name="role",
        field=schema.fields["role"],
        value="admin",
        error=None
    )
    assert 'select' in result
    assert 'value="admin"' in result
    assert 'selected' in result
    
    # Test field with error
    result = form.render_field(
        field_name="name",
        field=schema.fields["name"],
        value="",
        error="This field is required"
    )
    assert 'type="text"' in result
    assert 'required' in result
    
    # Test field_id generation with underscores
    result = form.render_field(
        field_name="first_name",
        field=String(),
        value="John",
        error=None
    )
    assert 'id="first-name"' in result
    assert 'name="first_name"' in result
    
    # Test required flag
    required_field = String(allow_null=False)
    result = form.render_field(
        field_name="required_field",
        field=required_field,
        value=None,
        error=None
    )
    assert 'required' in result
    
    # Test not required field
    optional_field = String(allow_null=True)
    result = form.render_field(
        field_name="optional_field",
        field=optional_field,
        value=None,
        error=None
    )
    assert 'required' not in result


# LLM-generated content at query #28
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms1 = Jinja2Forms(directory="/test/dir")
    env1 = forms1.env
    assert isinstance(env1, jinja2.Environment)
    assert isinstance(env1.loader, jinja2.FileSystemLoader)
    assert env1.loader.searchpath == ["/test/dir"]
    assert env1.autoescape == True

    # Test with package only
    forms2 = Jinja2Forms(package="test_package")
    env2 = forms2.env
    assert isinstance(env2, jinja2.Environment)
    assert isinstance(env2.loader, jinja2.PackageLoader)
    assert env2.loader.package_name == "test_package"
    assert env2.loader.package_path == "templates"
    assert env2.autoescape == True

    # Test with both directory and package
    forms3 = Jinja2Forms(directory="/test/dir", package="test_package")
    env3 = forms3.env
    assert isinstance(env3, jinja2.Environment)
    assert isinstance(env3.loader, jinja2.ChoiceLoader)
    assert len(env3.loader.loaders) == 2
    assert isinstance(env3.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env3.loader.loaders[1], jinja2.PackageLoader)
    assert env3.autoescape == True


# LLM-generated content at query #29
#--------------------------

```python
def test_Form___html__():
    import jinja2
    import markupsafe
    
    # Create a simple schema for testing
    class TestSchema(Schema):
        name = String(title="Full Name")
        email = String(format="email")
    
    # Create a minimal Jinja2 environment with templates
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            'forms/select.html': '<select name="{{ field_name }}">{{ value }}</select>'
        }),
        autoescape=True
    )
    
    # Test 1: Basic __html__ method returns Markup object
    schema = TestSchema()
    form = Form(env=env, schema=schema, values={"name": "John", "email": "john@example.com"})
    html_result = form.__html__()
    
    assert isinstance(html_result, markupsafe.Markup)
    assert "name" in str(html_result)
    assert "email" in str(html_result)
    
    # Test 2: __html__ returns same content as render_fields()
    form2 = Form(env=env, schema=schema, values={"name": "Jane", "email": "jane@example.com"})
    render_result = form2.render_fields()
    html_result2 = form2.__html__()
    
    assert str(html_result2) == render_result
    
    # Test 3: __html__ escapes content properly
    schema_with_xss = Schema(fields={"comment": String()})
    form3 = Form(env=env, schema=schema_with_xss, values={"comment": "<script>alert('xss')</script>"})
    html_result3 = form3.__html__()
    
    assert isinstance(html_result3, markupsafe.Markup)
    assert "&lt;script&gt;" in str(html_result3)
    assert "<script>" not in str(html_result3)
    
    # Test 4: __html__ works with empty values
    form4 = Form(env=env, schema=schema, values=None)
    html_result4 = form4.__html__()
    
    assert isinstance(html_result4, markupsafe.Markup)
    assert "name" in str(html_result4)
    assert "email" in str(html_result4)
    
    # Test 5: __html__ works after validation
    form5 = Form(env=env, schema=schema, values=None)
    form5.validate({"name": "Test", "email": "test@example.com"})
    html_result5 = form5.__html__()
    
    assert isinstance(html_result5, markupsafe.Markup)
    assert "test@example.com" in str(html_result5)
    
    # Test 6: __html__ with different field types
    class ComplexSchema(Schema):
        active = Boolean()
        category = Choice(choices=[("a", "A"), ("b", "B")])
        description = String(format="text")
    
    form6 = Form(env=env, schema=ComplexSchema(), values={
        "active": True,
        "category": "a",
        "description": "Some text"
    })
    html_result6 = form6.__html__()
    
    assert isinstance(html_result6, markupsafe.Markup)
    assert "checkbox" in str(html_result6).lower()
    assert "select" in str(html_result6).lower()
    assert "textarea" in str(html_result6).lower()


# LLM-generated content at query #30
#--------------------------

```python
def test_Form_validate():
    # Mock schema with validate_or_error method
    class MockSchema:
        def __init__(self, fields=None):
            self.fields = fields or {}
        
        def serialize(self, values):
            return values or {}
        
        def validate_or_error(self, data):
            if data and "invalid" in str(data):
                return None, {"field": "Error message"}
            return data or {}, None
    
    # Mock jinja2 environment
    class MockEnv:
        def get_template(self, name):
            return MockTemplate()
    
    class MockTemplate:
        def render(self, context):
            return ""
    
    # Test 1: Successful validation
    schema = MockSchema()
    form = Form(env=MockEnv(), schema=schema)
    test_data = {"field": "value"}
    form.validate(test_data)
    assert form._validate_called == True
    assert form.data == test_data
    assert form.values == test_data
    assert form.errors == None
    assert form.is_valid == True
    assert form.validated_data == test_data
    
    # Test 2: Failed validation
    schema = MockSchema()
    form = Form(env=MockEnv(), schema=schema)
    test_data = {"field": "invalid"}
    form.validate(test_data)
    assert form._validate_called == True
    assert form.data == test_data
    assert form.values == None
    assert form.errors == {"field": "Error message"}
    assert form.is_valid == False
    
    # Test 3: Validate with None data
    schema = MockSchema()
    form = Form(env=MockEnv(), schema=schema)
    form.validate(None)
    assert form._validate_called == True
    assert form.data == None
    assert form.values == {}
    assert form.errors == None
    assert form.is_valid == True
    
    # Test 4: Validate with empty dict
    schema = MockSchema()
    form = Form(env=MockEnv(), schema=schema)
    form.validate({})
    assert form._validate_called == True
    assert form.data == {}
    assert form.values == {}
    assert form.errors == None
    assert form.is_valid == True
    
    # Test 5: Cannot call validate twice
    schema = MockSchema()
    form = Form(env=MockEnv(), schema=schema)
    form.validate({"test": "data"})
    try:
        form.validate({"another": "attempt"})
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert str(e) == "validate() has already been called."
    
    # Test 6: Initial values are serialized
    class SerializingSchema(MockSchema):
        def serialize(self, values):
            return {"serialized": values} if values else {}
    
    schema = SerializingSchema()
    form = Form(env=MockEnv(), schema=schema, values={"original": "data"})
    assert form.values == {"serialized": {"original": "data"}}
    
    # Test 7: Validate updates values and errors
    schema = MockSchema()
    form = Form(env=MockEnv(), schema=schema, values={"initial": "value"})
    form.validate({"new": "data"})
    assert form.values == {"new": "data"}
    assert form.errors == None
    assert form.data == {"new": "data"}


# LLM-generated content at query #31
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user"])
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
        })
    )
    
    form = Form(env=env, schema=TestSchema)
    result = str(form)
    
    assert '<input type="text" name="name" value="">' in result
    assert '<input type="email" name="email" value="">' in result
    assert '<input type="text" name="age" value="">' in result
    assert '<input type="checkbox" name="active">' in result
    assert '<select name="role">' in result
    
    form_with_values = Form(env=env, schema=TestSchema, values={
        "name": "John",
        "email": "john@example.com",
        "age": 30,
        "active": True,
        "role": "admin"
    })
    result_with_values = str(form_with_values)
    
    assert 'value="John"' in result_with_values
    assert 'value="john@example.com"' in result_with_values
    assert 'value="30"' in result_with_values
    assert 'checked' in result_with_values
    assert 'value="admin"' in result_with_values
    
    form_with_errors = Form(env=env, schema=TestSchema)
    form_with_errors.validate({"name": "", "email": "invalid"})
    result_with_errors = str(form_with_errors)
    
    assert 'value=""' in result_with_errors
    assert 'value="invalid"' in result_with_errors
    
    class ReadOnlySchema(Schema):
        name = fields.String()
        id = fields.String(read_only=True)
    
    form_readonly = Form(env=env, schema=ReadOnlySchema)
    result_readonly = str(form_readonly)
    
    assert 'name="name"' in result_readonly
    assert 'name="id"' not in result_readonly


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Form_render_field():
    import jinja2
    from typesystem.fields import String, Boolean, Choice, Object
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(title="Full Name")
        email = String(format="email")
        password = String(format="password")
        bio = String(format="text")
        active = Boolean()
        status = Choice(choices=[("active", "Active"), ("inactive", "Inactive")])
        read_only_field = String(read_only=True)
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" 
                   value="{{ value }}" {% if required %}required{% endif %}>
            """,
            "forms/textarea.html": """
            <textarea id="{{ field_id }}" name="{{ field_name }}" 
                      {% if required %}required{% endif %}>{{ value }}</textarea>
            """,
            "forms/checkbox.html": """
            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" 
                   value="true" {% if value %}checked{% endif %}>
            """,
            "forms/select.html": """
            <select id="{{ field_id }}" name="{{ field_name }}" 
                    {% if required %}required{% endif %}>
                <option value="">Select...</option>
                {% for choice_value, choice_label in field.choices %}
                <option value="{{ choice_value }}" 
                        {% if choice_value == value %}selected{% endif %}>
                    {{ choice_label }}
                </option>
                {% endfor %}
            </select>
            """
        })
    )
    
    schema = TestSchema()
    form = Form(env=env, schema=schema)
    
    # Test String field with default format
    result = form.render_field(field_name="name", field=schema.fields["name"], value="John")
    assert 'type="text"' in result
    assert 'id="name"' in result
    assert 'name="name"' in result
    assert 'value="John"' in result
    assert "required" in result
    
    # Test String field with email format
    result = form.render_field(field_name="email", field=schema.fields["email"], value="test@example.com")
    assert 'type="email"' in result
    assert 'id="email"' in result
    assert 'value="test@example.com"' in result
    
    # Test String field with password format (value should be empty string)
    result = form.render_field(field_name="password", field=schema.fields["password"], value="secret")
    assert 'type="password"' in result
    assert 'value=""' in result
    
    # Test String field with text format (should use textarea template)
    result = form.render_field(field_name="bio", field=schema.fields["bio"], value="Some bio")
    assert "<textarea" in result
    assert "id=\"bio\"" in result
    assert "Some bio" in result
    
    # Test Boolean field (should use checkbox template)
    result = form.render_field(field_name="active", field=schema.fields["active"], value=True)
    assert 'type="checkbox"' in result
    assert "checked" in result
    assert 'id="active"' in result
    
    # Test Boolean field unchecked
    result = form.render_field(field_name="active", field=schema.fields["active"], value=False)
    assert 'type="checkbox"' in result
    assert "checked" not in result
    
    # Test Choice field (should use select template)
    result = form.render_field(field_name="status", field=schema.fields["status"], value="active")
    assert "<select" in result
    assert 'id="status"' in result
    assert 'value="active"' in result
    assert "Active" in result
    assert "Inactive" in result
    
    # Test field with error
    result = form.render_field(field_name="name", field=schema.fields["name"], 
                               value="John", error="Invalid name")
    assert 'type="text"' in result
    assert 'value="John"' in result
    
    # Test field with None value
    result = form.render_field(field_name="name", field=schema.fields["name"], value=None)
    assert 'value=""' in result
    
    # Test field with empty string value
    result = form.render_field(field_name="name", field=schema.fields["name"], value="")
    assert 'value=""' in result
    
    # Test field_id generation with underscores
    result = form.render_field(field_name="test_field", field=String(), value="test")
    assert 'id="test-field"' in result
    
    # Test required field detection
    required_field = String()
    result = form.render_field(field_name="required_field", field=required_field, value="test")
    assert "required" in result
    
    # Test non-required field (allow_null=True)
    optional_field = String(allow_null=True)
    result = form.render_field(field_name="optional_field", field=optional_field, value="test")
    assert "required" not in result
    
    # Test field with default value (should not be required)
    field_with_default = String(default="default")
    result = form.render_field(field_name="with_default", field=field_with_default, value="test")
    assert "required" not in result
    
    # Test field with allow_blank=True (should not be required)
    field_allow_blank = String(allow_blank=True)
    result = form.render_field(field_name="allow_blank", field=field_allow_blank, value="test")
    assert "required" not in result


# LLM-generated content at query #2
#--------------------------

```python
def test_Form_input_type_for_field():
    # Mock environment for Form initialization
    class MockEnv:
        def get_template(self, name):
            return None

    # Mock schema for Form initialization
    class MockSchema:
        fields = {}
        
        def serialize(self, values):
            return values or {}
        
        def validate_or_error(self, data):
            return (data or {}, None)

    # Test 1: Field without format attribute returns "text"
    class FieldWithoutFormat:
        pass
    
    form = Form(env=MockEnv(), schema=MockSchema())
    field = FieldWithoutFormat()
    assert form.input_type_for_field(field) == "text"

    # Test 2: Field with format but not in FORMAT_TO_INPUTTYPE returns "text"
    class FieldWithUnknownFormat:
        format = "unknown_format"
    
    field = FieldWithUnknownFormat()
    assert form.input_type_for_field(field) == "text"

    # Test 3: Field with format in FORMAT_TO_INPUTTYPE returns correct type
    test_cases = [
        ("color", "color"),
        ("datetime", "datetime-local"),
        ("date", "date"),
        ("email", "email"),
        ("hidden", "hidden"),
        ("month", "month"),
        ("number", "number"),
        ("password", "password"),
        ("range", "range"),
        ("search", "search"),
        ("tel", "tel"),
        ("text", "text"),
        ("time", "time"),
        ("url", "url"),
        ("week", "week"),
    ]
    
    for format_str, expected_type in test_cases:
        class FieldWithFormat:
            format = format_str
        
        field = FieldWithFormat()
        assert form.input_type_for_field(field) == expected_type

    # Test 4: Field with None format returns "text"
    class FieldWithNoneFormat:
        format = None
    
    field = FieldWithNoneFormat()
    assert form.input_type_for_field(field) == "text"

    # Test 5: Field with empty string format returns "text"
    class FieldWithEmptyFormat:
        format = ""
    
    field = FieldWithEmptyFormat()
    assert form.input_type_for_field(field) == "text"


# LLM-generated content at query #3
#--------------------------

```python
def test_Form_template_for_field():
    # Mock jinja2 environment
    class MockEnv:
        def get_template(self, name):
            return name

    # Create a Form instance with minimal setup
    mock_schema = type('MockSchema', (), {'fields': {}})()
    form = Form(env=MockEnv(), schema=mock_schema)

    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    assert form.template_for_field(choice_field) == "forms/select.html"

    # Test Boolean field
    boolean_field = Boolean()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"

    # Test String field with format="text"
    string_text_field = String(format="text")
    assert form.template_for_field(string_text_field) == "forms/textarea.html"

    # Test String field with other format
    string_email_field = String(format="email")
    assert form.template_for_field(string_email_field) == "forms/input.html"

    # Test String field without format
    string_plain_field = String()
    assert form.template_for_field(string_plain_field) == "forms/input.html"

    # Test generic Field
    generic_field = Field()
    assert form.template_for_field(generic_field) == "forms/input.html"

    # Test Object field should raise assertion
    object_field = Object(properties={})
    try:
        form.template_for_field(object_field)
        assert False, "Should have raised AssertionError for Object field"
    except AssertionError as e:
        assert str(e) == "Forms do not support rendering Object fields"


# LLM-generated content at query #4
#--------------------------

```python
def test_Form_render_field():
    # Mock jinja2 environment and template
    class MockTemplate:
        def render(self, context):
            return f"rendered_with_{context['field_id']}"

    class MockEnv:
        def get_template(self, name):
            return MockTemplate()

    # Mock schema with fields
    class MockSchema:
        fields = {}

    # Test basic field rendering
    env = MockEnv()
    schema = MockSchema()
    form = Form(env=env, schema=schema, values={})
    
    # Test String field with text format
    string_field = String(format="text", title="Description")
    result = form.render_field(field_name="description", field=string_field, value="test value")
    assert "rendered_with_description" in result
    
    # Test String field with email format
    email_field = String(format="email", title="Email")
    result = form.render_field(field_name="user_email", field=email_field, value="test@example.com")
    assert "rendered_with_user-email" in result
    
    # Test Boolean field
    boolean_field = Boolean(title="Agree")
    result = form.render_field(field_name="agree_terms", field=boolean_field, value=True)
    assert "rendered_with_agree-terms" in result
    
    # Test Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")], title="Selection")
    result = form.render_field(field_name="selection", field=choice_field, value="a")
    assert "rendered_with_selection" in result
    
    # Test field with error
    result = form.render_field(field_name="username", field=String(), value="", error="Required field")
    assert "rendered_with_username" in result
    
    # Test password field (value should be empty string)
    password_field = String(format="password")
    result = form.render_field(field_name="password", field=password_field, value="secret")
    # The template rendering would show empty value for password
    
    # Test required field detection
    required_field = String(allow_null=False, allow_blank=False)
    result = form.render_field(field_name="required_field", field=required_field)
    assert "rendered_with_required-field" in result
    
    # Test non-required field
    optional_field = String(allow_null=True)
    result = form.render_field(field_name="optional_field", field=optional_field)
    assert "rendered_with_optional-field" in result
    
    # Test field with default value (should not be required)
    default_field = String(default="default")
    result = form.render_field(field_name="with_default", field=default_field)
    assert "rendered_with_with-default" in result
    
    # Test field title fallback to field_name
    no_title_field = String()
    result = form.render_field(field_name="field_without_title", field=no_title_field)
    assert "rendered_with_field-without-title" in result


# LLM-generated content at query #5
#--------------------------

```python
def test_Form_validate():
    # Mock schema and environment
    class MockSchema:
        def __init__(self):
            self.fields = {}
        
        def serialize(self, values):
            return values or {}
        
        def validate_or_error(self, data):
            if data and data.get("error"):
                return None, {"field": "Validation error"}
            return data or {}, None
    
    class MockEnv:
        def get_template(self, name):
            return MockTemplate()
    
    class MockTemplate:
        def render(self, context):
            return ""
    
    # Test 1: Successful validation
    schema = MockSchema()
    env = MockEnv()
    form = Form(env=env, schema=schema)
    
    test_data = {"field": "value"}
    form.validate(test_data)
    
    assert form._validate_called == True
    assert form.is_valid == True
    assert form.validated_data == test_data
    assert form.errors == None
    assert form.values == test_data
    
    # Test 2: Failed validation
    schema2 = MockSchema()
    form2 = Form(env=env, schema=schema2)
    
    error_data = {"field": "value", "error": True}
    form2.validate(error_data)
    
    assert form2._validate_called == True
    assert form2.is_valid == False
    assert form2.validated_data == None
    assert form2.errors == {"field": "Validation error"}
    
    # Test 3: Double validation call raises assertion
    schema3 = MockSchema()
    form3 = Form(env=env, schema=schema3)
    
    form3.validate({"test": "data"})
    
    try:
        form3.validate({"test": "data"})
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert str(e) == "validate() has already been called."
    
    # Test 4: Validation with None data
    schema4 = MockSchema()
    form4 = Form(env=env, schema=schema4)
    
    form4.validate(None)
    
    assert form4._validate_called == True
    assert form4.is_valid == True
    assert form4.validated_data == {}
    assert form4.errors == None
    
    # Test 5: Initial values are serialized
    class SerializingSchema(MockSchema):
        def serialize(self, values):
            return {"serialized": True, **values} if values else {}
    
    schema5 = SerializingSchema()
    form5 = Form(env=env, schema=schema5, values={"initial": "value"})
    
    assert form5.values == {"serialized": True, "initial": "value"}
    
    # Test 6: is_valid without validate() raises assertion
    schema6 = MockSchema()
    form6 = Form(env=env, schema=schema6)
    
    try:
        _ = form6.is_valid
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert str(e) == "validate() has not been called."


# LLM-generated content at query #6
#--------------------------

```python
def test_Form___html__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}"></select>'
        }),
        autoescape=True
    )
    
    form = Form(env=env, schema=TestSchema(), values={"name": "John", "email": "john@example.com"})
    form.validate({"name": "John", "email": "john@example.com"})
    
    html_output = form.__html__()
    
    assert isinstance(html_output, type(jinja2.utils.htmlsafe("test")))
    assert "input" in str(html_output)
    assert "name" in str(html_output)
    assert "email" in str(html_output)
    
    form2 = Form(env=env, schema=TestSchema())
    form2.validate({"name": "", "email": "invalid"})
    
    html_output2 = form2.__html__()
    assert isinstance(html_output2, type(jinja2.utils.htmlsafe("test")))
    
    form3 = Form(env=env, schema=TestSchema())
    html_output3 = form3.__html__()
    assert isinstance(html_output3, type(jinja2.utils.htmlsafe("test")))
    assert "input" in str(html_output3)


# LLM-generated content at query #7
#--------------------------

```python
def test_Form_render_fields():
    # Mock jinja2 environment and templates
    class MockTemplate:
        def render(self, context):
            field_name = context["field_name"]
            field_id = context["field_id"]
            required = context["required"]
            value = context["value"]
            error = context["error"]
            
            if context.get("input_type") == "password":
                return f'<input type="password" id="{field_id}" name="{field_name}" value="">'
            elif context.get("input_type") == "email":
                return f'<input type="email" id="{field_id}" name="{field_name}" value="{value}">'
            elif context.get("input_type") == "text":
                return f'<input type="text" id="{field_id}" name="{field_name}" value="{value}">'
            elif context["template_name"] == "forms/textarea.html":
                return f'<textarea id="{field_id}" name="{field_name}">{value}</textarea>'
            elif context["template_name"] == "forms/checkbox.html":
                checked = "checked" if value else ""
                return f'<input type="checkbox" id="{field_id}" name="{field_name}" {checked}>'
            elif context["template_name"] == "forms/select.html":
                return f'<select id="{field_id}" name="{field_name}"></select>'
            return ""

    class MockEnvironment:
        def get_template(self, name):
            template = MockTemplate()
            template.name = name
            return template

    # Mock schema with fields
    class MockField:
        def __init__(self, read_only=False, title=None, allow_null=False, allow_blank=False, has_default=False, format=None):
            self.read_only = read_only
            self.title = title
            self.allow_null = allow_null
            self.allow_blank = allow_blank
            self._has_default = has_default
            self.format = format
            
        def has_default(self):
            return self._has_default

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            
        def serialize(self, values):
            return values or {}
            
        def validate_or_error(self, data):
            if data == {"field1": "invalid"}:
                return None, {"field1": "Invalid value"}
            return data, None

    # Test 1: Render fields with no errors
    env = MockEnvironment()
    fields = {
        "field1": MockField(title="Field One", allow_null=True),
        "field2": MockField(read_only=True),
        "field3": MockField(has_default=True),
    }
    schema = MockSchema(fields)
    form = Form(env=env, schema=schema, values={"field1": "test value", "field3": "default"})
    form.validate({"field1": "test value", "field3": "default"})
    
    result = form.render_fields()
    assert 'name="field1"' in result
    assert 'value="test value"' in result
    assert 'name="field2"' not in result  # read_only field should be skipped
    assert 'name="field3"' in result
    
    # Test 2: Render fields with errors
    fields_with_error = {
        "field1": MockField(title="Field One"),
        "field2": MockField(title="Field Two", allow_blank=True),
    }
    schema_with_error = MockSchema(fields_with_error)
    form_with_error = Form(env=env, schema=schema_with_error, values={"field1": "old", "field2": "old2"})
    form_with_error.validate({"field1": "invalid"})
    
    result_with_error = form_with_error.render_fields()
    assert form_with_error.errors == {"field1": "Invalid value"}
    assert 'name="field1"' in result_with_error
    assert 'name="field2"' in result_with_error
    
    # Test 3: Render fields with password type
    class MockStringField(MockField):
        def __init__(self, format=None, **kwargs):
            super().__init__(**kwargs)
            self.format = format
    
    fields_password = {
        "password_field": MockStringField(format="password"),
        "text_field": MockStringField(format="text"),
    }
    schema_password = MockSchema(fields_password)
    form_password = Form(env=env, schema=schema_password, values={"password_field": "secret", "text_field": "visible"})
    form_password.validate({"password_field": "secret", "text_field": "visible"})
    
    result_password = form_password.render_fields()
    assert 'type="password"' in result_password
    assert 'value=""' in result_password  # password should be empty in rendered form
    assert 'value="visible"' in result_password  # text field should show value
    
    # Test 4: Render fields with different field types
    class MockChoiceField(MockField):
        pass
    
    class MockBooleanField(MockField):
        pass
    
    fields_mixed = {
        "choice_field": MockChoiceField(),
        "bool_field": MockBooleanField(),
        "text_area": MockStringField(format="text"),
    }
    schema_mixed = MockSchema(fields_mixed)
    form_mixed = Form(env=env, schema=schema_mixed, values={"bool_field": True})
    form_mixed.validate({"bool_field": True})
    
    result_mixed = form_mixed.render_fields()
    assert '<select' in result_mixed
    assert 'type="checkbox"' in result_mixed
    assert 'checked' in result_mixed
    assert '<textarea' in result_mixed
    
    # Test 5: Render fields with no values
    form_empty = Form(env=env, schema=MockSchema({"field1": MockField()}))
    form_empty.validate({})
    result_empty = form_empty.render_fields()
    assert 'name="field1"' in result_empty
    assert 'value=""' in result_empty or 'value=None' in result_empty


# LLM-generated content at query #8
#--------------------------

```python
def test_Form_input_type_for_field():
    # Mock environment and schema for Form initialization
    mock_env = type("MockEnv", (), {"get_template": lambda x: None})()
    mock_schema = type("MockSchema", (), {"fields": {}, "serialize": lambda x: x, "validate_or_error": lambda x: (x, None)})()
    
    form = Form(env=mock_env, schema=mock_schema)
    
    # Test with field without format attribute
    field_without_format = type("MockField", (), {})()
    assert form.input_type_for_field(field_without_format) == "text"
    
    # Test with field with empty format
    field_empty_format = type("MockField", (), {"format": ""})()
    assert form.input_type_for_field(field_empty_format) == "text"
    
    # Test with field with None format
    field_none_format = type("MockField", (), {"format": None})()
    assert form.input_type_for_field(field_none_format) == "text"
    
    # Test all format mappings from FORMAT_TO_INPUTTYPE
    for format_value, expected_type in Form.FORMAT_TO_INPUTTYPE.items():
        field = type("MockField", (), {"format": format_value})()
        assert form.input_type_for_field(field) == expected_type
    
    # Test with unknown format
    field_unknown_format = type("MockField", (), {"format": "unknown"})()
    assert form.input_type_for_field(field_unknown_format) == "text"
    
    # Test with format that exists but not in mapping (should fall back to text)
    field_existing_attr = type("MockField", (), {"format": "checkbox"})()
    assert form.input_type_for_field(field_existing_attr) == "text"


# LLM-generated content at query #9
#--------------------------

```python
def test_Form_input_type_for_field():
    # Mock environment and schema for Form instantiation
    mock_env = type("MockEnv", (), {"get_template": lambda x: None})()
    mock_schema = type("MockSchema", (), {"fields": {}, "serialize": lambda x: x, "validate_or_error": lambda x: (x, None)})()

    form = Form(env=mock_env, schema=mock_schema)

    # Test field without format attribute
    field_without_format = type("MockField", (), {})()
    assert form.input_type_for_field(field_without_format) == "text"

    # Test field with format but not in FORMAT_TO_INPUTTYPE
    field_with_unknown_format = type("MockField", (), {"format": "unknown"})()
    assert form.input_type_for_field(field_with_unknown_format) == "text"

    # Test all formats in FORMAT_TO_INPUTTYPE
    for format_str, expected_type in Form.FORMAT_TO_INPUTTYPE.items():
        field = type("MockField", (), {"format": format_str})()
        assert form.input_type_for_field(field) == expected_type

    # Test field with format=None (should default to "text")
    field_with_none_format = type("MockField", (), {"format": None})()
    assert form.input_type_for_field(field_with_none_format) == "text"

    # Test field with empty string format
    field_with_empty_format = type("MockField", (), {"format": ""})()
    assert form.input_type_for_field(field_with_empty_format) == "text"


# LLM-generated content at query #10
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        category = fields.Choice(choices=[("a", "Option A"), ("b", "Option B")])
        read_only_field = fields.String(read_only=True)
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
                <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" 
                       value="{{ value }}" {% if required %}required{% endif %}>
            """,
            "forms/textarea.html": """
                <textarea name="{{ field_name }}" id="{{ field_id }}" 
                          {% if required %}required{% endif %}>{{ value }}</textarea>
            """,
            "forms/checkbox.html": """
                <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" 
                       {% if value %}checked{% endif %} {% if required %}required{% endif %}>
            """,
            "forms/select.html": """
                <select name="{{ field_name }}" id="{{ field_id }}" 
                        {% if required %}required{% endif %}>
                    <option value="a" {% if value == 'a' %}selected{% endif %}>Option A</option>
                    <option value="b" {% if value == 'b' %}selected{% endif %}>Option B</option>
                </select>
            """,
        })
    )
    
    form = Form(env=env, schema=TestSchema)
    
    result = str(form)
    
    assert "name" in result
    assert "email" in result
    assert "age" in result
    assert "active" in result
    assert "category" in result
    assert "read_only_field" not in result
    
    assert 'type="text"' in result
    assert 'type="email"' in result
    assert 'type="number"' in result
    
    assert "<textarea" in result
    assert 'type="checkbox"' in result
    assert "<select" in result
    
    form_with_values = Form(env=env, schema=TestSchema, values={
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "active": True,
        "category": "a"
    })
    
    result_with_values = str(form_with_values)
    
    assert 'value="John Doe"' in result_with_values
    assert 'value="john@example.com"' in result_with_values
    assert 'value="30"' in result_with_values
    assert "checked" in result_with_values
    assert "selected" in result_with_values
    
    form.validate({"name": "", "email": "invalid", "age": -5})
    
    result_with_errors = str(form)
    
    assert result_with_errors


# LLM-generated content at query #11
#--------------------------

```python
def test_Form_template_for_field():
    # Mock jinja2 environment
    class MockEnv:
        def get_template(self, name):
            return name

    # Mock schema with fields
    class MockSchema:
        def __init__(self, fields):
            self.fields = fields

    # Test String field with format='text' returns textarea template
    string_field_text = String(format="text")
    form = Form(env=MockEnv(), schema=MockSchema({}))
    assert form.template_for_field(string_field_text) == "forms/textarea.html"

    # Test String field with other format returns input template
    string_field_email = String(format="email")
    assert form.template_for_field(string_field_email) == "forms/input.html"

    # Test String field without format returns input template
    string_field_no_format = String()
    assert form.template_for_field(string_field_no_format) == "forms/input.html"

    # Test Choice field returns select template
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    assert form.template_for_field(choice_field) == "forms/select.html"

    # Test Boolean field returns checkbox template
    boolean_field = Boolean()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"

    # Test other Field types return input template
    class CustomField(Field):
        pass

    custom_field = CustomField()
    assert form.template_for_field(custom_field) == "forms/input.html"

    # Test Object field raises assertion error
    object_field = Object(properties={})
    try:
        form.template_for_field(object_field)
        assert False, "Should have raised AssertionError for Object field"
    except AssertionError as e:
        assert "Forms do not support rendering Object fields" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_Form___html__():
    import jinja2
    import markupsafe
    from typesystem import Schema, String, Integer, Boolean
    
    class TestSchema(Schema):
        name = String(title="Full Name")
        age = Integer()
        active = Boolean()
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select name="{{ field_name }}"></select>'
    }))
    
    form = Form(env=env, schema=TestSchema(), values={"name": "John", "age": 25, "active": True})
    
    html_output = form.__html__()
    
    assert isinstance(html_output, markupsafe.Markup)
    assert '<input type="text" name="name" value="John">' in str(html_output)
    assert '<input type="number" name="age" value="25">' in str(html_output)
    assert 'checked' in str(html_output)
    
    form2 = Form(env=env, schema=TestSchema(), values=None)
    html_output2 = form2.__html__()
    assert isinstance(html_output2, markupsafe.Markup)
    assert 'value=""' in str(html_output2)
    
    form3 = Form(env=env, schema=TestSchema(), values={"name": "Test"})
    form3.validate({"name": "", "age": "invalid", "active": False})
    html_output3 = form3.__html__()
    assert isinstance(html_output3, markupsafe.Markup)
    assert 'value=""' in str(html_output3)


# LLM-generated content at query #13
#--------------------------

```python
def test_Form_render_fields():
    # Mock jinja2 environment and templates
    class MockTemplate:
        def render(self, context):
            field_id = context["field_id"]
            field_name = context["field_name"]
            required = context["required"]
            input_type = context["input_type"]
            value = context["value"]
            error = context["error"]
            return f'<input type="{input_type}" name="{field_name}" id="{field_id}" value="{value}" required="{required}" error="{error}">'

    class MockEnv:
        def get_template(self, template_name):
            return MockTemplate()

    # Mock schema with fields
    class MockField:
        def __init__(self, read_only=False, title=None, allow_null=False, allow_blank=False, has_default=False, format=None):
            self.read_only = read_only
            self.title = title
            self.allow_null = allow_null
            self.allow_blank = allow_blank
            self._has_default = has_default
            self.format = format

        def has_default(self):
            return self._has_default

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields

        def serialize(self, values):
            return values

        def validate_or_error(self, data):
            if data == {"username": "invalid"}:
                return None, {"username": "Invalid username"}
            return data, None

    # Test 1: Render fields with values and no errors
    fields = {
        "username": MockField(title="Username", allow_null=False, allow_blank=False, has_default=False),
        "email": MockField(title="Email", allow_null=True, allow_blank=False, has_default=False, format="email"),
    }
    schema = MockSchema(fields)
    env = MockEnv()
    form = Form(env=env, schema=schema, values={"username": "john", "email": "john@example.com"})
    form.validate({"username": "john", "email": "john@example.com"})
    result = form.render_fields()
    assert 'name="username"' in result
    assert 'value="john"' in result
    assert 'name="email"' in result
    assert 'value="john@example.com"' in result
    assert 'type="email"' in result
    assert 'error="None"' in result

    # Test 2: Render fields with errors
    form2 = Form(env=env, schema=schema, values={"username": "invalid"})
    form2.validate({"username": "invalid"})
    result2 = form2.render_fields()
    assert 'error="Invalid username"' in result2
    assert 'value="invalid"' in result2

    # Test 3: Render fields with read_only field (should be skipped)
    fields_with_readonly = {
        "username": MockField(title="Username"),
        "id": MockField(read_only=True),
    }
    schema3 = MockSchema(fields_with_readonly)
    form3 = Form(env=env, schema=schema3, values={"username": "test", "id": 123})
    form3.validate({"username": "test"})
    result3 = form3.render_fields()
    assert 'name="username"' in result3
    assert 'name="id"' not in result3

    # Test 4: Render fields with no values
    form4 = Form(env=env, schema=schema, values=None)
    form4.validate(None)
    result4 = form4.render_fields()
    assert 'value="None"' in result4

    # Test 5: Password field should have empty value
    fields_password = {
        "password": MockField(format="password"),
    }
    schema5 = MockSchema(fields_password)
    form5 = Form(env=env, schema=schema5, values={"password": "secret"})
    form5.validate({"password": "secret"})
    result5 = form5.render_fields()
    assert 'value=""' in result5
    assert 'type="password"' in result5


# LLM-generated content at query #14
#--------------------------

```python
def test_Form_input_type_for_field():
    # Mock environment for Form initialization
    class MockEnv:
        def get_template(self, name):
            return None

    # Mock schema for Form initialization
    class MockSchema:
        def __init__(self, fields=None):
            self.fields = fields or {}
        
        def serialize(self, values):
            return values or {}
        
        def validate_or_error(self, data):
            return (data, None)

    # Test 1: Field without format attribute returns "text"
    class FieldWithoutFormat:
        pass
    
    form = Form(env=MockEnv(), schema=MockSchema())
    field_without_format = FieldWithoutFormat()
    assert form.input_type_for_field(field_without_format) == "text"

    # Test 2: Field with format but not in FORMAT_TO_INPUTTYPE returns "text"
    class FieldWithUnknownFormat:
        format = "unknown_format"
    
    field_with_unknown_format = FieldWithUnknownFormat()
    assert form.input_type_for_field(field_with_unknown_format) == "text"

    # Test 3: Field with format in FORMAT_TO_INPUTTYPE returns correct input type
    test_cases = [
        ("color", "color"),
        ("datetime", "datetime-local"),
        ("date", "date"),
        ("email", "email"),
        ("hidden", "hidden"),
        ("month", "month"),
        ("number", "number"),
        ("password", "password"),
        ("range", "range"),
        ("search", "search"),
        ("tel", "tel"),
        ("text", "text"),
        ("time", "time"),
        ("url", "url"),
        ("week", "week"),
    ]
    
    for format_value, expected_input_type in test_cases:
        class FieldWithFormat:
            format = format_value
        
        field = FieldWithFormat()
        assert form.input_type_for_field(field) == expected_input_type

    # Test 4: Field with None format returns "text"
    class FieldWithNoneFormat:
        format = None
    
    field_with_none_format = FieldWithNoneFormat()
    assert form.input_type_for_field(field_with_none_format) == "text"

    # Test 5: Field with empty string format returns "text"
    class FieldWithEmptyFormat:
        format = ""
    
    field_with_empty_format = FieldWithEmptyFormat()
    assert form.input_type_for_field(field_with_empty_format) == "text"


# LLM-generated content at query #15
#--------------------------

```python
def test_Form___html__():
    import jinja2
    import markupsafe
    
    # Create a minimal schema for testing
    class TestSchema(Schema):
        name = String(title="Full Name")
        email = String(format="email")
    
    # Create a Jinja2 environment with a simple template
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
        }),
        autoescape=True
    )
    
    # Test 1: Basic __html__ method returns Markup object
    schema = TestSchema()
    form = Form(env=env, schema=schema, values={"name": "John", "email": "john@example.com"})
    html_output = form.__html__()
    
    assert isinstance(html_output, markupsafe.Markup)
    assert "name" in str(html_output)
    assert "email" in str(html_output)
    
    # Test 2: __html__ returns same content as render_fields()
    form2 = Form(env=env, schema=schema, values={"name": "Jane", "email": "jane@example.com"})
    render_fields_output = form2.render_fields()
    html_output2 = form2.__html__()
    
    assert str(html_output2) == render_fields_output
    
    # Test 3: __html__ escapes content properly
    env_with_escape = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
        }),
        autoescape=True
    )
    
    schema2 = TestSchema()
    form3 = Form(env=env_with_escape, schema=schema2, values={"name": "<script>alert('xss')</script>", "email": "test@example.com"})
    html_output3 = form3.__html__()
    
    assert isinstance(html_output3, markupsafe.Markup)
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in str(html_output3)
    assert "<script>" not in str(html_output3)
    
    # Test 4: __html__ works with empty values
    form4 = Form(env=env, schema=schema)
    html_output4 = form4.__html__()
    
    assert isinstance(html_output4, markupsafe.Markup)
    assert "name" in str(html_output4)
    assert "email" in str(html_output4)
    
    # Test 5: __html__ returns Markup that can be safely concatenated
    form5 = Form(env=env, schema=schema, values={"name": "Test", "email": "test@example.com"})
    html_output5 = form5.__html__()
    
    # This should not escape the already safe Markup content
    combined = markupsafe.Markup("<div>") + html_output5 + markupsafe.Markup("</div>")
    assert isinstance(combined, markupsafe.Markup)
    assert "<div>" in str(combined)
    assert "</div>" in str(combined)


# LLM-generated content at query #16
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user"])
        bio = fields.String(format="text")
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": """
        <input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">
        """,
        "forms/checkbox.html": """
        <input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>
        """,
        "forms/select.html": """
        <select name="{{ field_name }}">
            <option value="admin" {% if value == "admin" %}selected{% endif %}>admin</option>
            <option value="user" {% if value == "user" %}selected{% endif %}>user</option>
        </select>
        """,
        "forms/textarea.html": """
        <textarea name="{{ field_name }}">{{ value }}</textarea>
        """
    }))
    
    schema = TestSchema()
    form = Form(env=env, schema=schema, values={
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "active": True,
        "role": "admin",
        "bio": "Software developer"
    })
    
    result = str(form)
    
    assert '<input type="text" name="name" value="John Doe">' in result
    assert '<input type="email" name="email" value="john@example.com">' in result
    assert '<input type="text" name="age" value="30">' in result
    assert '<input type="checkbox" name="active" checked>' in result
    assert '<option value="admin" selected' in result
    assert '<textarea name="bio">Software developer</textarea>' in result
    
    form2 = Form(env=env, schema=schema, values=None)
    result2 = str(form2)
    
    assert '<input type="text" name="name" value="">' in result2
    assert '<input type="email" name="email" value="">' in result2
    assert '<input type="text" name="age" value="">' in result2
    assert '<input type="checkbox" name="active">' in result2
    assert '<textarea name="bio"></textarea>' in result2


# LLM-generated content at query #17
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user", "guest"])
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
        })
    )
    
    form = Form(env=env, schema=TestSchema())
    
    result = str(form)
    
    assert "name" in result
    assert "email" in result
    assert "age" in result
    assert "active" in result
    assert "role" in result
    
    form_with_values = Form(env=env, schema=TestSchema(), values={
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "active": True,
        "role": "admin"
    })
    
    result_with_values = str(form_with_values)
    assert "John Doe" in result_with_values
    assert "john@example.com" in result_with_values
    assert "admin" in result_with_values
    
    form.validate({"name": "", "email": "invalid"})
    result_with_errors = str(form)
    assert "name" in result_with_errors
    assert "email" in result_with_errors


# LLM-generated content at query #18
#--------------------------

```python
def test_Form_render_field():
    import jinja2
    from typesystem.fields import String, Boolean, Choice, Object
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(title="Full Name")
        email = String(format="email")
        password = String(format="password")
        bio = String(format="text")
        active = Boolean()
        role = Choice(choices=[("admin", "Admin"), ("user", "User")])
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": """
        <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" 
               value="{{ value }}" {% if required %}required{% endif %}>
        """,
        "forms/textarea.html": """
        <textarea id="{{ field_id }}" name="{{ field_name }}" 
                  {% if required %}required{% endif %}>{{ value }}</textarea>
        """,
        "forms/checkbox.html": """
        <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" 
               value="true" {% if value %}checked{% endif %}>
        """,
        "forms/select.html": """
        <select id="{{ field_id }}" name="{{ field_name }}" 
                {% if required %}required{% endif %}>
            <option value="">Select...</option>
            {% for choice_value, choice_label in field.choices %}
            <option value="{{ choice_value }}" 
                    {% if choice_value == value %}selected{% endif %}>
                {{ choice_label }}
            </option>
            {% endfor %}
        </select>
        """
    }))
    
    schema = TestSchema()
    form = Form(env=env, schema=schema)
    
    # Test String field with default text input
    result = form.render_field(field_name="name", field=schema.fields["name"], value="John")
    assert 'type="text"' in result
    assert 'id="name"' in result
    assert 'name="name"' in result
    assert 'value="John"' in result
    assert "required" in result
    
    # Test String field with email format
    result = form.render_field(field_name="email", field=schema.fields["email"], value="test@example.com")
    assert 'type="email"' in result
    assert 'id="email"' in result
    assert 'value="test@example.com"' in result
    
    # Test String field with password format (value should be empty)
    result = form.render_field(field_name="password", field=schema.fields["password"], value="secret123")
    assert 'type="password"' in result
    assert 'id="password"' in result
    assert 'value=""' in result
    
    # Test String field with text format (should use textarea)
    result = form.render_field(field_name="bio", field=schema.fields["bio"], value="My bio")
    assert "<textarea" in result
    assert 'id="bio"' in result
    assert "My bio</textarea>" in result
    
    # Test Boolean field (should use checkbox template)
    result = form.render_field(field_name="active", field=schema.fields["active"], value=True)
    assert 'type="checkbox"' in result
    assert 'id="active"' in result
    assert "checked" in result
    
    # Test Choice field (should use select template)
    result = form.render_field(field_name="role", field=schema.fields["role"], value="admin")
    assert "<select" in result
    assert 'id="role"' in result
    assert 'value="admin"' in result
    assert "selected" in result
    
    # Test field with error
    result = form.render_field(field_name="name", field=schema.fields["name"], 
                              value="John", error="Invalid name")
    assert 'type="text"' in result
    assert 'value="John"' in result
    
    # Test field with underscore in name (should be converted to hyphen in id)
    result = form.render_field(field_name="test_field", field=String(), value="test")
    assert 'id="test-field"' in result
    assert 'name="test_field"' in result
    
    # Test field with custom title
    field_with_title = String(title="Custom Label")
    result = form.render_field(field_name="custom", field=field_with_title, value="test")
    assert 'Custom Label' in result
    
    # Test field without title (should use field_name)
    field_without_title = String()
    result = form.render_field(field_name="untitled", field=field_without_title, value="test")
    assert 'untitled' in result
    
    # Test required field
    required_field = String(allow_null=False)
    result = form.render_field(field_name="required", field=required_field, value="test")
    assert "required" in result
    
    # Test non-required field (allow_null=True)
    optional_field = String(allow_null=True)
    result = form.render_field(field_name="optional", field=optional_field, value="test")
    assert "required" not in result
    
    # Test field with allow_blank
    blank_field = String(allow_blank=True)
    result = form.render_field(field_name="blank", field=blank_field, value="test")
    assert "required" not in result
    
    # Test field with default value (should not be required)
    default_field = String(default="default")
    result = form.render_field(field_name="with_default", field=default_field, value="custom")
    assert "required" not in result
    
    # Test unknown format (should default to text)
    unknown_format_field = String(format="unknown")
    result = form.render_field(field_name="unknown", field=unknown_format_field, value="test")
    assert 'type="text"' in result
    
    # Test all known formats from FORMAT_TO_INPUTTYPE
    for format_str, expected_type in Form.FORMAT_TO_INPUTTYPE.items():
        format_field = String(format=format_str)
        result = form.render_field(field_name=format_str, field=format_field, value="test")
        assert f'type="{expected_type}"' in result


# LLM-generated content at query #19
#--------------------------

```python
def test_Form_render_fields():
    from unittest.mock import Mock, patch, MagicMock
    import typesystem
    
    # Mock jinja2 environment
    mock_env = Mock()
    mock_template = Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a simple schema with various field types
    class TestSchema(typesystem.Schema):
        name = typesystem.String(title="Full Name", max_length=100)
        email = typesystem.String(format="email")
        age = typesystem.Integer(minimum=0, maximum=150)
        active = typesystem.Boolean(default=True)
        role = typesystem.Choice(choices=["admin", "user", "guest"])
        bio = typesystem.String(format="text")
        read_only_field = typesystem.String(read_only=True)
    
    schema = TestSchema()
    
    # Test 1: Render fields with no errors (using values)
    form = Form(env=mock_env, schema=schema, values={"name": "John", "email": "john@example.com"})
    
    # Mock template rendering
    mock_template.render.return_value = "<rendered_field>"
    
    # Call render_fields before validate (should use self.values)
    result = form.render_fields()
    
    # Should render all non-readonly fields
    assert mock_env.get_template.call_count == 6  # name, email, age, active, role, bio
    assert "read_only_field" not in [call[0][0] for call in mock_template.render.call_args_list]
    
    # Test 2: Render fields after validation with errors (should use self.data)
    form2 = Form(env=mock_env, schema=schema)
    form2.validate({"name": "", "email": "invalid"})
    
    # Reset mock calls
    mock_env.get_template.reset_mock()
    mock_template.render.reset_mock()
    
    result2 = form2.render_fields()
    
    # Should still render all non-readonly fields
    assert mock_env.get_template.call_count == 6
    
    # Test 3: Render fields with None values
    form3 = Form(env=mock_env, schema=schema, values=None)
    
    mock_env.get_template.reset_mock()
    mock_template.render.reset_mock()
    
    result3 = form3.render_fields()
    
    # Should render all fields with None values
    assert mock_env.get_template.call_count == 6
    
    # Test 4: Verify field rendering order matches schema fields
    form4 = Form(env=mock_env, schema=schema)
    
    mock_env.get_template.reset_mock()
    mock_template.render.reset_mock()
    
    result4 = form4.render_fields()
    
    # Check that fields are rendered in schema order
    call_args = [call[0][0] for call in mock_env.get_template.call_args_list]
    assert "forms/input.html" in call_args[0]  # name
    assert "forms/input.html" in call_args[1]  # email
    assert "forms/input.html" in call_args[2]  # age
    assert "forms/checkbox.html" in call_args[3]  # active
    assert "forms/select.html" in call_args[4]  # role
    assert "forms/textarea.html" in call_args[5]  # bio
    
    # Test 5: Render fields with empty values dict
    form5 = Form(env=mock_env, schema=schema, values={})
    
    mock_env.get_template.reset_mock()
    mock_template.render.reset_mock()
    
    result5 = form5.render_fields()
    
    # Should render all fields
    assert mock_env.get_template.call_count == 6
    
    # Test 6: Verify that read_only fields are skipped
    # Create schema with only read_only field
    class ReadOnlySchema(typesystem.Schema):
        read_only = typesystem.String(read_only=True)
        normal = typesystem.String()
    
    read_only_schema = ReadOnlySchema()
    form6 = Form(env=mock_env, schema=read_only_schema)
    
    mock_env.get_template.reset_mock()
    mock_template.render.reset_mock()
    
    result6 = form6.render_fields()
    
    # Should only render normal field
    assert mock_env.get_template.call_count == 1
    assert mock_template.render.call_count == 1


# LLM-generated content at query #20
#--------------------------

```python
def test_Form_render_fields():
    # Mock jinja2 environment and template
    class MockTemplate:
        def render(self, context):
            return f"rendered_{context['field_name']}"

    class MockEnv:
        def get_template(self, name):
            return MockTemplate()

    # Mock schema with fields
    class MockField:
        def __init__(self, read_only=False, title=None, allow_null=False, allow_blank=False, has_default=False, format=None):
            self.read_only = read_only
            self.title = title
            self.allow_null = allow_null
            self.allow_blank = allow_blank
            self._has_default = has_default
            self.format = format

        def has_default(self):
            return self._has_default

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields

        def serialize(self, values):
            return values

        def validate_or_error(self, data):
            return data, None

    # Test 1: Basic rendering with no errors
    fields = {
        "username": MockField(title="Username"),
        "email": MockField(title="Email", format="email"),
    }
    schema = MockSchema(fields)
    env = MockEnv()
    form = Form(env=env, schema=schema, values={"username": "john", "email": "john@example.com"})
    form.validate({"username": "john", "email": "john@example.com"})
    result = form.render_fields()
    assert "rendered_username" in result
    assert "rendered_email" in result
    assert result == "rendered_usernamerendered_email"

    # Test 2: With errors
    class MockSchemaWithErrors:
        def __init__(self, fields):
            self.fields = fields

        def serialize(self, values):
            return values

        def validate_or_error(self, data):
            return None, {"username": "Invalid username", "email": "Invalid email"}

    schema_with_errors = MockSchemaWithErrors(fields)
    form_with_errors = Form(env=env, schema=schema_with_errors)
    form_with_errors.validate({"username": "", "email": "invalid"})
    result = form_with_errors.render_fields()
    assert "rendered_username" in result
    assert "rendered_email" in result

    # Test 3: Skip read_only fields
    fields_with_readonly = {
        "id": MockField(read_only=True),
        "username": MockField(),
        "created_at": MockField(read_only=True),
    }
    schema_readonly = MockSchema(fields_with_readonly)
    form_readonly = Form(env=env, schema=schema_readonly, values={"id": 1, "username": "john", "created_at": "2023-01-01"})
    form_readonly.validate({"username": "john"})
    result = form_readonly.render_fields()
    assert "rendered_id" not in result
    assert "rendered_username" in result
    assert "rendered_created_at" not in result
    assert result == "rendered_username"

    # Test 4: Empty values
    form_empty = Form(env=env, schema=schema)
    form_empty.validate({})
    result = form_empty.render_fields()
    assert "rendered_username" in result
    assert "rendered_email" in result

    # Test 5: Password field with empty string value
    password_field = MockField(format="password")
    fields_password = {"password": password_field}
    schema_password = MockSchema(fields_password)
    form_password = Form(env=env, schema=schema_password, values={"password": "secret"})
    form_password.validate({"password": "secret"})
    # The render_field method should convert password value to empty string
    result = form_password.render_fields()
    assert "rendered_password" in result


# LLM-generated content at query #21
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user"])
        bio = fields.String(format="text")
        read_only_field = fields.String(read_only=True)
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
                <input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">
            """,
            "forms/textarea.html": """
                <textarea name="{{ field_name }}">{{ value }}</textarea>
            """,
            "forms/checkbox.html": """
                <input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>
            """,
            "forms/select.html": """
                <select name="{{ field_name }}">
                    <option value="{{ value }}">{{ value }}</option>
                </select>
            """,
        })
    )
    
    form = Form(env=env, schema=TestSchema)
    result = str(form)
    
    assert "name" in result
    assert "email" in result
    assert "age" in result
    assert "active" in result
    assert "role" in result
    assert "bio" in result
    assert "read_only_field" not in result
    
    form_with_values = Form(
        env=env, 
        schema=TestSchema, 
        values={"name": "John", "email": "john@example.com", "active": True}
    )
    result_with_values = str(form_with_values)
    
    assert "John" in result_with_values
    assert "john@example.com" in result_with_values
    
    form_with_errors = Form(env=env, schema=TestSchema)
    form_with_errors.validate({"email": "invalid"})
    result_with_errors = str(form_with_errors)
    
    assert "invalid" in result_with_errors


# LLM-generated content at query #22
#--------------------------

```python
def test_Form_template_for_field():
    # Mock jinja2 environment
    mock_env = type("MockEnv", (), {"get_template": lambda x: None})()
    
    # Mock schema with fields
    mock_schema = type("MockSchema", (), {"fields": {}})()
    
    # Test with Choice field
    form = Form(env=mock_env, schema=mock_schema)
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    assert form.template_for_field(choice_field) == "forms/select.html"
    
    # Test with Boolean field
    boolean_field = Boolean()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"
    
    # Test with String field with text format
    string_text_field = String(format="text")
    assert form.template_for_field(string_text_field) == "forms/textarea.html"
    
    # Test with String field with other format
    string_email_field = String(format="email")
    assert form.template_for_field(string_email_field) == "forms/input.html"
    
    # Test with regular String field (no format)
    string_field = String()
    assert form.template_for_field(string_field) == "forms/input.html"
    
    # Test with other field type
    number_field = type("MockField", (), {})()
    assert form.template_for_field(number_field) == "forms/input.html"
    
    # Test that Object field raises assertion error
    object_field = Object(properties={})
    try:
        form.template_for_field(object_field)
        assert False, "Should have raised AssertionError for Object field"
    except AssertionError as e:
        assert "Forms do not support rendering Object fields" in str(e)


# LLM-generated content at query #23
#--------------------------

```python
def test_Form_template_for_field():
    # Mock jinja2 environment
    mock_env = type("MockEnv", (), {"get_template": lambda self, name: type("MockTemplate", (), {"render": lambda self, ctx: ""})()})()
    
    # Mock schema with fields
    mock_schema = type("MockSchema", (), {"fields": {}})()
    
    # Test String field with format="text"
    string_field_text = type("StringField", (), {"format": "text"})()
    form = Form(env=mock_env, schema=mock_schema)
    assert form.template_for_field(string_field_text) == "forms/textarea.html"
    
    # Test String field with other format
    string_field_other = type("StringField", (), {"format": "email"})()
    assert form.template_for_field(string_field_other) == "forms/input.html"
    
    # Test String field without format attribute
    string_field_no_format = type("StringField", (), {})()
    assert form.template_for_field(string_field_no_format) == "forms/input.html"
    
    # Test Choice field
    choice_field = type("ChoiceField", (), {})()
    assert form.template_for_field(choice_field) == "forms/select.html"
    
    # Test Boolean field
    boolean_field = type("BooleanField", (), {})()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"
    
    # Test Object field - should raise AssertionError
    object_field = type("ObjectField", (), {})()
    try:
        form.template_for_field(object_field)
        assert False, "Should have raised AssertionError for Object field"
    except AssertionError as e:
        assert str(e) == "Forms do not support rendering Object fields"
    
    # Test generic Field (not String, Choice, or Boolean)
    generic_field = type("GenericField", (), {})()
    assert form.template_for_field(generic_field) == "forms/input.html"


# LLM-generated content at query #24
#--------------------------

```python
def test_Form_render_field():
    import jinja2
    from typesystem.fields import String, Choice, Boolean, Object
    from typesystem.schemas import Schema
    
    # Create a minimal Jinja2 environment with test templates
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %}>',
        "forms/textarea.html": '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>',
        "forms/select.html": '<select name="{{ field_name }}" id="{{ field_id }}"><option value="option1">Option 1</option></select>',
    }))
    
    # Test 1: Basic String field with text format
    class TestSchema(Schema):
        name = String(title="Full Name")
    
    schema = TestSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="name", field=schema.fields["name"], value="John")
    assert 'type="text"' in result
    assert 'name="name"' in result
    assert 'id="name"' in result
    assert 'value="John"' in result
    
    # Test 2: String field with email format
    class EmailSchema(Schema):
        email = String(format="email")
    
    schema = EmailSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="email", field=schema.fields["email"], value="test@example.com")
    assert 'type="email"' in result
    
    # Test 3: String field with password format
    class PasswordSchema(Schema):
        password = String(format="password")
    
    schema = PasswordSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="password", field=schema.fields["password"], value="secret")
    assert 'type="password"' in result
    assert 'value=""' in result  # Password fields should have empty value
    
    # Test 4: String field with text format (textarea)
    class TextareaSchema(Schema):
        description = String(format="text")
    
    schema = TextareaSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="description", field=schema.fields["description"], value="Some text")
    assert "<textarea" in result
    assert "Some text" in result
    
    # Test 5: Boolean field (checkbox)
    class BooleanSchema(Schema):
        active = Boolean()
    
    schema = BooleanSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="active", field=schema.fields["active"], value=True)
    assert 'type="checkbox"' in result
    assert "checked" in result
    
    # Test 6: Choice field (select)
    class ChoiceSchema(Schema):
        color = Choice(choices=[("red", "Red"), ("blue", "Blue")])
    
    schema = ChoiceSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="color", field=schema.fields["color"])
    assert "<select" in result
    
    # Test 7: Field with error
    result = form.render_field(field_name="name", field=schema.fields["name"], value="", error="This field is required")
    assert 'type="text"' in result
    
    # Test 8: Field with custom format
    class CustomFormatSchema(Schema):
        birthday = String(format="date")
    
    schema = CustomFormatSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="birthday", field=schema.fields["birthday"])
    assert 'type="date"' in result
    
    # Test 9: Field with underscore in name
    class UnderscoreSchema(Schema):
        first_name = String()
    
    schema = UnderscoreSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="first_name", field=schema.fields["first_name"])
    assert 'id="first-name"' in result
    
    # Test 10: Required field
    class RequiredSchema(Schema):
        required_field = String(allow_null=False)
        optional_field = String(allow_null=True)
    
    schema = RequiredSchema()
    form = Form(env=env, schema=schema)
    
    result1 = form.render_field(field_name="required_field", field=schema.fields["required_field"])
    assert "required" in result1
    
    result2 = form.render_field(field_name="optional_field", field=schema.fields["optional_field"])
    assert "required" not in result2
    
    # Test 11: Field with title
    class TitleSchema(Schema):
        field_with_title = String(title="Custom Title")
    
    schema = TitleSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="field_with_title", field=schema.fields["field_with_title"])
    # The label should use the custom title
    
    # Test 12: Unknown format falls back to text
    class UnknownFormatSchema(Schema):
        unknown = String(format="unknown_format")
    
    schema = UnknownFormatSchema()
    form = Form(env=env, schema=schema)
    
    result = form.render_field(field_name="unknown", field=schema.fields["unknown"])
    assert 'type="text"' in result


# LLM-generated content at query #25
#--------------------------

```python
def test_Form_template_for_field():
    # Mock environment for Form initialization
    mock_env = type('MockEnv', (), {'get_template': lambda self, name: type('MockTemplate', (), {'render': lambda self, ctx: ''})()})()

    # Mock schema with fields
    mock_schema = type('MockSchema', (), {'fields': {}})()

    form = Form(env=mock_env, schema=mock_schema)

    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    assert form.template_for_field(choice_field) == "forms/select.html"

    # Test Boolean field
    boolean_field = Boolean()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"

    # Test String field with format="text"
    string_text_field = String(format="text")
    assert form.template_for_field(string_text_field) == "forms/textarea.html"

    # Test String field with other format
    string_email_field = String(format="email")
    assert form.template_for_field(string_email_field) == "forms/input.html"

    # Test generic Field
    generic_field = Field()
    assert form.template_for_field(generic_field) == "forms/input.html"

    # Test Object field should raise assertion
    object_field = Object(properties={})
    try:
        form.template_for_field(object_field)
        assert False, "Should have raised AssertionError for Object field"
    except AssertionError as e:
        assert str(e) == "Forms do not support rendering Object fields"


# LLM-generated content at query #26
#--------------------------

```python
def test_Form_render_field():
    # Mock jinja2 environment and template
    class MockTemplate:
        def render(self, context):
            return f"rendered: {context['field_name']}"

    class MockEnv:
        def get_template(self, name):
            return MockTemplate()

    # Mock schema with fields
    class MockSchema:
        def __init__(self):
            self.fields = {}

    # Test with String field
    env = MockEnv()
    schema = MockSchema()
    form = Form(env=env, schema=schema, values={})
    
    string_field = String(title="Test Field", format="email")
    result = form.render_field(
        field_name="test_field",
        field=string_field,
        value="test@example.com",
        error=None
    )
    assert "rendered: test_field" in result
    
    # Test with password field
    password_field = String(format="password")
    result = form.render_field(
        field_name="password",
        field=password_field,
        value="secret",
        error="Invalid password"
    )
    assert "rendered: password" in result
    # Password value should be empty string in template context
    
    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    result = form.render_field(
        field_name="choice_field",
        field=choice_field,
        value="a",
        error=None
    )
    assert "rendered: choice_field" in result
    
    # Test with Boolean field
    boolean_field = Boolean(title="Agree to terms")
    result = form.render_field(
        field_name="agree",
        field=boolean_field,
        value=True,
        error="Must agree"
    )
    assert "rendered: agree" in result
    
    # Test with required field
    required_field = String(title="Required Field")
    result = form.render_field(
        field_name="required_field",
        field=required_field,
        value=None,
        error=None
    )
    assert "rendered: required_field" in result
    
    # Test field_id generation
    field_with_underscores = String(title="Test Field")
    result = form.render_field(
        field_name="field_with_underscores",
        field=field_with_underscores,
        value="test",
        error=None
    )
    assert "rendered: field_with_underscores" in result
    
    # Test with textarea (String with format="text")
    textarea_field = String(format="text", title="Description")
    result = form.render_field(
        field_name="description",
        field=textarea_field,
        value="Some text",
        error=None
    )
    assert "rendered: description" in result
    
    # Test with different input types
    for format_name, input_type in Form.FORMAT_TO_INPUTTYPE.items():
        field = String(format=format_name)
        result = form.render_field(
            field_name=f"field_{format_name}",
            field=field,
            value="test",
            error=None
        )
        assert "rendered: field_" in result
    
    # Test with field without title
    no_title_field = String()
    result = form.render_field(
        field_name="no_title",
        field=no_title_field,
        value="test",
        error=None
    )
    assert "rendered: no_title" in result


# LLM-generated content at query #27
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user"])
        bio = fields.String(format="text")
        read_only_field = fields.String(read_only=True)
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": """
                <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" 
                       value="{{ value }}" {% if required %}required{% endif %}>
            """,
            "forms/textarea.html": """
                <textarea name="{{ field_name }}" id="{{ field_id }}" 
                          {% if required %}required{% endif %}>{{ value }}</textarea>
            """,
            "forms/checkbox.html": """
                <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" 
                       {% if value %}checked{% endif %} {% if required %}required{% endif %}>
            """,
            "forms/select.html": """
                <select name="{{ field_name }}" id="{{ field_id }}" 
                        {% if required %}required{% endif %}>
                    {% for choice in field.choices %}
                        <option value="{{ choice.value }}" 
                                {% if choice.value == value %}selected{% endif %}>
                            {{ choice.text }}
                        </option>
                    {% endfor %}
                </select>
            """
        })
    )
    
    form = Form(env=env, schema=TestSchema)
    
    result = str(form)
    
    assert "name" in result
    assert "email" in result
    assert "age" in result
    assert "active" in result
    assert "role" in result
    assert "bio" in result
    assert "read_only_field" not in result
    
    assert 'type="text"' in result
    assert 'type="email"' in result
    assert 'type="number"' in result
    assert 'type="checkbox"' in result
    assert "<textarea" in result
    assert "<select" in result
    
    assert 'id="name"' in result
    assert 'id="email"' in result
    assert 'id="age"' in result
    assert 'id="active"' in result
    assert 'id="role"' in result
    assert 'id="bio"' in result
    
    assert 'name="name"' in result
    assert 'name="email"' in result
    assert 'name="age"' in result
    assert 'name="active"' in result
    assert 'name="role"' in result
    assert 'name="bio"' in result


# LLM-generated content at query #28
#--------------------------

```python
def test_Form___html__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
        })
    )
    
    form = Form(env=env, schema=TestSchema())
    html_output = form.__html__()
    
    assert isinstance(html_output, markupsafe.Markup)
    assert str(html_output) == form.render_fields()
    assert "name" in str(html_output)
    assert "email" in str(html_output)
    
    form_with_values = Form(env=env, schema=TestSchema(), values={"name": "John", "email": "john@example.com"})
    html_with_values = form_with_values.__html__()
    
    assert isinstance(html_with_values, markupsafe.Markup)
    assert "John" in str(html_with_values)
    assert "john@example.com" in str(html_with_values)
    
    form.validate({"name": "Jane", "email": "invalid-email"})
    html_with_errors = form.__html__()
    
    assert isinstance(html_with_errors, markupsafe.Markup)
    assert "Jane" in str(html_with_errors)


# LLM-generated content at query #29
#--------------------------

```python
def test_Form___html__():
    import jinja2
    from typesystem import Schema, fields
    from typesystem.schemas import Schema as BaseSchema

    class TestSchema(BaseSchema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")

    env = jinja2.Environment(
        loader=jinja2.DictLoader(
            {
                "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
                "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
                "forms/select.html": '<select name="{{ field_name }}"></select>',
                "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}">',
            }
        ),
        autoescape=True,
    )

    form = Form(env=env, schema=TestSchema, values={"name": "John", "email": "john@example.com"})
    
    html_output = form.__html__()
    assert isinstance(html_output, jinja2.Markup)
    assert "name" in str(html_output)
    assert "email" in str(html_output)
    assert '<input type="text"' in str(html_output) or 'value="John"' in str(html_output)
    assert 'type="email"' in str(html_output)

    form2 = Form(env=env, schema=TestSchema, values=None)
    html_output2 = form2.__html__()
    assert isinstance(html_output2, jinja2.Markup)
    assert "name" in str(html_output2)
    assert "email" in str(html_output2)

    class EmptySchema(BaseSchema):
        pass

    form3 = Form(env=env, schema=EmptySchema, values=None)
    html_output3 = form3.__html__()
    assert isinstance(html_output3, jinja2.Markup)
    assert str(html_output3) == ""


# LLM-generated content at query #30
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, String, Integer, Boolean, Choice
    
    class TestSchema(Schema):
        name = String(title="Full Name")
        age = Integer()
        active = Boolean()
        status = Choice(choices=["active", "inactive"])
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>',
            "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        })
    )
    
    form = Form(env=env, schema=TestSchema)
    result = str(form)
    
    assert '<input type="text" name="name" value="">' in result
    assert '<input type="text" name="age" value="">' in result
    assert '<input type="checkbox" name="active">' in result
    assert '<select name="status">' in result
    
    form_with_values = Form(env=env, schema=TestSchema, values={"name": "John", "active": True})
    result_with_values = str(form_with_values)
    
    assert 'value="John"' in result_with_values
    assert 'checked' in result_with_values
    
    form.validate({"name": "Jane", "age": 25, "active": False, "status": "active"})
    result_validated = str(form)
    
    assert 'value="Jane"' in result_validated
    assert 'value="25"' in result_validated
    assert 'checked' not in result_validated
    assert 'value="active"' in result_validated
    
    form_with_errors = Form(env=env, schema=TestSchema)
    form_with_errors.validate({"age": "invalid"})
    result_with_errors = str(form_with_errors)
    
    assert 'value="invalid"' in result_with_errors


# LLM-generated content at query #31
#--------------------------

```python
def test_Form_input_type_for_field():
    # Mock environment and schema for Form instantiation
    class MockEnv:
        def get_template(self, name):
            return None

    class MockSchema:
        fields = {}

    form = Form(env=MockEnv(), schema=MockSchema())

    # Test default case when field has no format attribute
    class FieldWithoutFormat:
        pass

    field_without_format = FieldWithoutFormat()
    assert form.input_type_for_field(field_without_format) == "text"

    # Test when field has format but not in FORMAT_TO_INPUTTYPE
    class FieldWithUnknownFormat:
        format = "unknown"

    field_unknown_format = FieldWithUnknownFormat()
    assert form.input_type_for_field(field_unknown_format) == "text"

    # Test all mappings in FORMAT_TO_INPUTTYPE
    for format_value, expected_input_type in Form.FORMAT_TO_INPUTTYPE.items():
        class FieldWithFormat:
            format = format_value

        field = FieldWithFormat()
        assert form.input_type_for_field(field) == expected_input_type

    # Test with String field that has format attribute
    class StringField:
        format = "email"

    string_field = StringField()
    assert form.input_type_for_field(string_field) == "email"

    # Test with field that has empty string format
    class FieldWithEmptyFormat:
        format = ""

    field_empty_format = FieldWithEmptyFormat()
    assert form.input_type_for_field(field_empty_format) == "text"

    # Test with field that has None format
    class FieldWithNoneFormat:
        format = None

    field_none_format = FieldWithNoneFormat()
    assert form.input_type_for_field(field_none_format) == "text"


# LLM-generated content at query #32
#--------------------------

```python
def test_Form_input_type_for_field():
    # Mock environment for Form initialization
    mock_env = type("MockEnv", (), {"get_template": lambda x: None})()
    
    # Mock schema for Form initialization
    mock_schema = type("MockSchema", (), {"fields": {}, "serialize": lambda x: x, "validate_or_error": lambda x: (x, None)})()
    
    form = Form(env=mock_env, schema=mock_schema)
    
    # Test field with no format attribute
    field_without_format = type("MockField", (), {})()
    assert form.input_type_for_field(field_without_format) == "text"
    
    # Test field with format attribute but not in FORMAT_TO_INPUTTYPE
    field_with_unknown_format = type("MockField", (), {"format": "unknown"})()
    assert form.input_type_for_field(field_with_unknown_format) == "text"
    
    # Test all formats in FORMAT_TO_INPUTTYPE
    for format_str, expected_type in Form.FORMAT_TO_INPUTTYPE.items():
        field = type("MockField", (), {"format": format_str})()
        assert form.input_type_for_field(field) == expected_type
    
    # Test field with empty string format
    field_empty_format = type("MockField", (), {"format": ""})()
    assert form.input_type_for_field(field_empty_format) == "text"
    
    # Test field with None format
    field_none_format = type("MockField", (), {"format": None})()
    assert form.input_type_for_field(field_none_format) == "text"


# LLM-generated content at query #33
#--------------------------

```python
def test_Form___str__():
    import jinja2
    from typesystem import Schema, fields
    
    class TestSchema(Schema):
        name = fields.String(title="Full Name")
        email = fields.String(format="email")
        age = fields.Integer()
        active = fields.Boolean()
        role = fields.Choice(choices=["admin", "user", "guest"])
    
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
            "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>',
        })
    )
    
    form = Form(env=env, schema=TestSchema)
    
    result = str(form)
    
    assert '<input type="text" name="name" value="">' in result
    assert '<input type="email" name="email" value="">' in result
    assert '<input type="text" name="age" value="">' in result
    assert '<input type="checkbox" name="active" >' in result
    assert '<select name="role"><option value=""></option></select>' in result
    
    form_with_values = Form(
        env=env, 
        schema=TestSchema, 
        values={"name": "John", "email": "john@test.com", "age": 30, "active": True, "role": "admin"}
    )
    
    result_with_values = str(form_with_values)
    
    assert '<input type="text" name="name" value="John">' in result_with_values
    assert '<input type="email" name="email" value="john@test.com">' in result_with_values
    assert '<input type="text" name="age" value="30">' in result_with_values
    assert '<input type="checkbox" name="active" checked>' in result_with_values
    assert '<select name="role"><option value="admin">admin</option></select>' in result_with_values


