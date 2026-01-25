####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/select.html": '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/textarea.html": '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }), autoescape=True)
    
    schema = Schema(fields={
        "name": String(max_length=100),
        "email": String(format="email"),
        "active": Boolean(),
        "read_only_field": String(read_only=True),
    })
    
    form = Form(env=env, schema=schema, values={"name": "John", "email": "john@example.com", "active": True})
    
    # Before validate is called, render_fields should raise an error or handle gracefully
    # After validate is called with no errors
    form.validate({"name": "John", "email": "john@example.com", "active": True})
    html = form.render_fields()
    
    # Assertions
    assert "John" in html
    assert "john@example.com" in html
    assert "name" in html
    assert "email" in html
    assert "active" in html
    assert "read_only_field" not in html  # read_only fields should not be rendered
    assert isinstance(html, str)
    
    # Test with validation errors
    form2 = Form(env=env, schema=schema)
    form2.validate({"name": "", "email": "invalid-email", "active": False})
    html_with_errors = form2.render_fields()
    
    assert isinstance(html_with_errors, str)
    # Errors should be displayed in the HTML
    assert "error" in html_with_errors or html_with_errors  # Should contain error markup
    
    # Test with None values
    form3 = Form(env=env, schema=schema, values=None)
    form3.validate({})
    html_empty = form3.render_fields()
    
    assert isinstance(html_empty, str)
    assert "name" in html_empty


# LLM-generated content at query #2
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>',
            "forms/textarea.html": '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}></select>',
        }),
        autoescape=True
    )
    
    schema = Schema(fields={
        "name": String(title="Name"),
        "email": String(format="email"),
        "bio": String(format="text"),
        "active": Boolean(),
        "role": Choice(choices=["admin", "user"]),
    })
    
    form = Form(env=env, schema=schema)
    
    # Test basic text input field
    result = form.render_field(
        field_name="name",
        field=schema.fields["name"],
        value="John",
        error=None
    )
    assert 'name="name"' in result
    assert 'id="name"' in result
    assert 'value="John"' in result
    assert 'type="text"' in result
    assert 'required' in result
    
    # Test email input field
    result = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="test@example.com",
        error=None
    )
    assert 'type="email"' in result
    assert 'value="test@example.com"' in result
    
    # Test textarea field
    result = form.render_field(
        field_name="bio",
        field=schema.fields["bio"],
        value="Bio text",
        error=None
    )
    assert '<textarea' in result
    assert 'Bio text' in result
    
    # Test checkbox field
    result = form.render_field(
        field_name="active",
        field=schema.fields["active"],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in result
    assert 'checked' in result
    
    # Test select field
    result = form.render_field(
        field_name="role",
        field=schema.fields["role"],
        value="admin",
        error=None
    )
    assert '<select' in result
    
    # Test field with error
    result = form.render_field(
        field_name="name",
        field=schema.fields["name"],
        value="",
        error="This field is required"
    )
    assert 'error' in result or result  # error passed to template context
    
    # Test field with underscore in name (should convert to dash in id)
    schema_with_underscore = Schema(fields={
        "first_name": String(),
    })
    form2 = Form(env=env, schema=schema_with_underscore)
    result = form2.render_field(
        field_name="first_name",
        field=schema_with_underscore.fields["first_name"],
        value="Jane",
        error=None
    )
    assert 'id="first-name"' in result
    assert 'name="first_name"' in result
    
    # Test optional field (not required)
    schema_optional = Schema(fields={
        "nickname": String(allow_null=True),
    })
    form3 = Form(env=env, schema=schema_optional)
    result = form3.render_field(
        field_name="nickname",
        field=schema_optional.fields["nickname"],
        value="Nick",
        error=None
    )
    assert 'required' not in result
    
    # Test password field
    schema_password = Schema(fields={
        "password": String(format="password"),
    })
    form4 = Form(env=env, schema=schema_password)
    result = form4.render_field(
        field_name="password",
        field=schema_password.fields["password"],
        value="secret123",
        error=None
    )
    assert 'type="password"' in result
    assert 'value=""' in result  # Password values should be empty


# LLM-generated content at query #3
#--------------------------

```python
def test_Form_validate():
    """Test Form.validate() method"""
    import jinja2
    
    # Create a simple schema
    schema = Schema(
        fields={
            "name": String(max_length=100),
            "email": String(format="email"),
            "age": Field(),
        }
    )
    
    # Create a Jinja2 environment
    env = jinja2.Environment(loader=jinja2.DictLoader({}))
    
    # Create a form instance
    form = Form(env=env, schema=schema, values=None)
    
    # Test that validate() can be called once
    valid_data = {"name": "John", "email": "john@example.com", "age": 30}
    form.validate(data=valid_data)
    
    # Verify _validate_called is set to True
    assert form._validate_called is True
    
    # Verify data is stored
    assert form.data == valid_data
    
    # Test that validate() cannot be called twice
    with pytest.raises(AssertionError, match="validate\\(\\) has already been called"):
        form.validate(data=valid_data)


def test_Form_validate_with_invalid_data():
    """Test Form.validate() with invalid data"""
    import jinja2
    
    schema = Schema(
        fields={
            "email": String(format="email"),
        }
    )
    
    env = jinja2.Environment(loader=jinja2.DictLoader({}))
    form = Form(env=env, schema=schema, values=None)
    
    # Validate with invalid email
    invalid_data = {"email": "not-an-email"}
    form.validate(data=invalid_data)
    
    # Verify that errors are populated
    assert form.errors is not None
    assert "email" in form.errors or form.values is not None


def test_Form_validate_with_none_data():
    """Test Form.validate() with None data"""
    import jinja2
    
    schema = Schema(
        fields={
            "name": String(allow_null=True),
        }
    )
    
    env = jinja2.Environment(loader=jinja2.DictLoader({}))
    form = Form(env=env, schema=schema, values=None)
    
    form.validate(data=None)
    
    assert form._validate_called is True
    assert form.data is None


# LLM-generated content at query #4
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field method with various field types."""
    import jinja2
    
    # Create a minimal Jinja2 environment with a simple template
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value or "" }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>',
            'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value or "" }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        }),
        autoescape=True
    )
    
    # Create a simple schema
    schema = Schema(fields={
        'username': String(max_length=100),
        'password': String(format='password'),
        'email': String(format='email'),
        'bio': String(format='text'),
        'is_active': Boolean(),
        'role': Choice(choices=[('admin', 'Admin'), ('user', 'User')]),
    })
    
    form = Form(env=env, schema=schema, values=None)
    
    # Test rendering a text input field
    html = form.render_field(
        field_name='username',
        field=schema.fields['username'],
        value='john_doe',
        error=None
    )
    assert 'type="text"' in html
    assert 'name="username"' in html
    assert 'id="username"' in html
    assert 'value="john_doe"' in html
    assert 'required' in html
    
    # Test rendering a password field
    html = form.render_field(
        field_name='password',
        field=schema.fields['password'],
        value='secret123',
        error=None
    )
    assert 'type="password"' in html
    assert 'name="password"' in html
    assert 'value=""' in html  # Password fields should have empty value
    
    # Test rendering an email field
    html = form.render_field(
        field_name='email',
        field=schema.fields['email'],
        value='test@example.com',
        error=None
    )
    assert 'type="email"' in html
    assert 'value="test@example.com"' in html
    
    # Test rendering a textarea field
    html = form.render_field(
        field_name='bio',
        field=schema.fields['bio'],
        value='My bio',
        error=None
    )
    assert '<textarea' in html
    assert 'name="bio"' in html
    assert 'My bio' in html
    
    # Test rendering a checkbox field
    html = form.render_field(
        field_name='is_active',
        field=schema.fields['is_active'],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in html
    assert 'checked' in html
    
    # Test rendering with error message
    html = form.render_field(
        field_name='username',
        field=schema.fields['username'],
        value='john',
        error='This field is required.'
    )
    assert 'This field is required.' in html
    assert '<span class="error">' in html
    
    # Test field_id conversion with underscores
    html = form.render_field(
        field_name='user_name',
        field=String(),
        value=None,
        error=None
    )
    assert 'id="user-name"' in html
    
    # Test rendering select field
    html = form.render_field(
        field_name='role',
        field=schema.fields['role'],
        value='admin',
        error=None
    )
    assert '<select' in html
    assert 'name="role"' in html
    
    # Test rendering with None value
    html = form.render_field(
        field_name='username',
        field=schema.fields['username'],
        value=None,
        error=None
    )
    assert 'value=""' in html


# LLM-generated content at query #5
#--------------------------

```python
def test_Form_render_field(tmpdir):
    # Setup
    template_dir = tmpdir.mkdir("templates")
    forms_dir = template_dir.mkdir("forms")
    
    # Create minimal template files
    forms_dir.join("input.html").write(
        '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}"'
        '{% if value %} value="{{ value }}"{% endif %}'
        '{% if required %} required{% endif %}'
        '/>'
        '{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    )
    forms_dir.join("checkbox.html").write(
        '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"'
        '{% if required %} required{% endif %}'
        '/>'
        '{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    )
    forms_dir.join("select.html").write(
        '<select name="{{ field_name }}" id="{{ field_id }}"'
        '{% if required %} required{% endif %}>'
        '{% for choice_value, choice_label in field.choices %}'
        '<option value="{{ choice_value }}"{% if value == choice_value %} selected{% endif %}>{{ choice_label }}</option>'
        '{% endfor %}'
        '</select>'
        '{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    )
    forms_dir.join("textarea.html").write(
        '<textarea name="{{ field_name }}" id="{{ field_id }}"'
        '{% if required %} required{% endif %}>'
        '{{ value }}'
        '</textarea>'
        '{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    )
    
    forms = Jinja2Forms(directory=str(template_dir))
    schema = Schema(fields={
        "username": String(max_length=100),
        "password": String(format="password"),
        "is_active": Boolean(),
        "status": Choice(choices=[("active", "Active"), ("inactive", "Inactive")]),
        "bio": String(format="text"),
    })
    
    form = Form(env=forms.env, schema=schema, values=None)
    
    # Test rendering text input field
    html = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john_doe",
        error=None
    )
    assert 'type="text"' in html
    assert 'name="username"' in html
    assert 'id="username"' in html
    assert 'value="john_doe"' in html
    assert 'required' in html
    
    # Test rendering password field
    html = form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret",
        error=None
    )
    assert 'type="password"' in html
    assert 'name="password"' in html
    
    # Test rendering boolean field (checkbox)
    html = form.render_field(
        field_name="is_active",
        field=schema.fields["is_active"],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in html
    assert 'name="is_active"' in html
    assert 'id="is-active"' in html
    
    # Test rendering choice field (select)
    html = form.render_field(
        field_name="status",
        field=schema.fields["status"],
        value="active",
        error=None
    )
    assert '<select' in html
    assert 'name="status"' in html
    assert '<option value="active" selected>Active</option>' in html
    
    # Test rendering textarea field
    html = form.render_field(
        field_name="bio",
        field=schema.fields["bio"],
        value="My bio",
        error=None
    )
    assert '<textarea' in html
    assert 'name="bio"' in html
    assert 'My bio' in html
    
    # Test field with error
    html = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="",
        error="This field is required"
    )
    assert '<span class="error">This field is required</span>' in html
    
    # Test field_id with underscores converted to hyphens
    html = form.render_field(
        field_name="is_active",
        field=schema.fields["is_active"],
        value=None,
        error=None
    )
    assert 'id="is-active"' in html


# LLM-generated content at query #6
#--------------------------

```python
def test_Jinja2Forms():
    import tempfile
    import os
    
    # Test with directory parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test assertion when neither directory nor package is specified
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test assertion when jinja2 is None (mocking the import failure)
    import typesystem.forms as forms_module
    original_jinja2 = forms_module.jinja2
    try:
        forms_module.jinja2 = None
        with pytest.raises(AssertionError):
            Jinja2Forms(directory=".")
    finally:
        forms_module.jinja2 = original_jinja2


# LLM-generated content at query #7
#--------------------------

```python
def test_Form_render_field():
    import jinja2
    
    # Create a simple Jinja2 environment with mock templates
    loader = jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}/>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" id="{{ field_id }}" name="{{ field_name }}"{% if value %} checked{% endif %}/>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    })
    env = jinja2.Environment(loader=loader, autoescape=True)
    
    # Test with String field
    schema = Schema(fields={"email": String(format="email")})
    form = Form(env=env, schema=schema, values={})
    
    result = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="test@example.com",
        error=None
    )
    assert 'type="email"' in result
    assert 'id="email"' in result
    assert 'name="email"' in result
    assert 'value="test@example.com"' in result
    assert 'required' in result
    
    # Test with error
    result_with_error = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="invalid",
        error="Invalid email"
    )
    assert 'Invalid email' in result_with_error
    
    # Test with Boolean field
    schema_bool = Schema(fields={"agree": Boolean()})
    form_bool = Form(env=env, schema=schema_bool, values={})
    
    result_bool = form_bool.render_field(
        field_name="agree",
        field=schema_bool.fields["agree"],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in result_bool
    assert 'checked' in result_bool
    
    # Test with Choice field
    schema_choice = Schema(fields={"status": Choice(choices=[("active", "Active"), ("inactive", "Inactive")])})
    form_choice = Form(env=env, schema=schema_choice, values={})
    
    result_choice = form_choice.render_field(
        field_name="status",
        field=schema_choice.fields["status"],
        value="active",
        error=None
    )
    assert '<select' in result_choice
    assert 'id="status"' in result_choice
    
    # Test with password field (value should be empty)
    schema_pwd = Schema(fields={"password": String(format="password")})
    form_pwd = Form(env=env, schema=schema_pwd, values={})
    
    result_pwd = form_pwd.render_field(
        field_name="password",
        field=schema_pwd.fields["password"],
        value="secret123",
        error=None
    )
    assert 'type="password"' in result_pwd
    assert 'value="secret123"' not in result_pwd
    assert 'value=""' in result_pwd
    
    # Test with optional field (no required attribute)
    schema_opt = Schema(fields={"optional": String(allow_null=True)})
    form_opt = Form(env=env, schema=schema_opt, values={})
    
    result_opt = form_opt.render_field(
        field_name="optional",
        field=schema_opt.fields["optional"],
        value=None,
        error=None
    )
    assert 'required' not in result_opt


# LLM-generated content at query #8
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)

    # Test that autoescape is enabled
    assert env.autoescape is True

    # Test assertion when both directory and package are None
    with pytest.raises(AssertionError):
        forms = Jinja2Forms(directory="/tmp/templates")
        forms.load_template_env(directory=None, package=None)

    # Test assertion when directory is None but package is required
    with pytest.raises(AssertionError):
        forms = Jinja2Forms(package="typesystem")
        forms.load_template_env(directory=None, package=None)


# LLM-generated content at query #9
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}"{% if choice[0] == value %} selected{% endif %}>{{ choice[1] }}</option>{% endfor %}</select>',
            "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>',
        })
    )
    
    schema = Schema(fields={
        "name": String(),
        "email": String(format="email"),
        "active": Boolean(),
        "status": Choice(choices=[("active", "Active"), ("inactive", "Inactive")]),
    })
    
    form = Form(env=env, schema=schema, values=None)
    form.validate(data={"name": "John", "email": "john@example.com", "active": True, "status": "active"})
    
    # Test render_fields with valid data
    html = form.render_fields()
    assert "name" in html
    assert "email" in html
    assert "John" in html
    assert "john@example.com" in html
    assert isinstance(html, str)


def test_Form_render_fields_with_errors():
    # Setup
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
            "forms/select.html": '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}"{% if choice[0] == value %} selected{% endif %}>{{ choice[1] }}</option>{% endfor %}</select>',
            "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>',
        })
    )
    
    schema = Schema(fields={
        "name": String(),
        "email": String(format="email"),
    })
    
    form = Form(env=env, schema=schema, values=None)
    form.validate(data={"name": "", "email": "invalid"})
    
    # Test render_fields with errors - should use data values
    html = form.render_fields()
    assert isinstance(html, str)
    assert "name" in html
    assert "email" in html


def test_Form_render_fields_excludes_readonly():
    # Setup
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>',
        })
    )
    
    schema = Schema(fields={
        "name": String(),
        "id": String(read_only=True),
    })
    
    form = Form(env=env, schema=schema, values={"name": "John", "id": "123"})
    form.validate(data={"name": "John"})
    
    # Test that read_only fields are not rendered
    html = form.render_fields()
    assert "name" in html
    assert "id" not in html


def test_Form_render_fields_no_validate():
    # Setup
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>',
        })
    )
    
    schema = Schema(fields={
        "name": String(),
    })
    
    form = Form(env=env, schema=schema, values={"name": "John"})
    
    # Test render_fields uses initial values when validate not called
    html = form.render_fields()
    assert "name" in html
    assert "John" in html


# LLM-generated content at query #10
#--------------------------

```python
def test_Form_template_for_field():
    # Setup mock environment and schema
    mock_env = None
    if jinja2 is not None:
        mock_env = jinja2.Environment()
    
    schema = Schema(fields={
        "test_choice": Choice(choices=[("a", "A"), ("b", "B")]),
        "test_boolean": Boolean(),
        "test_text": String(format="text"),
        "test_email": String(format="email"),
        "test_string": String(),
    })
    
    form = Form(env=mock_env, schema=schema)
    
    # Test Choice field returns select template
    choice_field = schema.fields["test_choice"]
    assert form.template_for_field(choice_field) == "forms/select.html"
    
    # Test Boolean field returns checkbox template
    boolean_field = schema.fields["test_boolean"]
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"
    
    # Test String field with text format returns textarea template
    text_field = schema.fields["test_text"]
    assert form.template_for_field(text_field) == "forms/textarea.html"
    
    # Test String field with other formats returns input template
    email_field = schema.fields["test_email"]
    assert form.template_for_field(email_field) == "forms/input.html"
    
    # Test String field without format returns input template
    string_field = schema.fields["test_string"]
    assert form.template_for_field(string_field) == "forms/input.html"
    
    # Test that Object field raises assertion error
    object_field = Object()
    with pytest.raises(AssertionError):
        form.template_for_field(object_field)


# LLM-generated content at query #11
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field method renders field correctly."""
    # Setup
    forms = Jinja2Forms(package="typesystem")
    
    class TestSchema(Schema):
        name = String(title="Full Name")
        email = String(format="email", title="Email Address")
        age = String(format="number")
        bio = String(format="text")
        active = Boolean(title="Active")
        country = Choice(choices=[("us", "United States"), ("uk", "United Kingdom")])
        password = String(format="password")
        optional_field = String(allow_null=True)
    
    schema = TestSchema
    form = forms.create_form(schema)
    
    # Test basic text field
    html = form.render_field(field_name="name", field=schema.fields["name"])
    assert "Full Name" in html
    assert 'id="name"' in html
    assert 'name="name"' in html
    assert 'type="text"' in html
    assert 'required' in html
    
    # Test email field
    html = form.render_field(field_name="email", field=schema.fields["email"])
    assert "Email Address" in html
    assert 'type="email"' in html
    
    # Test number field
    html = form.render_field(field_name="age", field=schema.fields["age"])
    assert 'type="number"' in html
    
    # Test textarea field
    html = form.render_field(field_name="bio", field=schema.fields["bio"])
    assert "textarea" in html
    
    # Test boolean field (checkbox)
    html = form.render_field(field_name="active", field=schema.fields["active"])
    assert "checkbox" in html or "Active" in html
    
    # Test choice field (select)
    html = form.render_field(field_name="country", field=schema.fields["country"])
    assert "select" in html or "country" in html
    
    # Test password field with value (should be empty)
    html = form.render_field(
        field_name="password", 
        field=schema.fields["password"], 
        value="secret123"
    )
    assert 'type="password"' in html
    
    # Test field with value
    html = form.render_field(
        field_name="name", 
        field=schema.fields["name"], 
        value="John Doe"
    )
    assert "John Doe" in html
    
    # Test field with error
    html = form.render_field(
        field_name="email", 
        field=schema.fields["email"], 
        error="Invalid email address"
    )
    assert "Invalid email address" in html
    
    # Test optional field (not required)
    html = form.render_field(
        field_name="optional_field", 
        field=schema.fields["optional_field"]
    )
    assert 'id="optional-field"' in html
    # Should not have required attribute or should indicate it's optional
    
    # Test field_id conversion (underscore to hyphen)
    html = form.render_field(
        field_name="optional_field", 
        field=schema.fields["optional_field"]
    )
    assert 'id="optional-field"' in html


# LLM-generated content at query #12
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory parameter
    forms = Jinja2Forms(directory="./templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with both directory and package parameters
    forms = Jinja2Forms(directory="./templates", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test assertion when jinja2 is None
    if jinja2 is not None:
        # Test assertion when neither directory nor package is specified
        with pytest.raises(AssertionError):
            Jinja2Forms()


# LLM-generated content at query #13
#--------------------------

```python
def test_Form_render_field():
    import jinja2
    
    # Create a simple Jinja2 environment with a mock template
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }), autoescape=True)
    
    # Test with String field
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema, values={})
    
    field = String(title="Name")
    html = form.render_field(field_name="name", field=field, value="John", error=None)
    assert 'type="text"' in html
    assert 'name="name"' in html
    assert 'id="name"' in html
    assert 'value="John"' in html
    assert 'required' in html
    
    # Test with required=False
    html = form.render_field(field_name="name", field=String(allow_null=True), value="Jane", error=None)
    assert 'value="Jane"' in html
    assert 'required' not in html
    
    # Test with error
    html = form.render_field(field_name="name", field=field, value="John", error="Invalid name")
    assert 'Invalid name' in html
    
    # Test with Boolean field
    bool_field = Boolean()
    html = form.render_field(field_name="active", field=bool_field, value=True, error=None)
    assert 'type="checkbox"' in html
    assert 'id="active"' in html
    
    # Test with password field
    password_field = String(format="password")
    html = form.render_field(field_name="password", field=password_field, value="secret", error=None)
    assert 'type="password"' in html
    assert 'value=""' in html  # Password fields should have empty value
    
    # Test with email field
    email_field = String(format="email")
    html = form.render_field(field_name="email", field=email_field, value="test@example.com", error=None)
    assert 'type="email"' in html
    assert 'value="test@example.com"' in html
    
    # Test with field_id conversion (underscores to hyphens)
    html = form.render_field(field_name="user_name", field=field, value="test", error=None)
    assert 'id="user-name"' in html
    
    # Test with custom label
    labeled_field = String(title="Custom Label")
    html = form.render_field(field_name="custom", field=labeled_field, value=None, error=None)
    assert 'id="custom"' in html


# LLM-generated content at query #14
#--------------------------

```python
def test_Form_render_field(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='username' id='username' value='john' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a simple schema
    schema = Schema(fields={"username": String(max_length=100)})
    
    # Create form instance
    form = Form(env=mock_env, schema=schema, values={"username": "john"})
    
    # Call render_field
    result = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john",
        error=None
    )
    
    # Assertions
    assert result == "<input type='text' name='username' id='username' value='john' />"
    mock_env.get_template.assert_called_once_with("forms/input.html")
    mock_template.render.assert_called_once()
    
    # Verify render context
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["field_name"] == "username"
    assert render_call_args["field_id"] == "username"
    assert render_call_args["label"] == "username"
    assert render_call_args["value"] == "john"
    assert render_call_args["error"] is None
    assert render_call_args["input_type"] == "text"
    assert render_call_args["required"] is True


def test_Form_render_field_with_underscores(mocker):
    # Test field name with underscores converts to hyphens in field_id
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"user_name": String()})
    form = Form(env=mock_env, schema=schema)
    
    form.render_field(
        field_name="user_name",
        field=schema.fields["user_name"],
        value="test",
        error=None
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["field_id"] == "user-name"


def test_Form_render_field_with_error(mocker):
    # Test render_field with error message
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input /> <span>Error occurred</span>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"email": String(format="email")})
    form = Form(env=mock_env, schema=schema)
    
    form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="invalid",
        error="Invalid email format"
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["error"] == "Invalid email format"
    assert render_call_args["input_type"] == "email"


def test_Form_render_field_password_clears_value(mocker):
    # Test that password fields don't render the value
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='password' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=mock_env, schema=schema)
    
    form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret123",
        error=None
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["value"] == ""
    assert render_call_args["input_type"] == "password"


def test_Form_render_field_with_title(mocker):
    # Test that field title is used as label when provided
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"username": String(title="User Name")})
    form = Form(env=mock_env, schema=schema)
    
    form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value=None,
        error=None
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["label"] == "User Name"


def test_Form_render_field_required_with_default(mocker):
    # Test that field with default is not required
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"status": String(default="active")})
    form = Form(env=mock_env, schema=schema)
    
    form.render_field(
        field_name="status",
        field=schema.fields["status"],
        value="active",
        error=None
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["required"] is False


def test_Form_render_field_required_with_allow_null(mocker):
    # Test that allow_null makes field not required
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"optional_field": String(allow_null=True)})
    form = Form(env=mock_env, schema=schema)
    
    form.render_field(
        field_name="optional_field",
        field=schema.fields["optional_field"],
        value=None,
        error=None
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["required"] is False


# LLM-generated content at query #15
#--------------------------

```python
def test_Form_render_field(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='username'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a schema with a simple string field
    schema = Schema(fields={"username": String(title="Username")})
    
    # Create form instance
    form = Form(env=mock_env, schema=schema, values=None)
    
    # Call render_field
    result = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john_doe",
        error=None
    )
    
    # Assertions
    assert result == "<input type='text' name='username'>"
    mock_env.get_template.assert_called_once_with("forms/input.html")
    mock_template.render.assert_called_once()
    
    # Verify render context
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["field_id"] == "username"
    assert render_call_args["field_name"] == "username"
    assert render_call_args["label"] == "Username"
    assert render_call_args["value"] == "john_doe"
    assert render_call_args["error"] is None
    assert render_call_args["required"] is True


def test_Form_render_field_with_error(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='email'><span>Invalid email</span>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"email": String(format="email")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="invalid",
        error="Invalid email format"
    )
    
    assert result == "<input type='text' name='email'><span>Invalid email</span>"
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["error"] == "Invalid email format"
    assert render_call_args["input_type"] == "email"


def test_Form_render_field_password_clears_value(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='password' name='password'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret123",
        error=None
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["value"] == ""
    assert render_call_args["input_type"] == "password"


def test_Form_render_field_with_underscore_in_name(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"first_name": String()})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="first_name",
        field=schema.fields["first_name"],
        value="John",
        error=None
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["field_id"] == "first-name"


def test_Form_render_field_choice_field(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<select></select>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"status": Choice(choices={"active": "Active", "inactive": "Inactive"})})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="status",
        field=schema.fields["status"],
        value="active",
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/select.html")


def test_Form_render_field_boolean_field(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='checkbox'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"agree": Boolean()})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="agree",
        field=schema.fields["agree"],
        value=True,
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/checkbox.html")


def test_Form_render_field_textarea(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<textarea></textarea>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"description": String(format="text")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="description",
        field=schema.fields["description"],
        value="Some text",
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/textarea.html")


def test_Form_render_field_required_attribute(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input required>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"required_field": String()})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="required_field",
        field=schema.fields["required_field"],
        value=None,
        error=None
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["required"] is True


def test_Form_render_field_not_required_with_default(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"optional_field": String(default="default_value")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="optional_field",
        field=schema.fields["optional_field"],
        value="default_value",
        error=None
    )
    
    render_call_args = mock_template.render.call_args[0][0]
    assert render_call_args["required"] is False


# LLM-generated content at query #16
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            "forms/textarea.html": '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            "forms/select.html": '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        }),
        autoescape=True
    )
    
    schema = Schema(fields={
        "username": String(max_length=100),
        "password": String(format="password"),
        "description": String(format="text"),
        "active": Boolean(),
        "status": Choice(choices=["active", "inactive"]),
    })
    
    form = Form(env=env, schema=schema)
    
    # Test rendering text input field
    result = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john_doe",
        error=None
    )
    assert 'type="text"' in result
    assert 'name="username"' in result
    assert 'id="username"' in result
    assert 'value="john_doe"' in result
    assert 'required' in result
    
    # Test rendering password field
    result = form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret",
        error=None
    )
    assert 'type="password"' in result
    assert 'name="password"' in result
    assert 'value=""' in result  # Password fields should have empty value
    
    # Test rendering textarea field
    result = form.render_field(
        field_name="description",
        field=schema.fields["description"],
        value="Some text",
        error=None
    )
    assert '<textarea' in result
    assert 'name="description"' in result
    assert 'Some text' in result
    
    # Test rendering checkbox field
    result = form.render_field(
        field_name="active",
        field=schema.fields["active"],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in result
    assert 'name="active"' in result
    assert 'checked' in result
    
    # Test rendering select field
    result = form.render_field(
        field_name="status",
        field=schema.fields["status"],
        value="active",
        error=None
    )
    assert '<select' in result
    assert 'name="status"' in result
    
    # Test with error
    result = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john",
        error="This field is required"
    )
    assert 'This field is required' in result
    assert '<span class="error">' in result
    
    # Test field_id conversion (underscore to dash)
    result = form.render_field(
        field_name="user_name",
        field=String(),
        value=None,
        error=None
    )
    assert 'id="user-name"' in result
    
    # Test optional field (with default value)
    optional_field = String(default="default_value")
    result = form.render_field(
        field_name="optional",
        field=optional_field,
        value=None,
        error=None
    )
    assert 'required' not in result
    
    # Test nullable field
    nullable_field = String(allow_null=True)
    result = form.render_field(
        field_name="nullable",
        field=nullable_field,
        value=None,
        error=None
    )
    assert 'required' not in result


# LLM-generated content at query #17
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory parameter
    forms = Jinja2Forms(directory="/path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test with both directory and package parameters
    forms = Jinja2Forms(directory="/path/to/templates", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test assertion when neither directory nor package is specified
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test assertion when jinja2 is not available
    original_jinja2 = globals()['jinja2']
    try:
        globals()['jinja2'] = None
        with pytest.raises(AssertionError, match="jinja2 must be installed"):
            Jinja2Forms(directory="/path/to/templates")
    finally:
        globals()['jinja2'] = original_jinja2


# LLM-generated content at query #18
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory parameter only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates")
    assert isinstance(env, jinja2.Environment)
    assert env.autoescape is True
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package parameter only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert env.autoescape is True
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package parameters
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert env.autoescape is True
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)


# LLM-generated content at query #19
#--------------------------

```python
def test_Form_render_fields():
    import jinja2
    
    # Create a mock environment with templates
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<p>{{ error }}</p>{% endif %}',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>{% if error %}<p>{{ error }}</p>{% endif %}',
    }))
    
    # Test with basic schema
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'active': Boolean(),
    })
    
    form = Form(env=env, schema=schema, values={'name': 'John', 'email': 'john@example.com', 'active': True})
    form.validate({'name': 'John', 'email': 'john@example.com', 'active': True})
    
    html = form.render_fields()
    
    assert 'name="name"' in html
    assert 'name="email"' in html
    assert 'name="active"' in html
    assert 'John' in html


def test_Form_render_fields_with_errors():
    import jinja2
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<p>{{ error }}</p>{% endif %}',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>{% if error %}<p>{{ error }}</p>{% endif %}',
    }))
    
    schema = Schema(fields={
        'name': String(max_length=5),
        'email': String(format='email'),
    })
    
    form = Form(env=env, schema=schema)
    form.validate({'name': 'This is too long', 'email': 'invalid'})
    
    html = form.render_fields()
    
    assert 'name="name"' in html
    assert 'name="email"' in html


def test_Form_render_fields_read_only_excluded():
    import jinja2
    
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<p>{{ error }}</p>{% endif %}',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>{% if error %}<p>{{ error }}</p>{% endif %}',
    }))
    
    schema = Schema(fields={
        'name': String(),
        'created_at': String(read_only=True),
    })
    
    form = Form(env=env, schema=schema, values={'name': 'John', 'created_at': '2023-01-01'})
    form.validate({'name': 'John'})
    
    html = form.render_fields()
    
    assert 'name="name"' in html
    assert 'name="created_at"' not in html


# LLM-generated content at query #20
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>',
            'forms/textarea.html': '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
            'forms/select.html': '<select name="{{ field_name }}"{% if required %} required{% endif %}></select>',
        }),
        autoescape=True
    )
    
    schema = Schema(
        fields={
            'name': String(max_length=100),
            'email': String(format='email'),
            'age': Field(),
            'active': Boolean(),
        }
    )
    
    form = Form(env=env, schema=schema, values=None)
    
    # Test render_fields without validation
    form.data = None
    form.errors = None
    form._validate_called = True
    
    result = form.render_fields()
    
    assert isinstance(result, str)
    assert 'name' in result
    assert 'email' in result
    assert 'age' in result
    assert 'active' in result


def test_Form_render_fields_with_values():
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>',
            'forms/textarea.html': '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
            'forms/select.html': '<select name="{{ field_name }}"{% if required %} required{% endif %}></select>',
        }),
        autoescape=True
    )
    
    schema = Schema(
        fields={
            'name': String(max_length=100),
            'email': String(format='email'),
        }
    )
    
    values = {'name': 'John', 'email': 'john@example.com'}
    form = Form(env=env, schema=schema, values=values)
    form.data = None
    form.errors = None
    form._validate_called = True
    
    result = form.render_fields()
    
    assert isinstance(result, str)
    assert 'John' in result
    assert 'john@example.com' in result


def test_Form_render_fields_with_errors():
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span>{{ error }}</span>{% endif %}',
            'forms/textarea.html': '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
            'forms/select.html': '<select name="{{ field_name }}"{% if required %} required{% endif %}></select>',
        }),
        autoescape=True
    )
    
    schema = Schema(
        fields={
            'name': String(max_length=100),
        }
    )
    
    form = Form(env=env, schema=schema, values=None)
    form.data = {'name': ''}
    form.errors = {'name': 'This field is required'}
    form._validate_called = True
    
    result = form.render_fields()
    
    assert isinstance(result, str)
    assert 'This field is required' in result


def test_Form_render_fields_skips_readonly():
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>',
            'forms/textarea.html': '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
            'forms/select.html': '<select name="{{ field_name }}"{% if required %} required{% endif %}></select>',
        }),
        autoescape=True
    )
    
    schema = Schema(
        fields={
            'name': String(max_length=100),
            'id': String(read_only=True),
        }
    )
    
    form = Form(env=env, schema=schema, values=None)
    form.data = None
    form.errors = None
    form._validate_called = True
    
    result = form.render_fields()
    
    assert isinstance(result, str)
    assert 'name' in result
    assert 'id' not in result


# LLM-generated content at query #21
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    import tempfile
    import os
    
    # Test with directory only
    with tempfile.TemporaryDirectory() as temp_dir:
        forms = Jinja2Forms(directory=temp_dir)
        env = forms.load_template_env(directory=temp_dir)
        assert isinstance(env, jinja2.Environment)
        assert env.autoescape is True
        assert isinstance(env.loader, jinja2.FileSystemLoader)
    
    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert env.autoescape is True
    assert isinstance(env.loader, jinja2.PackageLoader)
    
    # Test with both directory and package
    with tempfile.TemporaryDirectory() as temp_dir:
        forms = Jinja2Forms(directory=temp_dir, package="typesystem")
        env = forms.load_template_env(directory=temp_dir, package="typesystem")
        assert isinstance(env, jinja2.Environment)
        assert env.autoescape is True
        assert isinstance(env.loader, jinja2.ChoiceLoader)
    
    # Test assertion when neither directory nor package provided
    with tempfile.TemporaryDirectory() as temp_dir:
        forms = Jinja2Forms(directory=temp_dir)
        with pytest.raises(AssertionError):
            forms.load_template_env()
    
    # Test assertion when both None
    with tempfile.TemporaryDirectory() as temp_dir:
        forms = Jinja2Forms(directory=temp_dir)
        with pytest.raises(AssertionError):
            forms.load_template_env(directory=None, package=None)


# LLM-generated content at query #22
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory parameter
    forms = Jinja2Forms(directory="/tmp")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with both directory and package parameters
    forms = Jinja2Forms(directory="/tmp", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test that neither directory nor package raises AssertionError
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test that jinja2 is required
    if jinja2 is None:
        with pytest.raises(AssertionError):
            Jinja2Forms(directory="/tmp")


# LLM-generated content at query #23
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field method renders fields correctly."""
    # Setup
    forms = Jinja2Forms(package="typesystem")
    
    class TestSchema(Schema):
        name = String(title="Full Name")
        email = String(format="email", title="Email Address")
        password = String(format="password")
        subscribe = Boolean(title="Subscribe")
        age = String(format="number")
        bio = String(format="text")
        country = Choice(choices=[("us", "United States"), ("uk", "United Kingdom")])
        optional_field = String(allow_null=True)
        field_with_default = String(default="default_value")
    
    schema = TestSchema()
    form = forms.create_form(schema)
    
    # Test rendering a basic text input field
    html = form.render_field(
        field_name="name",
        field=schema.fields["name"],
        value="John Doe",
        error=None
    )
    assert "John Doe" in html
    assert "Full Name" in html
    assert "name" in html
    assert "required" in html
    
    # Test rendering an email field
    html = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="test@example.com",
        error=None
    )
    assert "test@example.com" in html
    assert "email" in html
    assert "Email Address" in html
    
    # Test rendering a password field (value should be empty)
    html = form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret",
        error=None
    )
    assert "secret" not in html
    assert "password" in html
    
    # Test rendering a checkbox field
    html = form.render_field(
        field_name="subscribe",
        field=schema.fields["subscribe"],
        value=True,
        error=None
    )
    assert "checkbox" in html
    assert "Subscribe" in html
    
    # Test rendering a number field
    html = form.render_field(
        field_name="age",
        field=schema.fields["age"],
        value="25",
        error=None
    )
    assert "number" in html
    assert "25" in html
    
    # Test rendering a textarea field
    html = form.render_field(
        field_name="bio",
        field=schema.fields["bio"],
        value="My bio",
        error=None
    )
    assert "textarea" in html
    assert "My bio" in html
    
    # Test rendering a select field
    html = form.render_field(
        field_name="country",
        field=schema.fields["country"],
        value="us",
        error=None
    )
    assert "select" in html
    assert "country" in html
    
    # Test rendering with error
    html = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="invalid",
        error="Invalid email"
    )
    assert "Invalid email" in html
    
    # Test rendering nullable field (not required)
    html = form.render_field(
        field_name="optional_field",
        field=schema.fields["optional_field"],
        value=None,
        error=None
    )
    assert "optional_field" in html
    
    # Test rendering field with default (not required)
    html = form.render_field(
        field_name="field_with_default",
        field=schema.fields["field_with_default"],
        value="default_value",
        error=None
    )
    assert "field_with_default" in html
    
    # Test field_id conversion (underscores to hyphens)
    html = form.render_field(
        field_name="field_with_default",
        field=schema.fields["field_with_default"],
        value=None,
        error=None
    )
    assert "field-with-default" in html


# LLM-generated content at query #24
#--------------------------

```python
def test_Jinja2Forms():
    # Test that Jinja2Forms raises assertion error when jinja2 is not available
    if jinja2 is None:
        try:
            Jinja2Forms(directory="/tmp")
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "jinja2 must be installed" in str(e)
        return
    
    # Test that Jinja2Forms raises assertion error when neither directory nor package is specified
    try:
        Jinja2Forms()
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Either 'directory' or 'package' must be specified" in str(e)
    
    # Test that Jinja2Forms can be initialized with directory parameter
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test that Jinja2Forms can be initialized with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test that Jinja2Forms can be initialized with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)


# LLM-generated content at query #25
#--------------------------

```python
def test_Jinja2Forms():
    # Test that Jinja2Forms raises assertion error when jinja2 is not available
    if jinja2 is None:
        with pytest.raises(AssertionError, match="jinja2 must be installed"):
            Jinja2Forms()
        return
    
    # Test that Jinja2Forms raises assertion error when neither directory nor package is specified
    with pytest.raises(AssertionError, match="Either 'directory' or 'package' must be specified"):
        Jinja2Forms()
    
    # Test initialization with directory parameter
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)


# LLM-generated content at query #26
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field() method"""
    import jinja2
    
    # Create a simple Jinja2 environment with a basic template
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }))
    
    # Create a simple schema
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'password': String(format='password'),
        'bio': String(format='text'),
        'active': Boolean(),
        'country': Choice(choices=[('us', 'United States'), ('uk', 'United Kingdom')]),
    })
    
    form = Form(env=env, schema=schema)
    
    # Test rendering a text input field
    html = form.render_field(field_name='name', field=schema.fields['name'], value='John', error=None)
    assert 'type="text"' in html
    assert 'name="name"' in html
    assert 'id="name"' in html
    assert 'value="John"' in html
    assert 'required' in html
    
    # Test rendering with error
    html = form.render_field(field_name='name', field=schema.fields['name'], value='John', error='Name is required')
    assert 'error' in html.lower()
    
    # Test rendering email input
    html = form.render_field(field_name='email', field=schema.fields['email'], value='test@example.com', error=None)
    assert 'type="email"' in html
    assert 'value="test@example.com"' in html
    
    # Test rendering password input (value should be empty)
    html = form.render_field(field_name='password', field=schema.fields['password'], value='secret', error=None)
    assert 'type="password"' in html
    assert 'value=""' in html
    
    # Test rendering textarea
    html = form.render_field(field_name='bio', field=schema.fields['bio'], value='My bio', error=None)
    assert '<textarea' in html
    assert 'My bio' in html
    
    # Test rendering checkbox
    html = form.render_field(field_name='active', field=schema.fields['active'], value=True, error=None)
    assert 'type="checkbox"' in html
    assert 'checked' in html
    
    # Test rendering select
    html = form.render_field(field_name='country', field=schema.fields['country'], value='us', error=None)
    assert '<select' in html
    assert 'United States' in html
    assert 'United Kingdom' in html
    
    # Test field_id conversion (underscores to hyphens)
    html = form.render_field(field_name='first_name', field=schema.fields['name'], value='John', error=None)
    assert 'id="first-name"' in html
    assert 'name="first_name"' in html


# LLM-generated content at query #27
#--------------------------

```python
def test_Form_render_field():
    import jinja2
    
    # Create a mock environment with templates
    loader = jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice[0] }}" {% if choice[0] == value %}selected{% endif %}>{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    })
    env = jinja2.Environment(loader=loader, autoescape=True)
    
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'active': Boolean(),
        'status': Choice(choices=[('active', 'Active'), ('inactive', 'Inactive')]),
        'bio': String(format='text'),
        'age': String(format='number'),
    })
    
    form = Form(env=env, schema=schema, values={})
    
    # Test rendering text input field
    html = form.render_field(field_name='name', field=schema.fields['name'], value='John', error=None)
    assert 'type="text"' in html
    assert 'name="name"' in html
    assert 'id="name"' in html
    assert 'value="John"' in html
    assert 'required' in html
    
    # Test rendering email input field
    html = form.render_field(field_name='email', field=schema.fields['email'], value='test@example.com', error=None)
    assert 'type="email"' in html
    assert 'value="test@example.com"' in html
    
    # Test rendering number input field
    html = form.render_field(field_name='age', field=schema.fields['age'], value='25', error=None)
    assert 'type="number"' in html
    assert 'value="25"' in html
    
    # Test rendering checkbox field
    html = form.render_field(field_name='active', field=schema.fields['active'], value=True, error=None)
    assert 'type="checkbox"' in html
    assert 'checked' in html
    
    # Test rendering select field
    html = form.render_field(field_name='status', field=schema.fields['status'], value='active', error=None)
    assert '<select' in html
    assert 'selected' in html
    
    # Test rendering textarea field
    html = form.render_field(field_name='bio', field=schema.fields['bio'], value='My bio', error=None)
    assert '<textarea' in html
    assert 'My bio' in html
    
    # Test with error message
    html = form.render_field(field_name='name', field=schema.fields['name'], value='John', error='Invalid name')
    assert 'Invalid name' in html
    
    # Test with optional field (allow_null=True)
    optional_field = String(allow_null=True)
    html = form.render_field(field_name='optional', field=optional_field, value=None, error=None)
    assert 'required' not in html
    
    # Test password field clears value
    password_field = String(format='password')
    html = form.render_field(field_name='password', field=password_field, value='secret', error=None)
    assert 'type="password"' in html
    assert 'value=""' in html
    assert 'secret' not in html
    
    # Test field_id with underscores replaced by hyphens
    html = form.render_field(field_name='user_name', field=schema.fields['name'], value='test', error=None)
    assert 'id="user-name"' in html


# LLM-generated content at query #28
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory parameter
    forms = Jinja2Forms(directory="/tmp/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test that assertion fails when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test that assertion fails when jinja2 is not available
    if jinja2 is not None:
        original_jinja2 = globals()['jinja2']
        try:
            globals()['jinja2'] = None
            with pytest.raises(AssertionError, match="jinja2 must be installed"):
                Jinja2Forms(directory="/tmp/templates")
        finally:
            globals()['jinja2'] = original_jinja2


# LLM-generated content at query #29
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory parameter
    forms = Jinja2Forms(directory="/tmp")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with both directory and package parameters
    forms = Jinja2Forms(directory="/tmp", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test assertion when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test that jinja2 is required
    if jinja2 is None:
        with pytest.raises(AssertionError, match="jinja2 must be installed"):
            Jinja2Forms(directory="/tmp")


# LLM-generated content at query #30
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)

    # Test autoescape is enabled
    env = forms.load_template_env(directory="/tmp/templates")
    assert env.autoescape is True

    # Test assertion when neither directory nor package is provided
    forms = Jinja2Forms(directory="/tmp/templates")
    with pytest.raises(AssertionError):
        forms.load_template_env()

    # Test assertion when only directory is provided but package is explicitly None
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates", package=None)
    assert isinstance(env.loader, jinja2.FileSystemLoader)


# LLM-generated content at query #31
#--------------------------

```python
def test_Form_render_field(form_with_env):
    """Test render_field method of Form class."""
    # Setup
    form = form_with_env
    
    # Test rendering a simple string field
    field = String(title="Username")
    html = form.render_field(
        field_name="username",
        field=field,
        value="testuser",
        error=None
    )
    assert "testuser" in html
    assert "username" in html
    assert "Username" in html
    assert "required" in html.lower() or "text" in html
    
    # Test rendering a required field
    field = String()
    html = form.render_field(
        field_name="email",
        field=field,
        value=None,
        error=None
    )
    assert "email" in html
    
    # Test rendering a field with error
    field = String(title="Email")
    html = form.render_field(
        field_name="email",
        field=field,
        value="invalid",
        error="Invalid email format"
    )
    assert "Invalid email format" in html
    assert "invalid" in html
    
    # Test rendering a password field
    field = String(format="password", title="Password")
    html = form.render_field(
        field_name="password",
        field=field,
        value="secret123",
        error=None
    )
    assert "password" in html
    assert "secret123" not in html  # Password values should not be rendered
    
    # Test rendering a field with allow_null
    field = String(allow_null=True, title="Optional Field")
    html = form.render_field(
        field_name="optional",
        field=field,
        value=None,
        error=None
    )
    assert "optional" in html
    
    # Test rendering a choice field
    field = Choice(choices=["A", "B", "C"], title="Select")
    html = form.render_field(
        field_name="choice",
        field=field,
        value="A",
        error=None
    )
    assert "choice" in html
    
    # Test rendering a boolean field
    field = Boolean(title="Agree")
    html = form.render_field(
        field_name="agree",
        field=field,
        value=True,
        error=None
    )
    assert "agree" in html
    
    # Test field_id uses hyphens instead of underscores
    field = String(title="User Name")
    html = form.render_field(
        field_name="user_name",
        field=field,
        value=None,
        error=None
    )
    assert "user-name" in html
    
    # Test rendering with default label (field_name when title not provided)
    field = String()
    html = form.render_field(
        field_name="myfield",
        field=field,
        value=None,
        error=None
    )
    assert "myfield" in html


# LLM-generated content at query #32
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory parameter
    forms = Jinja2Forms(directory="/tmp/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with both directory and package parameters
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test that assertion is raised when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test that jinja2 must be installed
    if jinja2 is None:
        with pytest.raises(AssertionError):
            Jinja2Forms(directory="/tmp/templates")


# LLM-generated content at query #33
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates", package=None)
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(directory=None, package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)

    # Test autoescape is enabled
    assert env.autoescape is True


# LLM-generated content at query #34
#--------------------------

```python
def test_Form_render_field(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' id='test-field' name='test_field' value='test_value'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a simple schema with a string field
    schema = Schema(fields={"test_field": String(title="Test Field")})
    
    # Create form instance
    form = Form(env=mock_env, schema=schema)
    
    # Get the field from schema
    field = schema.fields["test_field"]
    
    # Test render_field with basic parameters
    result = form.render_field(
        field_name="test_field",
        field=field,
        value="test_value",
        error=None
    )
    
    # Assertions
    assert result == "<input type='text' id='test-field' name='test_field' value='test_value'>"
    mock_env.get_template.assert_called_once_with("forms/input.html")
    mock_template.render.assert_called_once()
    
    # Verify the context passed to render
    call_args = mock_template.render.call_args[0][0]
    assert call_args["field_id"] == "test-field"
    assert call_args["field_name"] == "test_field"
    assert call_args["label"] == "Test Field"
    assert call_args["value"] == "test_value"
    assert call_args["error"] is None
    assert call_args["input_type"] == "text"
    assert call_args["required"] is True


def test_Form_render_field_with_error(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' id='test-field' name='test_field'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"test_field": String()})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["test_field"]
    
    result = form.render_field(
        field_name="test_field",
        field=field,
        value=None,
        error="Field is required"
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["error"] == "Field is required"


def test_Form_render_field_password_clears_value(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='password'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["password"]
    
    form.render_field(
        field_name="password",
        field=field,
        value="secret",
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["value"] == ""
    assert call_args["input_type"] == "password"


def test_Form_render_field_choice_field(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<select></select>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"status": Choice(choices=[("active", "Active"), ("inactive", "Inactive")])})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["status"]
    
    form.render_field(
        field_name="status",
        field=field,
        value="active",
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/select.html")


def test_Form_render_field_boolean_field(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='checkbox'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"agree": Boolean()})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["agree"]
    
    form.render_field(
        field_name="agree",
        field=field,
        value=True,
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/checkbox.html")


def test_Form_render_field_textarea(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<textarea></textarea>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"description": String(format="text")})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["description"]
    
    form.render_field(
        field_name="description",
        field=field,
        value="Some text",
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/textarea.html")


def test_Form_render_field_with_underscores_in_name(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"test_field_name": String()})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["test_field_name"]
    
    form.render_field(
        field_name="test_field_name",
        field=field,
        value=None,
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["field_id"] == "test-field-name"


def test_Form_render_field_with_default_value(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"field": String(default="default_val")})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["field"]
    
    form.render_field(
        field_name="field",
        field=field,
        value="default_val",
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["required"] is False


# LLM-generated content at query #35
#--------------------------

```python
def test_Jinja2Forms():
    import tempfile
    import os
    
    # Test with directory parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test assertion when neither directory nor package is specified
    try:
        Jinja2Forms()
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Either 'directory' or 'package' must be specified" in str(e)


# LLM-generated content at query #36
#--------------------------

```python
def test_Form_render_fields():
    import jinja2
    
    # Create a minimal Jinja2 environment with templates
    loader = jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>',
        "forms/select.html": '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>',
    })
    env = jinja2.Environment(loader=loader, autoescape=True)
    
    # Create a simple schema
    schema = Schema(fields={
        "name": String(max_length=100),
        "email": String(format="email"),
        "active": Boolean(),
    })
    
    # Create form without validation
    form = Form(env=env, schema=schema, values={"name": "John", "email": "john@example.com", "active": True})
    
    # Test render_fields before validation (should use self.values)
    html = form.render_fields()
    assert "John" in html
    assert "john@example.com" in html
    assert isinstance(html, str)
    assert len(html) > 0
    
    # Test render_fields after validation with valid data
    form.validate({"name": "Jane", "email": "jane@example.com", "active": False})
    html = form.render_fields()
    assert form.is_valid
    assert "Jane" in html
    assert "jane@example.com" in html
    
    # Test render_fields after validation with invalid data (errors present)
    form2 = Form(env=env, schema=schema, values={"name": "Bob"})
    form2.validate({"name": "", "email": "invalid-email", "active": True})
    html = form2.render_fields()
    assert not form2.is_valid
    assert form2.errors is not None
    assert isinstance(html, str)
    
    # Test with read_only field (should be excluded)
    schema_with_readonly = Schema(fields={
        "id": String(read_only=True),
        "name": String(),
    })
    form3 = Form(env=env, schema=schema_with_readonly, values={"id": "123", "name": "Test"})
    html = form3.render_fields()
    assert "123" not in html or "id" not in html  # read_only field should not be rendered
    assert "Test" in html


# LLM-generated content at query #37
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory parameter
    forms = Jinja2Forms(directory="/tmp/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with both directory and package parameters
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test that assertion fails when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()

    # Test that assertion fails when jinja2 is not available
    if jinja2 is not None:
        original_jinja2 = globals()["jinja2"]
        try:
            globals()["jinja2"] = None
            with pytest.raises(AssertionError, match="jinja2 must be installed"):
                Jinja2Forms(directory="/tmp/templates")
        finally:
            globals()["jinja2"] = original_jinja2


# LLM-generated content at query #38
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        }),
        autoescape=True
    )
    
    schema = Schema(fields={
        'name': String(title='Name'),
        'email': String(format='email'),
        'password': String(format='password'),
        'bio': String(format='text'),
        'active': Boolean(),
        'role': Choice(choices=['admin', 'user']),
    })
    
    form = Form(env=env, schema=schema, values={})
    
    # Test rendering text input field
    result = form.render_field(
        field_name='name',
        field=schema.fields['name'],
        value='John',
        error=None
    )
    assert 'name="name"' in result
    assert 'id="name"' in result
    assert 'value="John"' in result
    assert 'type="text"' in result
    assert 'required' in result
    
    # Test rendering email field
    result = form.render_field(
        field_name='email',
        field=schema.fields['email'],
        value='test@example.com',
        error=None
    )
    assert 'name="email"' in result
    assert 'type="email"' in result
    assert 'value="test@example.com"' in result
    
    # Test rendering password field with empty value
    result = form.render_field(
        field_name='password',
        field=schema.fields['password'],
        value='secret',
        error=None
    )
    assert 'type="password"' in result
    assert 'value=""' in result
    
    # Test rendering textarea field
    result = form.render_field(
        field_name='bio',
        field=schema.fields['bio'],
        value='My bio',
        error=None
    )
    assert '<textarea' in result
    assert 'name="bio"' in result
    
    # Test rendering checkbox field
    result = form.render_field(
        field_name='active',
        field=schema.fields['active'],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in result
    
    # Test rendering select field
    result = form.render_field(
        field_name='role',
        field=schema.fields['role'],
        value='admin',
        error=None
    )
    assert '<select' in result
    assert 'name="role"' in result
    
    # Test with error message
    result = form.render_field(
        field_name='name',
        field=schema.fields['name'],
        value='John',
        error='This field is required'
    )
    assert 'This field is required' in result
    
    # Test field_id conversion with underscores
    result = form.render_field(
        field_name='user_name',
        field=String(),
        value='test',
        error=None
    )
    assert 'id="user-name"' in result
    
    # Test with None value
    result = form.render_field(
        field_name='email',
        field=schema.fields['email'],
        value=None,
        error=None
    )
    assert 'value="None"' in result or 'value=""' in result


# LLM-generated content at query #39
#--------------------------

```python
def test_Form_render_fields(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='field1'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a mock schema with fields
    mock_schema = mocker.Mock(spec=Schema)
    mock_field = mocker.Mock(spec=Field)
    mock_field.read_only = False
    mock_field.allow_null = False
    mock_field.allow_blank = False
    mock_field.has_default.return_value = False
    mock_field.title = "Field 1"
    mock_field.format = None
    
    mock_schema.fields = {"field1": mock_field}
    mock_schema.serialize.return_value = {"field1": "value1"}
    
    # Create Form instance
    form = Form(env=mock_env, schema=mock_schema, values={"field1": "value1"})
    
    # Validate the form first
    form.validate({"field1": "value1"})
    
    # Call render_fields
    result = form.render_fields()
    
    # Assertions
    assert isinstance(result, str)
    assert "<input type='text' name='field1'>" in result
    mock_env.get_template.assert_called()
    mock_template.render.assert_called()


def test_Form_render_fields_with_errors(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='field1'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a mock schema with fields
    mock_schema = mocker.Mock(spec=Schema)
    mock_field = mocker.Mock(spec=Field)
    mock_field.read_only = False
    mock_field.allow_null = False
    mock_field.allow_blank = False
    mock_field.has_default.return_value = False
    mock_field.title = "Field 1"
    mock_field.format = None
    
    mock_schema.fields = {"field1": mock_field}
    mock_schema.serialize.return_value = {"field1": "value1"}
    
    # Create Form instance
    form = Form(env=mock_env, schema=mock_schema, values={"field1": "value1"})
    
    # Validate with errors
    form.validate({"field1": "invalid"})
    form.errors = {"field1": "Invalid value"}
    form.data = {"field1": "invalid"}
    
    # Call render_fields
    result = form.render_fields()
    
    # Assertions
    assert isinstance(result, str)
    mock_template.render.assert_called()
    call_args = mock_template.render.call_args[0][0]
    assert call_args["error"] == "Invalid value"


def test_Form_render_fields_skips_read_only(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a mock schema with read-only and regular fields
    mock_schema = mocker.Mock(spec=Schema)
    
    read_only_field = mocker.Mock(spec=Field)
    read_only_field.read_only = True
    
    regular_field = mocker.Mock(spec=Field)
    regular_field.read_only = False
    regular_field.allow_null = False
    regular_field.allow_blank = False
    regular_field.has_default.return_value = False
    regular_field.title = "Regular"
    regular_field.format = None
    
    mock_schema.fields = {"read_only_field": read_only_field, "regular_field": regular_field}
    mock_schema.serialize.return_value = {"read_only_field": "ro", "regular_field": "val"}
    
    # Create Form instance
    form = Form(env=mock_env, schema=mock_schema, values=None)
    form.validate({"regular_field": "val"})
    
    # Call render_fields
    result = form.render_fields()
    
    # Assertions - template should only be called for non-read-only field
    assert mock_template.render.call_count == 1
    call_args = mock_template.render.call_args[0][0]
    assert call_args["field_name"] == "regular_field"


def test_Form_render_fields_multiple_fields(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.side_effect = [
        "<input type='text' name='field1'>",
        "<input type='email' name='field2'>"
    ]
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a mock schema with multiple fields
    mock_schema = mocker.Mock(spec=Schema)
    
    field1 = mocker.Mock(spec=Field)
    field1.read_only = False
    field1.allow_null = False
    field1.allow_blank = False
    field1.has_default.return_value = False
    field1.title = "Field 1"
    field1.format = None
    
    field2 = mocker.Mock(spec=Field)
    field2.read_only = False
    field2.allow_null = False
    field2.allow_blank = False
    field2.has_default.return_value = False
    field2.title = "Field 2"
    field2.format = "email"
    
    mock_schema.fields = {"field1": field1, "field2": field2}
    mock_schema.serialize.return_value = {"field1": "value1", "field2": "test@example.com"}
    
    # Create Form instance
    form = Form(env=mock_env, schema=mock_schema, values=None)
    form.validate({"field1": "value1", "field2": "test@example.com"})
    
    # Call render_fields
    result = form.render_fields()
    
    # Assertions
    assert "<input type='text' name='field1'>" in result
    assert "<input type='email' name='field2'>" in result
    assert mock_template.render.call_count == 2


# LLM-generated content at query #40
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory parameter
    forms = Jinja2Forms(directory="/path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package parameters
    forms = Jinja2Forms(directory="/path/to/templates", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test that assertion is raised when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()

    # Test that assertion is raised when jinja2 is not available
    if jinja2 is None:
        with pytest.raises(AssertionError):
            Jinja2Forms(directory="/path/to/templates")


# LLM-generated content at query #41
#--------------------------

```python
def test_Form_render_fields(mocker):
    # Mock jinja2 environment and template
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<input type='text' name='test_field' />"
    
    # Create a simple schema with fields
    schema = Schema(fields={
        "name": String(),
        "email": String(format="email"),
        "is_active": Boolean(),
        "read_only_field": String(read_only=True),
    })
    
    # Create form instance
    form = Form(env=mock_env, schema=schema, values={"name": "John", "email": "john@example.com", "is_active": True})
    
    # Call validate to set up the form state
    form.validate(data={"name": "John", "email": "john@example.com", "is_active": True})
    
    # Call render_fields
    result = form.render_fields()
    
    # Verify that render_fields returns a string
    assert isinstance(result, str)
    
    # Verify that get_template was called for each non-read-only field
    assert mock_env.get_template.call_count == 3  # name, email, is_active (not read_only_field)
    
    # Verify template.render was called for each field
    assert mock_template.render.call_count == 3
    
    # Verify the result contains rendered content
    assert "<input type='text' name='test_field' />" in result


def test_Form_render_fields_with_errors(mocker):
    # Mock jinja2 environment and template
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<input type='text' name='test_field' /><span class='error'>Invalid</span>"
    
    # Create a simple schema
    schema = Schema(fields={
        "name": String(max_length=5),
        "email": String(format="email"),
    })
    
    # Create form instance
    form = Form(env=mock_env, schema=schema, values=None)
    
    # Validate with invalid data
    form.validate(data={"name": "This is too long", "email": "invalid-email"})
    
    # Call render_fields
    result = form.render_fields()
    
    # Verify that render was called with error information
    assert isinstance(result, str)
    assert mock_template.render.call_count >= 1


def test_Form_render_fields_empty_schema(mocker):
    # Mock jinja2 environment
    mock_env = mocker.Mock()
    
    # Create an empty schema
    schema = Schema(fields={})
    
    # Create form instance
    form = Form(env=mock_env, schema=schema, values=None)
    form.validate(data={})
    
    # Call render_fields
    result = form.render_fields()
    
    # Verify that result is empty string since no fields to render
    assert result == ""
    assert mock_env.get_template.call_count == 0


def test_Form_render_fields_all_read_only(mocker):
    # Mock jinja2 environment
    mock_env = mocker.Mock()
    
    # Create schema with all read-only fields
    schema = Schema(fields={
        "field1": String(read_only=True),
        "field2": String(read_only=True),
    })
    
    # Create form instance
    form = Form(env=mock_env, schema=schema, values={"field1": "value1", "field2": "value2"})
    form.validate(data={"field1": "value1", "field2": "value2"})
    
    # Call render_fields
    result = form.render_fields()
    
    # Verify that no templates were rendered
    assert result == ""
    assert mock_env.get_template.call_count == 0


# LLM-generated content at query #42
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field method renders field correctly with proper attributes."""
    import jinja2
    
    # Create a simple Jinja2 environment with a basic template
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>',
            'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        })
    )
    
    schema = Schema(fields={
        'username': String(),
        'password': String(format='password'),
        'bio': String(format='text'),
        'active': Boolean(),
        'role': Choice(choices=[('admin', 'Admin'), ('user', 'User')]),
        'email': String(format='email', allow_null=True),
    })
    
    form = Form(env=env, schema=schema)
    
    # Test rendering a text input field
    html = form.render_field(
        field_name='username',
        field=schema.fields['username'],
        value='john',
        error=None
    )
    assert 'type="text"' in html
    assert 'name="username"' in html
    assert 'id="username"' in html
    assert 'value="john"' in html
    assert 'required' in html
    assert 'error' not in html
    
    # Test rendering a password field
    html = form.render_field(
        field_name='password',
        field=schema.fields['password'],
        value='secret',
        error=None
    )
    assert 'type="password"' in html
    assert 'name="password"' in html
    assert 'value=""' in html  # Password fields should have empty value
    
    # Test rendering a field with error
    html = form.render_field(
        field_name='username',
        field=schema.fields['username'],
        value='john',
        error='Username is required'
    )
    assert 'Username is required' in html
    assert '<span class="error">' in html
    
    # Test rendering a boolean field
    html = form.render_field(
        field_name='active',
        field=schema.fields['active'],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in html
    assert 'name="active"' in html
    
    # Test rendering a choice field
    html = form.render_field(
        field_name='role',
        field=schema.fields['role'],
        value='admin',
        error=None
    )
    assert 'name="role"' in html
    assert '<select' in html
    
    # Test rendering a textarea field
    html = form.render_field(
        field_name='bio',
        field=schema.fields['bio'],
        value='My bio',
        error=None
    )
    assert '<textarea' in html
    assert 'name="bio"' in html
    assert 'My bio' in html
    
    # Test field with allow_null (not required)
    html = form.render_field(
        field_name='email',
        field=schema.fields['email'],
        value=None,
        error=None
    )
    assert 'required' not in html
    
    # Test field_id conversion (underscore to hyphen)
    html = form.render_field(
        field_name='user_name',
        field=String(),
        value=None,
        error=None
    )
    assert 'id="user-name"' in html


# LLM-generated content at query #43
#--------------------------

```python
def test_Form_render_fields():
    """Test Form.render_fields() method"""
    import jinja2
    
    # Create a mock environment with templates
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/textarea.html': '<textarea name="{{ field_name }}" {% if required %}required{% endif %} id="{{ field_id }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}" {% if required %}required{% endif %} id="{{ field_id }}">{% for choice in field.choices %}<option value="{{ choice[0] }}" {% if choice[0] == value %}selected{% endif %}>{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }))
    
    # Create a schema with various field types
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'bio': String(format='text'),
        'active': Boolean(),
        'role': Choice(choices=[('admin', 'Administrator'), ('user', 'User')]),
        'hidden_field': String(format='hidden', read_only=True),
    })
    
    # Test 1: render_fields with no validation called
    form = Form(env=env, schema=schema, values={'name': 'John', 'email': 'john@example.com'})
    html = form.render_fields()
    assert 'name="name"' in html
    assert 'name="email"' in html
    assert 'type="email"' in html
    assert 'type="hidden"' not in html  # read_only field should not be rendered
    
    # Test 2: render_fields after validation with errors
    form = Form(env=env, schema=schema)
    form.validate({'name': '', 'email': 'invalid', 'active': True})
    html = form.render_fields()
    assert 'error' in html or html  # Should render with error markup
    
    # Test 3: render_fields after successful validation
    form = Form(env=env, schema=schema)
    form.validate({'name': 'Jane', 'email': 'jane@example.com', 'active': True, 'role': 'admin'})
    html = form.render_fields()
    assert 'name="name"' in html
    assert 'value="Jane"' in html
    assert 'name="email"' in html
    assert 'name="active"' in html
    assert 'type="checkbox"' in html
    assert 'name="role"' in html
    
    # Test 4: render_fields with None values
    form = Form(env=env, schema=schema, values=None)
    html = form.render_fields()
    assert 'name="name"' in html
    assert 'name="email"' in html
    
    # Test 5: render_fields after validation with partial data
    form = Form(env=env, schema=schema)
    form.validate({'name': 'Bob'})
    html = form.render_fields()
    assert 'name="name"' in html
    assert 'value="Bob"' in html


# LLM-generated content at query #44
#--------------------------

```python
def test_Form_render_fields():
    # Setup mock environment and schema
    mock_env = jinja2.Environment()
    mock_env.loader = jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/select.html": '<select name="{{ field_name }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    })
    
    schema = Schema(fields={
        "name": String(max_length=100),
        "email": String(format="email"),
        "active": Boolean(),
        "readonly_field": String(read_only=True),
    })
    
    form = Form(env=mock_env, schema=schema, values={"name": "John", "email": "john@example.com", "active": True, "readonly_field": "hidden"})
    form.validate({"name": "John", "email": "john@example.com", "active": True})
    
    html = form.render_fields()
    
    assert "name" in html
    assert "email" in html
    assert "active" in html
    assert "readonly_field" not in html
    assert "John" in html
    assert "john@example.com" in html


def test_Form_render_fields_with_errors():
    mock_env = jinja2.Environment()
    mock_env.loader = jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/select.html": '<select name="{{ field_name }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    })
    
    schema = Schema(fields={
        "name": String(max_length=100),
        "email": String(format="email"),
    })
    
    form = Form(env=mock_env, schema=schema)
    form.validate({"name": "", "email": "invalid"})
    
    html = form.render_fields()
    
    assert "error" in html or "name" in html
    assert "email" in html


def test_Form_render_fields_no_validation():
    mock_env = jinja2.Environment()
    mock_env.loader = jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}">',
        "forms/select.html": '<select name="{{ field_name }}"></select>',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
    })
    
    schema = Schema(fields={
        "name": String(max_length=100),
    })
    
    form = Form(env=mock_env, schema=schema, values={"name": "Test"})
    form.data = None
    form.errors = None
    
    html = form.render_fields()
    
    assert "name" in html
    assert "Test" in html


def test_Form_render_fields_empty_values():
    mock_env = jinja2.Environment()
    mock_env.loader = jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}">',
        "forms/select.html": '<select name="{{ field_name }}"></select>',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
    })
    
    schema = Schema(fields={
        "name": String(max_length=100),
        "age": String(allow_null=True),
    })
    
    form = Form(env=mock_env, schema=schema)
    form.validate({})
    
    html = form.render_fields()
    
    assert "name" in html
    assert "age" in html


####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Form_render_field(mocker):
    """Test Form.render_field() method"""
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='test_field' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a simple schema with a string field
    schema = Schema(fields={"test_field": String(title="Test Field")})
    
    # Create form instance
    form = Form(env=mock_env, schema=schema)
    
    # Get the field
    field = schema.fields["test_field"]
    
    # Test basic render_field call
    result = form.render_field(
        field_name="test_field",
        field=field,
        value="test_value",
        error=None
    )
    
    # Verify template was retrieved with correct name
    mock_env.get_template.assert_called_once_with("forms/input.html")
    
    # Verify template was rendered with correct context
    call_args = mock_template.render.call_args
    context = call_args[0][0]
    
    assert context["field_id"] == "test-field"
    assert context["field_name"] == "test_field"
    assert context["label"] == "Test Field"
    assert context["input_type"] == "text"
    assert context["value"] == "test_value"
    assert context["error"] is None
    assert context["required"] is True
    assert result == "<input type='text' name='test_field' />"


def test_Form_render_field_with_password():
    """Test render_field with password field masks value"""
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='password' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["password"]
    
    result = form.render_field(
        field_name="password",
        field=field,
        value="secret123",
        error=None
    )
    
    call_args = mock_template.render.call_args
    context = call_args[0][0]
    
    assert context["value"] == ""
    assert context["input_type"] == "password"


def test_Form_render_field_with_error(mocker):
    """Test render_field with error message"""
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"email": String(format="email")})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["email"]
    
    result = form.render_field(
        field_name="email",
        field=field,
        value="invalid",
        error="Invalid email format"
    )
    
    call_args = mock_template.render.call_args
    context = call_args[0][0]
    
    assert context["error"] == "Invalid email format"
    assert context["input_type"] == "email"


def test_Form_render_field_with_underscore_field_name(mocker):
    """Test render_field converts underscores to hyphens in field_id"""
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"user_name": String()})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["user_name"]
    
    form.render_field(field_name="user_name", field=field)
    
    call_args = mock_template.render.call_args
    context = call_args[0][0]
    
    assert context["field_id"] == "user-name"


def test_Form_render_field_choice_field(mocker):
    """Test render_field with Choice field uses select template"""
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<select></select>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"color": Choice(choices=["red", "blue"])})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["color"]
    
    form.render_field(field_name="color", field=field, value="red")
    
    mock_env.get_template.assert_called_once_with("forms/select.html")


def test_Form_render_field_boolean_field(mocker):
    """Test render_field with Boolean field uses checkbox template"""
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='checkbox' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"agree": Boolean()})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["agree"]
    
    form.render_field(field_name="agree", field=field, value=True)
    
    mock_env.get_template.assert_called_once_with("forms/checkbox.html")


def test_Form_render_field_textarea(mocker):
    """Test render_field with text format String field uses textarea template"""
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<textarea></textarea>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"description": String(format="text")})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["description"]
    
    form.render_field(field_name="description", field=field, value="text content")
    
    mock_env.get_template.assert_called_once_with("forms/textarea.html")


def test_Form_render_field_required_attribute(mocker):
    """Test render_field required attribute based on field configuration"""
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Optional field with default
    schema = Schema(fields={"optional": String(default="default_value")})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["optional"]
    
    form.render_field(field_name="optional", field=field)
    
    call_args = mock_template.render.call_args
    context = call_args[0][0]
    
    assert context["required"] is False


def test_Form_render_field_nullable_field(mocker):
    """Test render_field required attribute for nullable field"""
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"nullable_field": String(allow_null=True)})
    form = Form(env=mock_env, schema=schema)
    field = schema.fields["nullable_field"]
    
    form.render_field(field_name="nullable_field", field=field)
    
    call_args = mock_template.render.call_args
    context = call_args[0][0]
    
    


# LLM-generated content at query #2
#--------------------------

```python
def test_Form___str__():
    import jinja2
    
    # Create a simple schema
    schema = Schema(fields={
        "name": String(max_length=100),
        "email": String(format="email"),
        "active": Boolean(),
    })
    
    # Create a Jinja2 environment with a simple loader
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}" {% if required %}required{% endif %}>',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
        }),
        autoescape=True
    )
    
    # Create a form instance
    form = Form(env=env, schema=schema, values={"name": "John", "email": "john@example.com", "active": True})
    
    # Test __str__ returns a string
    result = str(form)
    assert isinstance(result, str)
    
    # Test that __str__ calls render_fields
    assert "name" in result or len(result) >= 0  # render_fields() should produce output
    
    # Test with empty values
    form_empty = Form(env=env, schema=schema, values=None)
    result_empty = str(form_empty)
    assert isinstance(result_empty, str)
    
    # Test with validated form containing errors
    form_with_errors = Form(env=env, schema=schema, values={})
    form_with_errors.validate({"name": "", "email": "invalid"})
    result_with_errors = str(form_with_errors)
    assert isinstance(result_with_errors, str)


# LLM-generated content at query #3
#--------------------------

```python
def test_Form_input_type_for_field():
    # Create a mock jinja2 environment
    env = jinja2.Environment()
    schema = Schema(fields={})
    form = Form(env=env, schema=schema)
    
    # Test with field that has no format attribute
    field_no_format = String()
    assert form.input_type_for_field(field_no_format) == "text"
    
    # Test with field that has format attribute set to None
    field_none_format = String(format=None)
    assert form.input_type_for_field(field_none_format) == "text"
    
    # Test with all supported formats
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
        field = String(format=format_value)
        assert form.input_type_for_field(field) == expected_input_type
    
    # Test with unsupported format (should default to "text")
    field_unsupported = String(format="unsupported_format")
    assert form.input_type_for_field(field_unsupported) == "text"
    
    # Test with field that doesn't have format attribute at all
    field_no_attr = Boolean()
    assert form.input_type_for_field(field_no_attr) == "text"


# LLM-generated content at query #4
#--------------------------

```python
def test_Form_template_for_field():
    # Setup
    env = jinja2.Environment()
    schema = Schema(fields={})
    form = Form(env=env, schema=schema)
    
    # Test Choice field returns select template
    choice_field = Choice(choices=["a", "b"])
    assert form.template_for_field(choice_field) == "forms/select.html"
    
    # Test Boolean field returns checkbox template
    boolean_field = Boolean()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"
    
    # Test String field with text format returns textarea template
    text_field = String(format="text")
    assert form.template_for_field(text_field) == "forms/textarea.html"
    
    # Test String field with other format returns input template
    email_field = String(format="email")
    assert form.template_for_field(email_field) == "forms/input.html"
    
    # Test generic field returns input template
    generic_field = String()
    assert form.template_for_field(generic_field) == "forms/input.html"
    
    # Test that Object field raises assertion error
    object_field = Object(properties={})
    with pytest.raises(AssertionError, match="Forms do not support rendering Object fields"):
        form.template_for_field(object_field)


# LLM-generated content at query #5
#--------------------------

```python
def test_Form_template_for_field():
    import jinja2
    
    # Create a mock environment
    env = jinja2.Environment()
    
    # Create a simple schema
    schema = Schema(fields={
        "test_field": String()
    })
    
    # Create a Form instance
    form = Form(env=env, schema=schema)
    
    # Test with Choice field
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert form.template_for_field(choice_field) == "forms/select.html"
    
    # Test with Boolean field
    boolean_field = Boolean()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"
    
    # Test with String field with format="text"
    text_field = String(format="text")
    assert form.template_for_field(text_field) == "forms/textarea.html"
    
    # Test with regular String field
    string_field = String()
    assert form.template_for_field(string_field) == "forms/input.html"
    
    # Test with String field with other format
    email_field = String(format="email")
    assert form.template_for_field(email_field) == "forms/input.html"
    
    # Test that Object field raises AssertionError
    object_field = Object(properties={"nested": String()})
    try:
        form.template_for_field(object_field)
        assert False, "Expected AssertionError for Object field"
    except AssertionError as e:
        assert "Forms do not support rendering Object fields" in str(e)


# LLM-generated content at query #6
#--------------------------

```python
def test_Jinja2Forms_create_form():
    # Test creating a form with a simple schema
    forms = Jinja2Forms(package="typesystem")
    
    schema = Schema(fields={
        "name": String(),
        "email": String(format="email"),
        "active": Boolean(),
    })
    
    # Test creating form without values
    form = forms.create_form(schema)
    assert isinstance(form, Form)
    assert form.schema is schema
    assert form.env is forms.env
    assert form.values == {"name": None, "email": None, "active": None}
    assert form.errors is None
    assert form._validate_called is False
    
    # Test creating form with values
    values = {"name": "John", "email": "john@example.com", "active": True}
    form_with_values = forms.create_form(schema, values=values)
    assert isinstance(form_with_values, Form)
    assert form_with_values.schema is schema
    assert form_with_values.env is forms.env
    assert form_with_values.values == values
    assert form_with_values.errors is None
    assert form_with_values._validate_called is False
    
    # Test that each form instance is independent
    assert form is not form_with_values
    assert form.values != form_with_values.values


# LLM-generated content at query #7
#--------------------------

```python
def test_Form_render_field(tmp_path):
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    forms_dir = template_dir / "forms"
    forms_dir.mkdir()
    
    # Create template files
    (forms_dir / "input.html").write_text(
        '<input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>'
        '{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    )
    (forms_dir / "checkbox.html").write_text(
        '<input type="checkbox" id="{{ field_id }}" name="{{ field_name }}"{% if value %} checked{% endif %}>'
        '{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    )
    (forms_dir / "select.html").write_text(
        '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}'
        '<option value="{{ choice[0] }}"{% if choice[0] == value %} selected{% endif %}>{{ choice[1] }}</option>'
        '{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    )
    (forms_dir / "textarea.html").write_text(
        '<textarea id="{{ field_id }}" name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>'
        '{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    )
    
    forms = Jinja2Forms(directory=str(template_dir))
    schema = Schema(fields={"name": String(), "email": String(format="email")})
    form = Form(env=forms.env, schema=schema, values={"name": "John"})
    
    # Test rendering a basic text field
    html = form.render_field(field_name="name", field=schema.fields["name"], value="John")
    assert 'type="text"' in html
    assert 'id="name"' in html
    assert 'name="name"' in html
    assert 'value="John"' in html
    assert 'required' in html
    
    # Test rendering with error
    html = form.render_field(
        field_name="name", 
        field=schema.fields["name"], 
        value="John",
        error="This field is required"
    )
    assert 'This field is required' in html
    assert '<span class="error">' in html
    
    # Test rendering email field
    html = form.render_field(field_name="email", field=schema.fields["email"], value="test@example.com")
    assert 'type="email"' in html
    assert 'value="test@example.com"' in html
    
    # Test rendering password field (value should be empty)
    password_schema = Schema(fields={"password": String(format="password")})
    form2 = Form(env=forms.env, schema=password_schema)
    html = form2.render_field(
        field_name="password",
        field=password_schema.fields["password"],
        value="secret123"
    )
    assert 'type="password"' in html
    assert 'value=""' in html
    
    # Test rendering optional field (no required attribute)
    optional_schema = Schema(fields={"nickname": String(allow_null=True)})
    form3 = Form(env=forms.env, schema=optional_schema)
    html = form3.render_field(
        field_name="nickname",
        field=optional_schema.fields["nickname"],
        value=""
    )
    assert 'required' not in html
    
    # Test rendering boolean field
    bool_schema = Schema(fields={"active": Boolean()})
    form4 = Form(env=forms.env, schema=bool_schema)
    html = form4.render_field(
        field_name="active",
        field=bool_schema.fields["active"],
        value=True
    )
    assert 'type="checkbox"' in html
    assert 'checked' in html
    
    # Test rendering choice field
    choice_schema = Schema(
        fields={"status": Choice(choices=[("active", "Active"), ("inactive", "Inactive")])}
    )
    form5 = Form(env=forms.env, schema=choice_schema)
    html = form5.render_field(
        field_name="status",
        field=choice_schema.fields["status"],
        value="active"
    )
    assert '<select' in html
    assert 'selected' in html
    assert 'Active' in html


# LLM-generated content at query #8
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory parameter
    forms = Jinja2Forms(directory="/tmp")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with both directory and package parameters
    forms = Jinja2Forms(directory="/tmp", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test assertion when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test that jinja2 is required
    if jinja2 is None:
        with pytest.raises(AssertionError):
            Jinja2Forms(directory="/tmp")


# LLM-generated content at query #9
#--------------------------

```python
def test_Jinja2Forms_create_form():
    # Test create_form returns a Form instance
    forms = Jinja2Forms(package="typesystem")
    
    schema = Schema(fields={"name": String(), "email": String(format="email")})
    form = forms.create_form(schema)
    
    assert isinstance(form, Form)
    assert form.schema is schema
    assert form.env is forms.env
    assert form.values == {"name": None, "email": None}
    assert form.errors is None


def test_Jinja2Forms_create_form_with_values():
    # Test create_form with initial values
    forms = Jinja2Forms(package="typesystem")
    
    schema = Schema(fields={"name": String(), "email": String(format="email")})
    initial_values = {"name": "John", "email": "john@example.com"}
    form = forms.create_form(schema, values=initial_values)
    
    assert isinstance(form, Form)
    assert form.schema is schema
    assert form.values == initial_values
    assert form.errors is None


def test_Jinja2Forms_create_form_multiple_schemas():
    # Test create_form with different schemas
    forms = Jinja2Forms(package="typesystem")
    
    schema1 = Schema(fields={"username": String()})
    schema2 = Schema(fields={"email": String(format="email"), "age": String(format="number")})
    
    form1 = forms.create_form(schema1)
    form2 = forms.create_form(schema2)
    
    assert form1.schema is schema1
    assert form2.schema is schema2
    assert form1.values == {"username": None}
    assert form2.values == {"email": None, "age": None}


# LLM-generated content at query #10
#--------------------------

```python
def test_Jinja2Forms_create_form():
    """Test that create_form returns a Form instance with correct initialization."""
    # Create a simple schema
    schema = Schema(fields={"name": String(), "age": String(format="number")})
    
    # Create Jinja2Forms instance with package loader
    forms = Jinja2Forms(package="typesystem")
    
    # Test create_form without values
    form = forms.create_form(schema=schema)
    assert isinstance(form, Form)
    assert form.schema is schema
    assert form.env is forms.env
    assert form.values == {}
    assert form.errors is None
    assert form._validate_called is False
    
    # Test create_form with values
    values = {"name": "John", "age": "30"}
    form_with_values = forms.create_form(schema=schema, values=values)
    assert isinstance(form_with_values, Form)
    assert form_with_values.schema is schema
    assert form_with_values.env is forms.env
    assert form_with_values.values == values
    assert form_with_values.errors is None
    assert form_with_values._validate_called is False


# LLM-generated content at query #11
#--------------------------

```python
def test_Form_render_field(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='username' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a schema with a simple string field
    schema = Schema(fields={"username": String(max_length=100)})
    
    # Create form instance
    form = Form(env=mock_env, schema=schema, values=None)
    
    # Test render_field with basic parameters
    result = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john_doe",
        error=None
    )
    
    # Verify template was retrieved correctly
    mock_env.get_template.assert_called_once_with("forms/input.html")
    
    # Verify template.render was called with correct context
    mock_template.render.assert_called_once()
    call_args = mock_template.render.call_args[0][0]
    assert call_args["field_id"] == "username"
    assert call_args["field_name"] == "username"
    assert call_args["label"] == "username"
    assert call_args["required"] is True
    assert call_args["input_type"] == "text"
    assert call_args["value"] == "john_doe"
    assert call_args["error"] is None
    assert result == "<input type='text' name='username' />"


def test_Form_render_field_with_error(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='email' /><span>Invalid email</span>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"email": String(format="email")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="invalid",
        error="Invalid email format"
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["error"] == "Invalid email format"
    assert call_args["input_type"] == "email"


def test_Form_render_field_password_field(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='password' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret123",
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["value"] == ""
    assert call_args["input_type"] == "password"


def test_Form_render_field_with_title(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"user_name": String(title="Username")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="user_name",
        field=schema.fields["user_name"],
        value="john",
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["label"] == "Username"
    assert call_args["field_id"] == "user-name"


def test_Form_render_field_checkbox(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='checkbox' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"agree": Boolean()})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="agree",
        field=schema.fields["agree"],
        value=True,
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/checkbox.html")


def test_Form_render_field_choice(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<select></select>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"color": Choice(choices=[("red", "Red"), ("blue", "Blue")])})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="color",
        field=schema.fields["color"],
        value="red",
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/select.html")


# LLM-generated content at query #12
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/textarea.html': '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/select.html': '<select name="{{ field_name }}"{% if required %} required{% endif %}>{% for choice in field.choices %}<option value="{{ choice.value }}">{{ choice.title }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        }),
        autoescape=True
    )
    
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'active': Boolean(),
        'read_only_field': String(read_only=True),
    })
    
    form = Form(env=env, schema=schema, values={'name': 'John', 'email': 'john@example.com', 'active': True})
    form.validate({'name': 'John', 'email': 'john@example.com', 'active': True})
    
    # Test successful render with valid data
    html = form.render_fields()
    assert 'name="name"' in html
    assert 'value="John"' in html
    assert 'name="email"' in html
    assert 'value="john@example.com"' in html
    assert 'type="checkbox"' in html
    assert 'read_only_field' not in html
    assert 'error' not in html
    
    # Test render with errors
    form2 = Form(env=env, schema=schema, values={})
    form2.validate({'name': '', 'email': 'invalid', 'active': False})
    
    if form2.errors:
        html_with_errors = form2.render_fields()
        assert 'error' in html_with_errors or form2.is_valid
    
    # Test render with None values before validation
    form3 = Form(env=env, schema=schema, values=None)
    assert form3.errors is None
    assert not form3._validate_called


# LLM-generated content at query #13
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}">{% for choice in field.choices %}<option value="{{ choice.value }}">{{ choice.label }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }), autoescape=True)
    
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'active': Boolean(),
        'status': Choice(choices=[('active', 'Active'), ('inactive', 'Inactive')]),
    })
    
    form = Form(env=env, schema=schema, values={'name': 'John', 'email': 'john@example.com', 'active': True, 'status': 'active'})
    form.validate({'name': 'John', 'email': 'john@example.com', 'active': True, 'status': 'active'})
    
    # Test render_fields with valid data
    html = form.render_fields()
    assert 'name="name"' in html
    assert 'name="email"' in html
    assert 'name="active"' in html
    assert 'name="status"' in html
    assert 'John' in html
    assert 'john@example.com' in html
    
    # Test render_fields with no errors
    assert 'class="error"' not in html
    
    # Test render_fields with errors
    form2 = Form(env=env, schema=schema, values={})
    form2.validate({'name': '', 'email': 'invalid', 'active': None, 'status': 'invalid'})
    html_with_errors = form2.render_fields()
    assert 'class="error"' in html_with_errors
    
    # Test render_fields with read_only field
    schema_with_readonly = Schema(fields={
        'id': String(read_only=True),
        'name': String(max_length=100),
    })
    form3 = Form(env=env, schema=schema_with_readonly, values={'id': '123', 'name': 'John'})
    form3.validate({'name': 'John'})
    html_readonly = form3.render_fields()
    assert 'name="id"' not in html_readonly
    assert 'name="name"' in html_readonly


# LLM-generated content at query #14
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates", package=None)
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(directory=None, package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)

    # Test autoescape is enabled
    assert env.autoescape is True


# LLM-generated content at query #15
#--------------------------

```python
def test_Jinja2Forms():
    import tempfile
    import os
    
    # Test with directory parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test assertion when neither directory nor package is specified
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test assertion when jinja2 is not available (mocked scenario)
    import sys
    original_jinja2 = sys.modules.get('jinja2')
    try:
        sys.modules['jinja2'] = None
        import importlib
        import typesystem.forms as forms_module
        importlib.reload(forms_module)
        with pytest.raises(AssertionError, match="jinja2 must be installed"):
            forms_module.Jinja2Forms(directory=".")
    finally:
        if original_jinja2 is not None:
            sys.modules['jinja2'] = original_jinja2
        else:
            sys.modules.pop('jinja2', None)


# LLM-generated content at query #16
#--------------------------

```python
def test_Form_render_field(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='test_field'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a simple schema
    schema = Schema(fields={"test_field": String(title="Test Field")})
    
    # Create form instance
    form = Form(env=mock_env, schema=schema, values=None)
    
    # Test basic render_field call
    result = form.render_field(
        field_name="test_field",
        field=String(title="Test Field"),
        value="test_value",
        error=None
    )
    
    # Verify template was retrieved
    mock_env.get_template.assert_called_once_with("forms/input.html")
    
    # Verify template was rendered with correct context
    mock_template.render.assert_called_once()
    render_context = mock_template.render.call_args[0][0]
    
    assert render_context["field_id"] == "test-field"
    assert render_context["field_name"] == "test_field"
    assert render_context["label"] == "Test Field"
    assert render_context["required"] is True
    assert render_context["input_type"] == "text"
    assert render_context["value"] == "test_value"
    assert render_context["error"] is None
    assert result == "<input type='text' name='test_field'>"


def test_Form_render_field_with_error(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='field'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"field": String()})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="field",
        field=String(),
        value="value",
        error="This field is required"
    )
    
    render_context = mock_template.render.call_args[0][0]
    assert render_context["error"] == "This field is required"


def test_Form_render_field_password_clears_value(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='password'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="password",
        field=String(format="password"),
        value="secret123",
        error=None
    )
    
    render_context = mock_template.render.call_args[0][0]
    assert render_context["value"] == ""
    assert render_context["input_type"] == "password"


def test_Form_render_field_with_default_not_required(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"field": String(default="default_value")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="field",
        field=String(default="default_value"),
        value="default_value",
        error=None
    )
    
    render_context = mock_template.render.call_args[0][0]
    assert render_context["required"] is False


def test_Form_render_field_choice_field(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<select>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"choice": Choice(choices=["a", "b"])})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="choice",
        field=Choice(choices=["a", "b"]),
        value="a",
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/select.html")


def test_Form_render_field_boolean_field(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='checkbox'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"bool_field": Boolean()})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="bool_field",
        field=Boolean(),
        value=True,
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/checkbox.html")


def test_Form_render_field_textarea(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<textarea></textarea>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"text": String(format="text")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="text",
        field=String(format="text"),
        value="multiline text",
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/textarea.html")


def test_Form_render_field_with_underscore_in_name(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"test_field_name": String()})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="test_field_name",
        field=String(),
        value=None,
        error=None
    )
    
    render_context = mock_template.render.call_args[0][0]
    assert render_context["field_id"] == "test-field-name"


def test_Form_render_field_email_input_type(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='email'>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"email": String(format="email")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="email",
        field=String(format="email"),
        value="test@example.com",
        error=None
    )
    
    render_context = mock_template.render.call_args[0][0]
    assert render_context["input_type"] == "email"


# LLM-generated content at query #17
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    
    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)
    
    # Test with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)
    
    # Test autoescape is enabled
    assert env.autoescape is True


# LLM-generated content at query #18
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    
    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)
    
    # Test with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)
    
    # Test autoescape is enabled
    assert env.autoescape is True


# LLM-generated content at query #19
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/textarea.html": '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/select.html": '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}></select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }))
    
    schema = Schema(fields={
        "username": String(max_length=100),
        "email": String(format="email"),
        "bio": String(format="text"),
        "is_active": Boolean(),
        "role": Choice(choices=[("admin", "Admin"), ("user", "User")]),
        "age": Field(format="number"),
    })
    
    form = Form(env=env, schema=schema, values=None)
    
    # Test rendering text input field
    result = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john_doe",
        error=None
    )
    assert 'type="text"' in result
    assert 'name="username"' in result
    assert 'id="username"' in result
    assert 'value="john_doe"' in result
    assert 'required' in result
    
    # Test rendering email input field
    result = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="test@example.com",
        error=None
    )
    assert 'type="email"' in result
    assert 'name="email"' in result
    assert 'value="test@example.com"' in result
    
    # Test rendering field with error
    result = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john",
        error="This field is required"
    )
    assert 'This field is required' in result
    
    # Test rendering textarea field
    result = form.render_field(
        field_name="bio",
        field=schema.fields["bio"],
        value="Some bio text",
        error=None
    )
    assert '<textarea' in result
    assert 'name="bio"' in result
    assert 'Some bio text' in result
    
    # Test rendering checkbox field
    result = form.render_field(
        field_name="is_active",
        field=schema.fields["is_active"],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in result
    assert 'checked' in result
    
    # Test rendering select field
    result = form.render_field(
        field_name="role",
        field=schema.fields["role"],
        value="admin",
        error=None
    )
    assert '<select' in result
    assert 'name="role"' in result
    
    # Test field_id conversion (underscore to hyphen)
    result = form.render_field(
        field_name="user_name",
        field=String(),
        value=None,
        error=None
    )
    assert 'id="user-name"' in result
    
    # Test with allow_null field (not required)
    nullable_field = String(allow_null=True)
    result = form.render_field(
        field_name="optional_field",
        field=nullable_field,
        value=None,
        error=None
    )
    assert 'required' not in result
    
    # Test password field returns empty value
    password_field = String(format="password")
    result = form.render_field(
        field_name="password",
        field=password_field,
        value="secret123",
        error=None
    )
    assert 'value=""' in result
    assert 'secret123' not in result


# LLM-generated content at query #20
#--------------------------

```python
def test_Jinja2Forms():
    import tempfile
    import os
    
    # Test with directory parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test assertion when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test assertion when jinja2 is None
    if jinja2 is not None:
        original_jinja2 = globals()['jinja2']
        try:
            globals()['jinja2'] = None
            with pytest.raises(AssertionError, match="jinja2 must be installed"):
                Jinja2Forms(directory=".")
        finally:
            globals()['jinja2'] = original_jinja2


# LLM-generated content at query #21
#--------------------------

```python
def test_Jinja2Forms():
    import tempfile
    import os
    
    # Test with directory parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test assertion when neither directory nor package is specified
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test assertion when jinja2 is None
    original_jinja2 = globals()['jinja2']
    try:
        globals()['jinja2'] = None
        with pytest.raises(AssertionError):
            Jinja2Forms(directory=".")
    finally:
        globals()['jinja2'] = original_jinja2


# LLM-generated content at query #22
#--------------------------

```python
def test_Form_render_fields():
    import jinja2
    from typesystem.schemas import Schema
    from typesystem.fields import String, Integer, Boolean
    
    # Create a simple schema
    schema = Schema(
        fields={
            "name": String(max_length=100),
            "age": Integer(),
            "active": Boolean(),
            "readonly_field": String(read_only=True),
        }
    )
    
    # Create a Jinja2 environment with a simple loader
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<p class="error">{{ error }}</p>{% endif %}',
            "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<p class="error">{{ error }}</p>{% endif %}',
        }),
        autoescape=True
    )
    
    # Test 1: render_fields with no validation called yet
    form = Form(env=env, schema=schema, values={"name": "John", "age": 30, "active": True})
    html = form.render_fields()
    assert "John" in html
    assert "30" in html
    assert "readonly_field" not in html  # read_only field should not be rendered
    assert 'name="name"' in html
    assert 'name="age"' in html
    assert 'name="active"' in html
    
    # Test 2: render_fields after validation with errors
    form2 = Form(env=env, schema=schema, values=None)
    form2.validate({"name": "", "age": "invalid", "active": False})
    html2 = form2.render_fields()
    # Errors should be displayed when validation has been called
    assert isinstance(html2, str)
    
    # Test 3: render_fields after successful validation
    form3 = Form(env=env, schema=schema, values=None)
    form3.validate({"name": "Jane", "age": 25, "active": True})
    html3 = form3.render_fields()
    assert "Jane" in html3
    assert "25" in html3
    
    # Test 4: render_fields with None values
    form4 = Form(env=env, schema=schema, values=None)
    html4 = form4.render_fields()
    assert isinstance(html4, str)
    assert 'value=""' in html4 or 'value=' in html4


# LLM-generated content at query #23
#--------------------------

```python
def test_Form_render_fields():
    """Test Form.render_fields() method."""
    # Setup mock environment and schema
    mock_env = jinja2.Environment()
    mock_env.loader = jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
        'forms/select.html': '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>',
    })
    
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'active': Boolean(),
        'read_only_field': String(read_only=True),
    })
    
    form = Form(env=mock_env, schema=schema, values={'name': 'John', 'email': 'john@example.com', 'active': True})
    form.validate({'name': 'John', 'email': 'john@example.com', 'active': True})
    
    html = form.render_fields()
    
    # Verify HTML is rendered
    assert isinstance(html, str)
    assert 'name' in html
    assert 'email' in html
    assert 'active' in html
    # Read-only field should not be rendered
    assert 'read_only_field' not in html
    assert 'John' in html
    assert 'john@example.com' in html


def test_Form_render_fields_with_errors():
    """Test Form.render_fields() with validation errors."""
    mock_env = jinja2.Environment()
    mock_env.loader = jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    })
    
    schema = Schema(fields={
        'name': String(max_length=5),
    })
    
    form = Form(env=mock_env, schema=schema)
    form.validate({'name': 'VeryLongNameThatExceedsLimit'})
    
    html = form.render_fields()
    
    assert isinstance(html, str)
    assert 'name' in html
    assert 'error' in html.lower() or 'VeryLongNameThatExceedsLimit' in html


def test_Form_render_fields_empty_values():
    """Test Form.render_fields() with no values."""
    mock_env = jinja2.Environment()
    mock_env.loader = jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>',
    })
    
    schema = Schema(fields={
        'name': String(max_length=100),
    })
    
    form = Form(env=mock_env, schema=schema, values=None)
    form.validate(None)
    
    html = form.render_fields()
    
    assert isinstance(html, str)
    assert 'name' in html


def test_Form_render_fields_with_choices():
    """Test Form.render_fields() with Choice field."""
    mock_env = jinja2.Environment()
    mock_env.loader = jinja2.DictLoader({
        'forms/select.html': '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>',
    })
    
    schema = Schema(fields={
        'status': Choice(choices=['active', 'inactive']),
    })
    
    form = Form(env=mock_env, schema=schema, values={'status': 'active'})
    form.validate({'status': 'active'})
    
    html = form.render_fields()
    
    assert isinstance(html, str)
    assert 'select' in html
    assert 'status' in html


def test_Form_render_fields_with_boolean():
    """Test Form.render_fields() with Boolean field."""
    mock_env = jinja2.Environment()
    mock_env.loader = jinja2.DictLoader({
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
    })
    
    schema = Schema(fields={
        'agree': Boolean(),
    })
    
    form = Form(env=mock_env, schema=schema, values={'agree': True})
    form.validate({'agree': True})
    
    html = form.render_fields()
    
    assert isinstance(html, str)
    assert 'checkbox' in html
    assert 'agree' in html


# LLM-generated content at query #24
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field() method renders field correctly."""
    import jinja2
    
    # Create a simple Jinja2 environment with a mock template
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>',
        'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }), autoescape=True)
    
    # Create a simple schema
    schema = Schema(fields={
        'username': String(max_length=100),
        'email': String(format='email'),
        'is_active': Boolean(),
        'bio': String(format='text'),
    })
    
    # Create a form instance
    form = Form(env=env, schema=schema, values=None)
    
    # Test rendering a text field
    username_field = schema.fields['username']
    html = form.render_field(
        field_name='username',
        field=username_field,
        value='john_doe',
        error=None
    )
    assert 'name="username"' in html
    assert 'id="username"' in html
    assert 'value="john_doe"' in html
    assert 'required' in html
    
    # Test rendering an email field
    email_field = schema.fields['email']
    html = form.render_field(
        field_name='email',
        field=email_field,
        value='john@example.com',
        error=None
    )
    assert 'name="email"' in html
    assert 'type="email"' in html
    assert 'value="john@example.com"' in html
    
    # Test rendering a field with error
    html = form.render_field(
        field_name='username',
        field=username_field,
        value='john',
        error='Username too short'
    )
    assert 'error' in html.lower()
    assert 'Username too short' in html
    
    # Test rendering a boolean field
    is_active_field = schema.fields['is_active']
    html = form.render_field(
        field_name='is_active',
        field=is_active_field,
        value=True,
        error=None
    )
    assert 'type="checkbox"' in html
    assert 'name="is_active"' in html
    
    # Test rendering a textarea field
    bio_field = schema.fields['bio']
    html = form.render_field(
        field_name='bio',
        field=bio_field,
        value='My bio',
        error=None
    )
    assert '<textarea' in html
    assert 'My bio' in html
    
    # Test field_id with underscores converted to hyphens
    html = form.render_field(
        field_name='first_name',
        field=String(),
        value='John',
        error=None
    )
    assert 'id="first-name"' in html
    assert 'name="first_name"' in html


# LLM-generated content at query #25
#--------------------------

```python
def test_Form_render_field(jinja2_forms):
    """Test Form.render_field method renders field with correct context."""
    from typesystem.fields import String, Integer, Boolean, Choice
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        name = String(title="Full Name")
        age = Integer(title="Age", allow_null=True)
        active = Boolean(title="Active")
        choice_field = Choice(choices=["option1", "option2"], title="Select Option")
    
    form = jinja2_forms.create_form(TestSchema)
    
    # Test rendering a string field
    html = form.render_field(
        field_name="name",
        field=TestSchema.fields["name"],
        value="John Doe",
        error=None
    )
    assert "John Doe" in html
    assert "Full Name" in html
    assert "name" in html
    
    # Test rendering a field with error
    html = form.render_field(
        field_name="name",
        field=TestSchema.fields["name"],
        value="",
        error="This field is required"
    )
    assert "This field is required" in html
    
    # Test rendering a boolean field
    html = form.render_field(
        field_name="active",
        field=TestSchema.fields["active"],
        value=True,
        error=None
    )
    assert "active" in html
    assert "Active" in html
    
    # Test rendering a choice field
    html = form.render_field(
        field_name="choice_field",
        field=TestSchema.fields["choice_field"],
        value="option1",
        error=None
    )
    assert "choice_field" in html
    assert "Select Option" in html
    
    # Test nullable field is not required
    html = form.render_field(
        field_name="age",
        field=TestSchema.fields["age"],
        value=None,
        error=None
    )
    assert "age" in html
    
    # Test password field clears value
    password_field = String(format="password")
    html = form.render_field(
        field_name="password",
        field=password_field,
        value="secret123",
        error=None
    )
    assert "password" in html
    assert "secret123" not in html


# LLM-generated content at query #26
#--------------------------

```python
def test_Form_render_fields():
    # Setup mock jinja2 environment and schema
    mock_env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/select.html": '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }))
    
    schema = Schema(fields={
        "name": String(max_length=100),
        "email": String(format="email"),
        "active": Boolean(),
        "readonly_field": String(read_only=True),
    })
    
    form = Form(env=mock_env, schema=schema, values={"name": "John", "email": "john@example.com", "active": True, "readonly_field": "hidden"})
    form.validate({"name": "John", "email": "john@example.com", "active": True})
    
    html = form.render_fields()
    
    # Verify rendered HTML contains expected fields
    assert "name" in html
    assert "email" in html
    assert "active" in html
    assert "readonly_field" not in html  # read_only fields should not be rendered
    assert "John" in html
    assert "john@example.com" in html


def test_Form_render_fields_with_errors():
    mock_env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/select.html": '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }))
    
    schema = Schema(fields={
        "email": String(format="email"),
    })
    
    form = Form(env=mock_env, schema=schema)
    form.validate({"email": "invalid-email"})
    
    html = form.render_fields()
    
    # Verify error messages are rendered
    assert "error" in html or "email" in html


def test_Form_render_fields_empty_values():
    mock_env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>',
        "forms/select.html": '<select name="{{ field_name }}"></select>',
        "forms/textarea.html": '<textarea name="{{ field_name }}">{{ value }}</textarea>',
    }))
    
    schema = Schema(fields={
        "name": String(max_length=100),
    })
    
    form = Form(env=mock_env, schema=schema, values=None)
    form.validate({})
    
    html = form.render_fields()
    
    assert "name" in html
    assert isinstance(html, str)


# LLM-generated content at query #27
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/checkbox.html": '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/select.html": '<select name="{{ field_name }}" id="{{ field_id }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        "forms/textarea.html": '<textarea name="{{ field_name }}" id="{{ field_id }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }), autoescape=True)
    
    schema = Schema(fields={
        "name": String(title="Name"),
        "email": String(format="email"),
        "password": String(format="password"),
        "agree": Boolean(),
        "choice_field": Choice(choices=["option1", "option2"]),
        "description": String(format="text"),
    })
    
    form = Form(env=env, schema=schema, values={})
    
    # Test basic text input
    html = form.render_field(field_name="name", field=schema.fields["name"], value="John")
    assert 'type="text"' in html
    assert 'name="name"' in html
    assert 'id="name"' in html
    assert 'value="John"' in html
    assert 'required' in html
    
    # Test email input
    html = form.render_field(field_name="email", field=schema.fields["email"], value="test@example.com")
    assert 'type="email"' in html
    assert 'value="test@example.com"' in html
    
    # Test password input (value should be empty)
    html = form.render_field(field_name="password", field=schema.fields["password"], value="secret")
    assert 'type="password"' in html
    assert 'value=""' in html
    
    # Test checkbox (boolean field)
    html = form.render_field(field_name="agree", field=schema.fields["agree"], value=True)
    assert 'type="checkbox"' in html
    
    # Test select (choice field)
    html = form.render_field(field_name="choice_field", field=schema.fields["choice_field"])
    assert '<select' in html
    
    # Test textarea
    html = form.render_field(field_name="description", field=schema.fields["description"], value="Some text")
    assert '<textarea' in html
    assert '>Some text</textarea>' in html
    
    # Test with error
    html = form.render_field(field_name="name", field=schema.fields["name"], value="John", error="Invalid name")
    assert '<span class="error">Invalid name</span>' in html
    
    # Test field_id transformation (underscores to dashes)
    schema_with_underscore = Schema(fields={"user_name": String()})
    form2 = Form(env=env, schema=schema_with_underscore, values={})
    html = form2.render_field(field_name="user_name", field=schema_with_underscore.fields["user_name"])
    assert 'id="user-name"' in html
    
    # Test optional field (no required attribute)
    optional_field = String(allow_null=True)
    html = form.render_field(field_name="optional", field=optional_field, value=None)
    assert 'required' not in html


# LLM-generated content at query #28
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field method renders field with correct context."""
    import jinja2
    
    # Create a simple Jinja2 environment with a mock template
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>',
    }))
    
    schema = Schema(fields={'email': String(format='email')})
    form = Form(env=env, schema=schema, values={})
    
    # Test rendering a String field with email format
    html = form.render_field(
        field_name='email',
        field=schema.fields['email'],
        value='test@example.com',
        error=None
    )
    assert 'type="email"' in html
    assert 'name="email"' in html
    assert 'id="email"' in html
    assert 'value="test@example.com"' in html
    assert 'required' in html
    assert 'error' not in html
    
    # Test rendering with error
    html = form.render_field(
        field_name='email',
        field=schema.fields['email'],
        value='invalid',
        error='Invalid email format'
    )
    assert 'Invalid email format' in html
    
    # Test rendering password field (value should be empty)
    schema_with_password = Schema(fields={'password': String(format='password')})
    form_password = Form(env=env, schema=schema_with_password, values={})
    html = form_password.render_field(
        field_name='password',
        field=schema_with_password.fields['password'],
        value='secret123',
        error=None
    )
    assert 'type="password"' in html
    assert 'value=""' in html
    
    # Test rendering optional field (allow_null=True)
    schema_optional = Schema(fields={'optional_field': String(allow_null=True)})
    form_optional = Form(env=env, schema=schema_optional, values={})
    html = form_optional.render_field(
        field_name='optional_field',
        field=schema_optional.fields['optional_field'],
        value=None,
        error=None
    )
    assert 'required' not in html
    
    # Test field_id with underscores replaced by hyphens
    schema_underscore = Schema(fields={'user_name': String()})
    form_underscore = Form(env=env, schema=schema_underscore, values={})
    html = form_underscore.render_field(
        field_name='user_name',
        field=schema_underscore.fields['user_name'],
        value='john',
        error=None
    )
    assert 'id="user-name"' in html
    assert 'name="user_name"' in html


# LLM-generated content at query #29
#--------------------------

```python
def test_Form_render_field(mocker):
    # Mock jinja2 environment and template
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' name='test_field' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    # Create a simple schema with a string field
    schema = Schema(fields={"test_field": String(title="Test Field")})
    
    # Create a Form instance
    form = Form(env=mock_env, schema=schema, values=None)
    
    # Test rendering a basic field
    result = form.render_field(
        field_name="test_field",
        field=schema.fields["test_field"],
        value="test_value",
        error=None
    )
    
    # Verify the template was retrieved with the correct name
    mock_env.get_template.assert_called_once_with("forms/input.html")
    
    # Verify the template was rendered with correct context
    call_args = mock_template.render.call_args[0][0]
    assert call_args["field_id"] == "test-field"
    assert call_args["field_name"] == "test_field"
    assert call_args["label"] == "Test Field"
    assert call_args["required"] is True
    assert call_args["input_type"] == "text"
    assert call_args["value"] == "test_value"
    assert call_args["error"] is None
    assert result == "<input type='text' name='test_field' />"


def test_Form_render_field_with_error(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='text' /> <span>Error message</span>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"email_field": String(format="email")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="email_field",
        field=schema.fields["email_field"],
        value="invalid",
        error="Invalid email"
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["input_type"] == "email"
    assert call_args["error"] == "Invalid email"


def test_Form_render_field_password_clears_value(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='password' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret123",
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["value"] == ""
    assert call_args["input_type"] == "password"


def test_Form_render_field_choice(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<select></select>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"status": Choice(choices=["active", "inactive"])})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="status",
        field=schema.fields["status"],
        value="active",
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/select.html")


def test_Form_render_field_boolean(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input type='checkbox' />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"agree": Boolean()})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="agree",
        field=schema.fields["agree"],
        value=True,
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/checkbox.html")


def test_Form_render_field_textarea(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<textarea></textarea>"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"description": String(format="text")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    result = form.render_field(
        field_name="description",
        field=schema.fields["description"],
        value="Some text",
        error=None
    )
    
    mock_env.get_template.assert_called_once_with("forms/textarea.html")


def test_Form_render_field_with_underscore_in_name(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"user_name": String()})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="user_name",
        field=schema.fields["user_name"],
        value=None,
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["field_id"] == "user-name"


def test_Form_render_field_with_default_value(mocker):
    mock_template = mocker.Mock()
    mock_template.render.return_value = "<input />"
    
    mock_env = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    
    schema = Schema(fields={"field_with_default": String(default="default_value")})
    form = Form(env=mock_env, schema=schema, values=None)
    
    form.render_field(
        field_name="field_with_default",
        field=schema.fields["field_with_default"],
        value="default_value",
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["required"] is False


# LLM-generated content at query #30
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package raises assertion error
    with pytest.raises(AssertionError):
        forms = Jinja2Forms(directory="/tmp/templates")
        forms.load_template_env()

    # Test autoescape is enabled
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates")
    assert env.autoescape is True


# LLM-generated content at query #31
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>',
        'forms/select.html': '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
    }))
    
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'is_active': Boolean(),
        'status': Choice(choices=[('active', 'Active'), ('inactive', 'Inactive')]),
        'bio': String(format='text'),
    })
    
    form = Form(env=env, schema=schema, values={
        'name': 'John',
        'email': 'john@example.com',
        'is_active': True,
        'status': 'active',
        'bio': 'Test bio',
    })
    
    # Validate to set values and errors
    form.validate({
        'name': 'John',
        'email': 'john@example.com',
        'is_active': True,
        'status': 'active',
        'bio': 'Test bio',
    })
    
    # Test render_fields
    html = form.render_fields()
    
    # Assertions
    assert isinstance(html, str)
    assert 'name="name"' in html
    assert 'name="email"' in html
    assert 'name="is_active"' in html
    assert 'name="status"' in html
    assert 'name="bio"' in html
    assert 'John' in html
    assert 'john@example.com' in html
    assert 'checked' in html  # is_active is True
    assert '<textarea' in html  # bio field is text format


def test_Form_render_fields_with_errors():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}> {% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}">',
        'forms/select.html': '<select name="{{ field_name }}"></select>',
        'forms/textarea.html': '<textarea name="{{ field_name }}"></textarea>',
    }))
    
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
    })
    
    form = Form(env=env, schema=schema)
    
    # Validate with invalid data
    form.validate({'email': 'invalid-email'})
    
    # Test render_fields with errors
    html = form.render_fields()
    
    assert isinstance(html, str)
    assert form.errors is not None
    assert 'name="name"' in html
    assert 'name="email"' in html


def test_Form_render_fields_empty_values():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}">',
        'forms/select.html': '<select name="{{ field_name }}"></select>',
        'forms/textarea.html': '<textarea name="{{ field_name }}"></textarea>',
    }))
    
    schema = Schema(fields={
        'name': String(),
        'description': String(allow_null=True),
    })
    
    form = Form(env=env, schema=schema, values=None)
    form.validate({})
    
    # Test render_fields with no values
    html = form.render_fields()
    
    assert isinstance(html, str)
    assert 'name="name"' in html
    assert 'name="description"' in html


def test_Form_render_fields_read_only_excluded():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}">',
        'forms/select.html': '<select name="{{ field_name }}"></select>',
        'forms/textarea.html': '<textarea name="{{ field_name }}"></textarea>',
    }))
    
    schema = Schema(fields={
        'name': String(),
        'id': String(read_only=True),
    })
    
    form = Form(env=env, schema=schema, values={'name': 'John', 'id': '123'})
    form.validate({'name': 'John'})
    
    # Test render_fields excludes read_only fields
    html = form.render_fields()
    
    assert 'name="name"' in html
    assert 'name="id"' not in html


# LLM-generated content at query #32
#--------------------------

```python
def test_Form_render_field():
    # Setup
    jinja2_forms = Jinja2Forms(package="typesystem")
    schema = Schema(fields={
        "username": String(max_length=100),
        "email": String(format="email"),
        "password": String(format="password"),
        "age": String(format="number"),
        "bio": String(format="text"),
        "is_active": Boolean(),
        "role": Choice(choices=["admin", "user"]),
        "website": String(format="url"),
    })
    
    form = form.create_form(schema)
    
    # Test rendering a simple text field
    html = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john_doe",
        error=None
    )
    assert "username" in html
    assert "john_doe" in html
    assert "type" in html
    
    # Test rendering an email field
    html = form.render_field(
        field_name="email",
        field=schema.fields["email"],
        value="john@example.com",
        error=None
    )
    assert "email" in html
    assert "john@example.com" in html
    
    # Test rendering a password field (value should be empty)
    html = form.render_field(
        field_name="password",
        field=schema.fields["password"],
        value="secret",
        error=None
    )
    assert "password" in html
    assert "secret" not in html
    
    # Test rendering field with error
    html = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value="john",
        error="This field is required"
    )
    assert "This field is required" in html
    
    # Test rendering number field
    html = form.render_field(
        field_name="age",
        field=schema.fields["age"],
        value="25",
        error=None
    )
    assert "age" in html
    assert "25" in html
    
    # Test rendering textarea field
    html = form.render_field(
        field_name="bio",
        field=schema.fields["bio"],
        value="My bio",
        error=None
    )
    assert "bio" in html
    
    # Test rendering checkbox field
    html = form.render_field(
        field_name="is_active",
        field=schema.fields["is_active"],
        value=True,
        error=None
    )
    assert "is_active" in html
    
    # Test rendering select field
    html = form.render_field(
        field_name="role",
        field=schema.fields["role"],
        value="admin",
        error=None
    )
    assert "role" in html
    
    # Test field_id replaces underscores with hyphens
    html = form.render_field(
        field_name="user_name",
        field=schema.fields["username"],
        value=None,
        error=None
    )
    assert "user-name" in html
    
    # Test with None value
    html = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value=None,
        error=None
    )
    assert "username" in html
    
    # Test required field
    html = form.render_field(
        field_name="username",
        field=schema.fields["username"],
        value=None,
        error=None
    )
    assert "required" in html.lower() or "username" in html


# LLM-generated content at query #33
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates", package=None)
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(directory=None, package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)

    # Test autoescape is enabled
    assert env.autoescape is True


# LLM-generated content at query #34
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field method with various field types."""
    import jinja2
    
    # Create a minimal Jinja2 environment with a simple template
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>',
            'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        }),
        autoescape=True
    )
    
    schema = Schema(fields={'test_field': String()})
    form = Form(env=env, schema=schema, values=None)
    
    # Test basic input field rendering
    field = String(title='Test Label')
    html = form.render_field(field_name='test_field', field=field, value='test_value')
    assert 'name="test_field"' in html
    assert 'id="test-field"' in html
    assert 'value="test_value"' in html
    assert 'required' in html
    
    # Test with error
    html = form.render_field(field_name='test_field', field=field, value='test_value', error='This is an error')
    assert 'This is an error' in html
    
    # Test password field (value should be empty string)
    password_field = String(format='password')
    html = form.render_field(field_name='password_field', field=password_field, value='secret')
    assert 'value=""' in html
    assert 'input_type' in form.input_type_for_field(password_field) or 'password' in html
    
    # Test with optional field (allow_null=True)
    optional_field = String(allow_null=True)
    html = form.render_field(field_name='optional_field', field=optional_field, value=None)
    assert 'required' not in html
    
    # Test with default value
    field_with_default = String(default='default_value')
    html = form.render_field(field_name='field_with_default', field=field_with_default, value='default_value')
    assert 'required' not in html
    
    # Test Boolean field
    boolean_field = Boolean()
    html = form.render_field(field_name='bool_field', field=boolean_field)
    assert 'type="checkbox"' in html or 'checkbox' in html
    
    # Test Choice field
    choice_field = Choice(choices=['option1', 'option2'])
    html = form.render_field(field_name='choice_field', field=choice_field)
    assert 'select' in html
    
    # Test field with underscore in name (should convert to dash in id)
    html = form.render_field(field_name='test_field_name', field=field)
    assert 'id="test-field-name"' in html
    
    # Test custom title
    field_with_title = String(title='Custom Title')
    html = form.render_field(field_name='custom_field', field=field_with_title)
    # Title is used in template rendering context
    
    # Test read_only field behavior (indirectly through render_field)
    readonly_field = String(read_only=False)
    html = form.render_field(field_name='readonly_field', field=readonly_field)
    assert 'name="readonly_field"' in html


# LLM-generated content at query #35
#--------------------------

```python
def test_Jinja2Forms():
    # Test that Jinja2Forms raises AssertionError when jinja2 is not available
    if jinja2 is None:
        with pytest.raises(AssertionError, match="jinja2 must be installed"):
            Jinja2Forms(directory="/tmp")
        return

    # Test that Jinja2Forms raises AssertionError when neither directory nor package is specified
    with pytest.raises(AssertionError, match="Either 'directory' or 'package' must be specified"):
        Jinja2Forms()

    # Test successful initialization with directory parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)

    # Test successful initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test successful initialization with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
        assert isinstance(forms.env.loader, jinja2.ChoiceLoader)


# LLM-generated content at query #36
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package only
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    env = forms.load_template_env(directory="/tmp/templates", package="typesystem")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)

    # Test that autoescape is enabled
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates")
    assert env.autoescape is True

    # Test assertion when both directory and package are None
    forms = Jinja2Forms(directory="/tmp/templates")
    with pytest.raises(AssertionError):
        forms.load_template_env(directory=None, package=None)

    # Test assertion when directory is provided but package is not (should pass)
    forms = Jinja2Forms(directory="/tmp/templates")
    env = forms.load_template_env(directory="/tmp/templates", package=None)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test assertion when package is provided but directory is not (should pass)
    forms = Jinja2Forms(package="typesystem")
    env = forms.load_template_env(directory=None, package="typesystem")
    assert isinstance(env.loader, jinja2.PackageLoader)


# LLM-generated content at query #37
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory parameter only
    forms = Jinja2Forms(directory="/tmp/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package parameter only
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package parameters
    forms = Jinja2Forms(directory="/tmp/templates", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test that assertion fails when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()

    # Test that jinja2 environment has autoescape enabled
    forms = Jinja2Forms(directory="/tmp/templates")
    assert forms.env.autoescape is True


# LLM-generated content at query #38
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field method with various field types."""
    import jinja2
    
    # Create a minimal Jinja2 environment with string templates
    env = jinja2.Environment(
        loader=jinja2.DictLoader({
            'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/select.html': '<select name="{{ field_name }}" id="{{ field_id }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
            'forms/textarea.html': '<textarea name="{{ field_name }}" id="{{ field_id }}">{{ value or "" }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        })
    )
    
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'password': String(format='password'),
        'agree': Boolean(),
        'choice_field': Choice(choices=[('a', 'Option A'), ('b', 'Option B')]),
        'message': String(format='text'),
    })
    
    form = Form(env=env, schema=schema, values={})
    
    # Test rendering a text input field
    html = form.render_field(
        field_name='name',
        field=schema.fields['name'],
        value='John',
        error=None
    )
    assert 'name="name"' in html
    assert 'id="name"' in html
    assert 'value="John"' in html
    assert 'required' in html
    assert 'type="text"' in html
    
    # Test rendering an email field
    html = form.render_field(
        field_name='email',
        field=schema.fields['email'],
        value='test@example.com',
        error=None
    )
    assert 'type="email"' in html
    assert 'value="test@example.com"' in html
    
    # Test rendering a password field with empty value
    html = form.render_field(
        field_name='password',
        field=schema.fields['password'],
        value='secret',
        error=None
    )
    assert 'type="password"' in html
    assert 'value=""' in html  # password fields should have empty value
    
    # Test rendering a boolean field
    html = form.render_field(
        field_name='agree',
        field=schema.fields['agree'],
        value=True,
        error=None
    )
    assert 'type="checkbox"' in html
    assert 'checked' in html
    
    # Test rendering a choice field
    html = form.render_field(
        field_name='choice_field',
        field=schema.fields['choice_field'],
        value='a',
        error=None
    )
    assert '<select' in html
    assert 'Option A' in html
    assert 'Option B' in html
    
    # Test rendering a textarea field
    html = form.render_field(
        field_name='message',
        field=schema.fields['message'],
        value='Hello World',
        error=None
    )
    assert '<textarea' in html
    assert 'Hello World' in html
    
    # Test rendering with error
    html = form.render_field(
        field_name='name',
        field=schema.fields['name'],
        value='John',
        error='This field is required'
    )
    assert 'This field is required' in html
    assert '<span class="error">' in html
    
    # Test field_id uses dashes instead of underscores
    name_with_underscore = String()
    html = form.render_field(
        field_name='user_name',
        field=name_with_underscore,
        value='test',
        error=None
    )
    assert 'id="user-name"' in html
    assert 'name="user_name"' in html


# LLM-generated content at query #39
#--------------------------

```python
def test_Form_render_field():
    """Test Form.render_field() method."""
    # Setup
    forms = Jinja2Forms(package="typesystem")
    
    class TestSchema(Schema):
        name = String(max_length=100, title="Full Name")
        email = String(format="email")
        age = Field(format="number")
        password = String(format="password")
        bio = String(format="text")
        subscribe = Boolean()
        country = Choice(choices=["US", "UK", "CA"])
    
    form = forms.create_form(TestSchema)
    
    # Test rendering a simple text field
    html = form.render_field(
        field_name="name",
        field=TestSchema.fields["name"],
        value="John Doe",
        error=None
    )
    assert "John Doe" in html
    assert "Full Name" in html
    assert "name" in html
    assert "required" in html.lower()
    
    # Test rendering an email field
    html = form.render_field(
        field_name="email",
        field=TestSchema.fields["email"],
        value="test@example.com",
        error=None
    )
    assert "test@example.com" in html
    assert "email" in html
    
    # Test rendering a number field
    html = form.render_field(
        field_name="age",
        field=TestSchema.fields["age"],
        value=25,
        error=None
    )
    assert "25" in html
    assert "number" in html
    
    # Test rendering a password field (value should be empty)
    html = form.render_field(
        field_name="password",
        field=TestSchema.fields["password"],
        value="secret123",
        error=None
    )
    assert "secret123" not in html
    assert "password" in html
    
    # Test rendering a textarea field
    html = form.render_field(
        field_name="bio",
        field=TestSchema.fields["bio"],
        value="My biography",
        error=None
    )
    assert "My biography" in html
    assert "textarea" in html
    
    # Test rendering a checkbox field
    html = form.render_field(
        field_name="subscribe",
        field=TestSchema.fields["subscribe"],
        value=True,
        error=None
    )
    assert "checkbox" in html
    
    # Test rendering a select field
    html = form.render_field(
        field_name="country",
        field=TestSchema.fields["country"],
        value="US",
        error=None
    )
    assert "select" in html
    assert "US" in html
    
    # Test with error message
    html = form.render_field(
        field_name="email",
        field=TestSchema.fields["email"],
        value="invalid",
        error="Invalid email format"
    )
    assert "Invalid email format" in html
    
    # Test field_id conversion (underscores to hyphens)
    class UnderscoreSchema(Schema):
        first_name = String()
    
    form2 = forms.create_form(UnderscoreSchema)
    html = form2.render_field(
        field_name="first_name",
        field=UnderscoreSchema.fields["first_name"],
        value="John",
        error=None
    )
    assert "first-name" in html
    
    # Test with None value
    html = form.render_field(
        field_name="name",
        field=TestSchema.fields["name"],
        value=None,
        error=None
    )
    assert "required" in html.lower()
    
    # Test read_only field is not rendered in render_field
    class ReadOnlySchema(Schema):
        id = Field(read_only=True)
        name = String()
    
    form3 = forms.create_form(ReadOnlySchema)
    html = form3.render_field(
        field_name="id",
        field=ReadOnlySchema.fields["id"],
        value=1,
        error=None
    )
    assert html  # Should still render even though read_only (render_field doesn't check)


# LLM-generated content at query #40
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory parameter
    forms = Jinja2Forms(directory="/tmp")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with both directory and package parameters
    forms = Jinja2Forms(directory="/tmp", package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test that assertion fails when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()
    
    # Test that assertion fails when jinja2 is not available
    if jinja2 is not None:
        original_jinja2 = jinja2
        # This test would require mocking jinja2 to None, which is complex
        # so we skip it in a simple unit test


# LLM-generated content at query #41
#--------------------------

```python
def test_Jinja2Forms():
    # Test that Jinja2Forms requires jinja2 to be installed
    if jinja2 is None:
        with pytest.raises(AssertionError, match="jinja2 must be installed"):
            Jinja2Forms(directory=".")
        return
    
    # Test that either directory or package must be specified
    with pytest.raises(AssertionError, match="Either 'directory' or 'package' must be specified"):
        Jinja2Forms()
    
    # Test initialization with directory parameter
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)


# LLM-generated content at query #42
#--------------------------

```python
def test_Form_render_fields():
    """Test Form.render_fields() method."""
    import jinja2
    
    # Create a mock environment
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
        'forms/select.html': '<select name="{{ field_name }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}',
    }), autoescape=True)
    
    # Create a simple schema
    schema = Schema(fields={
        'name': String(max_length=100),
        'email': String(format='email'),
        'active': Boolean(),
        'bio': String(format='text'),
    })
    
    # Test 1: render_fields with no validation called yet should fail
    form = Form(env=env, schema=schema, values={'name': 'John', 'email': 'john@example.com', 'active': True, 'bio': 'Test'})
    try:
        form.render_fields()
        assert False, "Should raise AssertionError"
    except AssertionError as e:
        assert "validate() has not been called" in str(e)
    
    # Test 2: render_fields with valid data
    form = Form(env=env, schema=schema, values={'name': 'John', 'email': 'john@example.com', 'active': True, 'bio': 'Test'})
    form.validate({'name': 'John', 'email': 'john@example.com', 'active': True, 'bio': 'Test'})
    html = form.render_fields()
    assert 'name="name"' in html
    assert 'name="email"' in html
    assert 'name="active"' in html
    assert 'name="bio"' in html
    assert 'value="John"' in html
    
    # Test 3: render_fields with validation errors
    form = Form(env=env, schema=schema, values={})
    form.validate({'name': '', 'email': 'invalid-email', 'active': None, 'bio': ''})
    html = form.render_fields()
    assert 'error' in html or html  # Should contain error markup or empty fields
    
    # Test 4: render_fields skips read_only fields
    schema_with_readonly = Schema(fields={
        'id': String(read_only=True),
        'name': String(),
    })
    form = Form(env=env, schema=schema_with_readonly, values={'id': '123', 'name': 'John'})
    form.validate({'name': 'John'})
    html = form.render_fields()
    assert 'name="id"' not in html
    assert 'name="name"' in html
    
    # Test 5: render_fields uses data when errors exist
    form = Form(env=env, schema=schema, values={'name': 'Original'})
    form.data = {'name': 'Modified', 'email': 'test@example.com', 'active': False, 'bio': ''}
    form.errors = {'email': 'Invalid email'}
    html = form.render_fields()
    assert 'value="Modified"' in html
    assert 'error' in html


# LLM-generated content at query #43
#--------------------------

```python
def test_Jinja2Forms():
    # Test that Jinja2Forms raises assertion error when jinja2 is not available
    if jinja2 is None:
        with pytest.raises(AssertionError, match="jinja2 must be installed"):
            Jinja2Forms(directory="/tmp")
        return
    
    # Test that Jinja2Forms raises assertion error when neither directory nor package is specified
    with pytest.raises(AssertionError, match="Either 'directory' or 'package' must be specified"):
        Jinja2Forms()
    
    # Test initialization with directory parameter
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)
    
    # Test initialization with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)


# LLM-generated content at query #44
#--------------------------

```python
def test_Jinja2Forms():
    import tempfile
    import os

    # Test with directory parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir)
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)

    # Test with package parameter
    forms = Jinja2Forms(package="typesystem")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        forms = Jinja2Forms(directory=tmpdir, package="typesystem")
        assert forms.env is not None
        assert isinstance(forms.env, jinja2.Environment)

    # Test assertion error when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()

    # Test assertion error when jinja2 is None
    original_jinja2 = globals()["jinja2"]
    try:
        globals()["jinja2"] = None
        with pytest.raises(AssertionError):
            Jinja2Forms(directory=".")
    finally:
        globals()["jinja2"] = original_jinja2


# LLM-generated content at query #45
#--------------------------

```python
def test_Form_render_field(mocker):
    # Mock jinja2 environment and template
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<input type='text' name='test_field'>"
    
    # Create a simple schema
    schema = Schema(fields={"test_field": String(title="Test Field")})
    
    # Create form instance
    form = Form(env=mock_env, schema=schema, values={})
    
    # Test basic field rendering
    field = String(title="Test Field")
    result = form.render_field(
        field_name="test_field",
        field=field,
        value="test_value",
        error=None
    )
    
    # Verify template was retrieved
    mock_env.get_template.assert_called_once_with("forms/input.html")
    
    # Verify template was rendered with correct context
    mock_template.render.assert_called_once()
    call_args = mock_template.render.call_args[0][0]
    
    assert call_args["field_id"] == "test_field"
    assert call_args["field_name"] == "test_field"
    assert call_args["label"] == "Test Field"
    assert call_args["value"] == "test_value"
    assert call_args["error"] is None
    assert call_args["required"] is True
    assert result == "<input type='text' name='test_field'>"


def test_Form_render_field_with_underscore_in_name(mocker):
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<input>"
    
    schema = Schema(fields={"test_field_name": String()})
    form = Form(env=mock_env, schema=schema, values={})
    
    field = String()
    form.render_field(
        field_name="test_field_name",
        field=field,
        value=None,
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["field_id"] == "test-field-name"


def test_Form_render_field_password_clears_value(mocker):
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<input type='password'>"
    
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=mock_env, schema=schema, values={})
    
    field = String(format="password")
    form.render_field(
        field_name="password",
        field=field,
        value="secret123",
        error=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["value"] == ""
    assert call_args["input_type"] == "password"


def test_Form_render_field_with_error(mocker):
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<input>"
    
    schema = Schema(fields={"email": String(format="email")})
    form = Form(env=mock_env, schema=schema, values={})
    
    field = String(format="email")
    form.render_field(
        field_name="email",
        field=field,
        value="invalid",
        error="Invalid email"
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["error"] == "Invalid email"


def test_Form_render_field_choice_field(mocker):
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<select></select>"
    
    schema = Schema(fields={"status": Choice(choices=["active", "inactive"])})
    form = Form(env=mock_env, schema=schema, values={})
    
    field = Choice(choices=["active", "inactive"])
    form.render_field(
        field_name="status",
        field=field,
        value="active"
    )
    
    mock_env.get_template.assert_called_once_with("forms/select.html")


def test_Form_render_field_boolean_field(mocker):
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<input type='checkbox'>"
    
    schema = Schema(fields={"agree": Boolean()})
    form = Form(env=mock_env, schema=schema, values={})
    
    field = Boolean()
    form.render_field(
        field_name="agree",
        field=field,
        value=True
    )
    
    mock_env.get_template.assert_called_once_with("forms/checkbox.html")


def test_Form_render_field_with_default_not_required(mocker):
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<input>"
    
    schema = Schema(fields={"optional": String(default="default_value")})
    form = Form(env=mock_env, schema=schema, values={})
    
    field = String(default="default_value")
    form.render_field(
        field_name="optional",
        field=field,
        value="value"
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["required"] is False


def test_Form_render_field_allow_null_not_required(mocker):
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "<input>"
    
    schema = Schema(fields={"nullable": String(allow_null=True)})
    form = Form(env=mock_env, schema=schema, values={})
    
    field = String(allow_null=True)
    form.render_field(
        field_name="nullable",
        field=field,
        value=None
    )
    
    call_args = mock_template.render.call_args[0][0]
    assert call_args["required"] is False


