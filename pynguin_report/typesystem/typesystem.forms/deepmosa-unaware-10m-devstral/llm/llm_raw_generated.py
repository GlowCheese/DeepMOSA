####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Form_template_for_field():
    # Test for Choice field
    choice_field = Choice(choices=[("1", "Option 1"), ("2", "Option 2")])
    assert Form.template_for_field(choice_field) == "forms/select.html"

    # Test for Boolean field
    boolean_field = Boolean()
    assert Form.template_for_field(boolean_field) == "forms/checkbox.html"

    # Test for String field with text format
    text_field = String(format="text")
    assert Form.template_for_field(text_field) == "forms/textarea.html"

    # Test for String field without text format
    string_field = String()
    assert Form.template_for_field(string_field) == "forms/input.html"

    # Test for other field types
    number_field = Field()
    assert Form.template_for_field(number_field) == "forms/input.html"

    # Test for Object field (should raise assertion)
    object_field = Object()
    with pytest.raises(AssertionError):
        Form.template_for_field(object_field)


# LLM-generated content at query #2
#--------------------------

```python
def test_Form_input_type_for_field():
    # Setup
    form = Form(
        env=jinja2.Environment(),
        schema=Schema(fields={}),
        values={}
    )

    # Test default case (no format)
    field = String()
    assert form.input_type_for_field(field) == "text"

    # Test known formats
    for format, input_type in form.FORMAT_TO_INPUTTYPE.items():
        field = String(format=format)
        assert form.input_type_for_field(field) == input_type

    # Test unknown format
    field = String(format="unknown")
    assert form.input_type_for_field(field) == "text"

    # Test field without format attribute
    field = Boolean()
    assert form.input_type_for_field(field) == "text"


# LLM-generated content at query #3
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}">{% for option in field.choices %}<option value="{{ option.value }}">{{ option.display_name }}</option>{% endfor %}</select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'name': String(max_length=100),
        'age': String(format='number'),
        'description': String(format='text'),
        'country': Choice(choices=[('US', 'United States'), ('UK', 'United Kingdom')]),
        'subscribe': Boolean()
    })
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '25', 'description': 'Test', 'country': 'US', 'subscribe': True})

    # Test rendering fields without validation
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in html
    assert '<input id="age" name="age" type="number" value="25">' in html
    assert '<textarea id="description" name="description">Test</textarea>' in html
    assert '<select id="country" name="country">' in html
    assert '<option value="US">United States</option>' in html
    assert '<option value="UK">United Kingdom</option>' in html
    assert '<input id="subscribe" name="subscribe" type="checkbox" checked>' in html

    # Test rendering fields with validation errors
    form.validate({'name': '', 'age': 'invalid', 'description': '', 'country': 'INVALID', 'subscribe': 'not_a_boolean'})
    html = form.render_fields()
    assert 'value=""' in html  # Empty name
    assert 'value="invalid"' in html  # Invalid age
    assert '<textarea id="description" name="description"></textarea>' in html  # Empty description
    assert '<select id="country" name="country">' in html  # Invalid country
    assert '<input id="subscribe" name="subscribe" type="checkbox"' in html  # Invalid subscribe


# LLM-generated content at query #4
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'number_field': String(format='number'),
        'password_field': String(format='password'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text'),
    })
    form = Form(env=env, schema=schema, values={})

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test password field (value should be empty string)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Error message')
    assert 'Error message' in result


# LLM-generated content at query #5
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #6
#--------------------------

```python
def test_Form___html__():
    # Setup
    env = jinja2.Environment()
    schema = Schema(fields={"name": String(), "age": Integer()})
    form = Form(env=env, schema=schema, values={"name": "John", "age": 30})

    # Exercise
    result = form.__html__()

    # Verify
    assert isinstance(result, markupsafe.Markup)
    assert str(result) == form.render_fields()


# LLM-generated content at query #7
#--------------------------

```python
def test_Form___html__():
    # Setup
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema, values={"name": "test"})

    # Exercise
    result = form.__html__()

    # Verify
    assert isinstance(result, markupsafe.Markup)
    assert str(result) == str(form)


# LLM-generated content at query #8
#--------------------------

```python
def test_Form_template_for_field():
    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    form = Form(env=None, schema=Schema(fields={"test": choice_field}))
    assert form.template_for_field(choice_field) == "forms/select.html"

    # Test Boolean field
    boolean_field = Boolean()
    form = Form(env=None, schema=Schema(fields={"test": boolean_field}))
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"

    # Test String field with text format
    text_field = String(format="text")
    form = Form(env=None, schema=Schema(fields={"test": text_field}))
    assert form.template_for_field(text_field) == "forms/textarea.html"

    # Test String field without text format
    string_field = String()
    form = Form(env=None, schema=Schema(fields={"test": string_field}))
    assert form.template_for_field(string_field) == "forms/input.html"

    # Test Object field (should raise assertion)
    object_field = Object(fields={"subfield": String()})
    form = Form(env=None, schema=Schema(fields={"test": object_field}))
    with pytest.raises(AssertionError):
        form.template_for_field(object_field)


# LLM-generated content at query #9
#--------------------------

```python
def test_Form___str__():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    }))
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'description': String(format='text'),
        'active': Boolean(),
        'type': Choice(choices=['A', 'B', 'C'])
    })
    form = Form(env=env, schema=schema)

    # Exercise
    result = str(form)

    # Verify
    assert '<input id="name" name="name" type="text" value="">' in result
    assert '<input id="age" name="age" type="number" value="">' in result
    assert '<textarea id="description" name="description"></textarea>' in result
    assert '<input id="active" name="active" type="checkbox"' in result
    assert '<select id="type" name="type"></select>' in result


# LLM-generated content at query #10
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'text_field': String(format='text'),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'default_field': String()
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (should have empty value)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field (checked)
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test boolean field (unchecked)
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=False)
    assert '<input id="bool-field" name="bool_field" type="checkbox">' in result
    assert 'checked' not in result

    # Test default field (should be text type)
    result = form.render_field(field_name='default_field', field=schema.fields['default_field'], value='default')
    assert '<input id="default-field" name="default_field" type="text" value="default">' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid value')
    assert 'Invalid value' in result


# LLM-generated content at query #11
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'bio': String(format='text'),
        'active': Boolean(),
        'gender': Choice(choices=['M', 'F']),
    })
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'bio': 'Developer', 'active': True, 'gender': 'M'})

    # Test
    result = form.render_fields()

    # Assert
    assert '<input id="name" name="name" type="text" value="John">' in result
    assert '<input id="age" name="age" type="number" value="30">' in result
    assert '<textarea id="bio" name="bio">Developer</textarea>' in result
    assert '<input id="active" name="active" type="checkbox" checked>' in result
    assert '<select id="gender" name="gender"></select>' in result


# LLM-generated content at query #12
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals["forms/select.html"] = "<select name='{{ field_name }}'></select>"
    env.globals["forms/checkbox.html"] = "<input type='checkbox' name='{{ field_name }}'>"
    env.globals["forms/textarea.html"] = "<textarea name='{{ field_name }}'></textarea>"
    env.globals["forms/input.html"] = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}'>"

    schema = Schema(fields={
        "text_field": String(format="text"),
        "email_field": String(format="email"),
        "choice_field": Choice(choices=["a", "b", "c"]),
        "bool_field": Boolean(),
        "password_field": String(format="password"),
    })

    form = Form(env=env, schema=schema, values={
        "text_field": "test",
        "email_field": "test@example.com",
        "choice_field": "a",
        "bool_field": True,
        "password_field": "secret",
    })

    # Test text field
    html = form.render_field(
        field_name="text_field",
        field=schema.fields["text_field"],
        value="test",
        error=None
    )
    assert "<input type='text' name='text_field' value='test'>" in html

    # Test email field
    html = form.render_field(
        field_name="email_field",
        field=schema.fields["email_field"],
        value="test@example.com",
        error=None
    )
    assert "<input type='email' name='email_field' value='test@example.com'>" in html

    # Test choice field
    html = form.render_field(
        field_name="choice_field",
        field=schema.fields["choice_field"],
        value="a",
        error=None
    )
    assert "<select name='choice_field'></select>" in html

    # Test boolean field
    html = form.render_field(
        field_name="bool_field",
        field=schema.fields["bool_field"],
        value=True,
        error=None
    )
    assert "<input type='checkbox' name='bool_field'>" in html

    # Test password field (value should be empty)
    html = form.render_field(
        field_name="password_field",
        field=schema.fields["password_field"],
        value="secret",
        error=None
    )
    assert "<input type='password' name='password_field' value=''>" in html

    # Test with error
    html = form.render_field(
        field_name="text_field",
        field=schema.fields["text_field"],
        value="test",
        error="Invalid value"
    )
    assert "Invalid value" in html


# LLM-generated content at query #13
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)


# LLM-generated content at query #14
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #15
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals["forms/input.html"] = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    env.globals["forms/select.html"] = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    env.globals["forms/checkbox.html"] = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    env.globals["forms/textarea.html"] = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"

    schema = Schema(fields={
        "text_field": String(),
        "email_field": String(format="email"),
        "password_field": String(format="password"),
        "number_field": String(format="number"),
        "choice_field": Choice(choices=[("a", "A"), ("b", "B")]),
        "bool_field": Boolean(),
        "text_area_field": String(format="text"),
    })

    form = Form(env=env, schema=schema)

    # Test text field
    assert "<input id='text-field' name='text_field' type='text' value='test'>" in form.render_field(
        field_name="text_field", field=schema.fields["text_field"], value="test"
    )

    # Test email field
    assert "<input id='email-field' name='email_field' type='email' value='test@example.com'>" in form.render_field(
        field_name="email_field", field=schema.fields["email_field"], value="test@example.com"
    )

    # Test password field (value should be empty string)
    assert "<input id='password-field' name='password_field' type='password' value=''>" in form.render_field(
        field_name="password_field", field=schema.fields["password_field"], value="secret"
    )

    # Test number field
    assert "<input id='number-field' name='number_field' type='number' value='42'>" in form.render_field(
        field_name="number_field", field=schema.fields["number_field"], value=42
    )

    # Test choice field
    assert "<select id='choice-field' name='choice_field'></select>" in form.render_field(
        field_name="choice_field", field=schema.fields["choice_field"]
    )

    # Test boolean field
    assert "<input id='bool-field' name='bool_field' type='checkbox' checked>" in form.render_field(
        field_name="bool_field", field=schema.fields["bool_field"], value=True
    )
    assert "<input id='bool-field' name='bool_field' type='checkbox'>" in form.render_field(
        field_name="bool_field", field=schema.fields["bool_field"], value=False
    )

    # Test textarea field
    assert "<textarea id='text-area-field' name='text_area_field'>multiline text</textarea>" in form.render_field(
        field_name="text_area_field", field=schema.fields["text_area_field"], value="multiline text"
    )

    # Test with error
    assert "error-message" in form.render_field(
        field_name="text_field", field=schema.fields["text_field"], error="error-message"
    )


# LLM-generated content at query #16
#--------------------------

```python
def test_Form_validate():
    # Setup
    schema = Schema(fields={"name": String(), "age": Integer()})
    env = jinja2.Environment(loader=jinja2.DictLoader({}))
    form = Form(env=env, schema=schema, values={"name": "John", "age": 30})

    # Test valid data
    form.validate({"name": "Jane", "age": 25})
    assert form.is_valid
    assert form.validated_data == {"name": "Jane", "age": 25}
    assert form.errors is None

    # Test invalid data
    form2 = Form(env=env, schema=schema, values={"name": "John", "age": 30})
    form2.validate({"name": "", "age": "invalid"})
    assert not form2.is_valid
    assert form2.errors is not None
    assert "name" in form2.errors
    assert "age" in form2.errors

    # Test validate called twice
    form3 = Form(env=env, schema=schema, values={"name": "John", "age": 30})
    form3.validate({"name": "Jane", "age": 25})
    with pytest.raises(AssertionError):
        form3.validate({"name": "Jane", "age": 25})


# LLM-generated content at query #17
#--------------------------

```python
def test_Form_template_for_field():
    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    assert Form.template_for_field(Form, choice_field) == "forms/select.html"

    # Test Boolean field
    boolean_field = Boolean()
    assert Form.template_for_field(Form, boolean_field) == "forms/checkbox.html"

    # Test String field with text format
    text_field = String(format="text")
    assert Form.template_for_field(Form, text_field) == "forms/textarea.html"

    # Test String field without format
    string_field = String()
    assert Form.template_for_field(Form, string_field) == "forms/input.html"

    # Test Object field (should raise assertion)
    object_field = Object()
    with pytest.raises(AssertionError):
        Form.template_for_field(Form, object_field)


# LLM-generated content at query #18
#--------------------------

```python
def test_Form_validate():
    # Setup
    schema = Schema(fields={"name": String(), "age": Integer()})
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    form = Form(env=env, schema=schema, values={"name": "John", "age": 30})

    # Test validation with valid data
    form.validate({"name": "Jane", "age": 25})
    assert form.is_valid
    assert form.validated_data == {"name": "Jane", "age": 25}
    assert form.errors is None

    # Test validation with invalid data
    form2 = Form(env=env, schema=schema, values={"name": "John", "age": 30})
    form2.validate({"name": "Jane", "age": "invalid"})
    assert not form2.is_valid
    assert form2.errors is not None
    assert "age" in form2.errors

    # Test that validate() can only be called once
    form3 = Form(env=env, schema=schema, values={"name": "John", "age": 30})
    form3.validate({"name": "Jane", "age": 25})
    try:
        form3.validate({"name": "Jane", "age": 25})
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "validate() has already been called."


# LLM-generated content at query #19
#--------------------------

```python
def test_Form_template_for_field():
    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    form = Form(env=None, schema=Schema(fields={"test": choice_field}))
    assert form.template_for_field(choice_field) == "forms/select.html"

    # Test Boolean field
    boolean_field = Boolean()
    form = Form(env=None, schema=Schema(fields={"test": boolean_field}))
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"

    # Test String field with text format
    text_field = String(format="text")
    form = Form(env=None, schema=Schema(fields={"test": text_field}))
    assert form.template_for_field(text_field) == "forms/textarea.html"

    # Test String field without text format
    string_field = String()
    form = Form(env=None, schema=Schema(fields={"test": string_field}))
    assert form.template_for_field(string_field) == "forms/input.html"

    # Test Object field raises assertion
    object_field = Object(fields={"nested": String()})
    form = Form(env=None, schema=Schema(fields={"test": object_field}))
    with pytest.raises(AssertionError):
        form.template_for_field(object_field)


# LLM-generated content at query #20
#--------------------------

```python
def test_Form_validate():
    schema = Schema(fields={"name": String(), "age": Integer()})
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    form = Form(env=env, schema=schema, values={"name": "John", "age": 30})

    # Test successful validation
    form.validate({"name": "Jane", "age": 25})
    assert form.is_valid
    assert form.validated_data == {"name": "Jane", "age": 25}
    assert form.errors is None

    # Test validation with errors
    form2 = Form(env=env, schema=schema, values={"name": "John", "age": 30})
    form2.validate({"name": "", "age": "invalid"})
    assert not form2.is_valid
    assert form2.errors is not None
    assert "name" in form2.errors
    assert "age" in form2.errors

    # Test validate() called twice
    form3 = Form(env=env, schema=schema, values={"name": "John", "age": 30})
    form3.validate({"name": "Jane", "age": 25})
    with pytest.raises(AssertionError):
        form3.validate({"name": "Jane", "age": 25})


# LLM-generated content at query #21
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)


# LLM-generated content at query #22
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment()
    schema = Schema(fields={
        'text_field': String(format='text'),
        'email_field': String(format='email'),
        'number_field': String(format='number'),
        'password_field': String(format='password'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'boolean_field': Boolean(),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    text_template = jinja2.Template("""
        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">
    """)
    env.get_template = lambda name: text_template
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input type="text" name="text_field" id="text-field" value="test">' in result

    # Test email field
    email_template = jinja2.Template("""
        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">
    """)
    env.get_template = lambda name: email_template
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input type="email" name="email_field" id="email-field" value="test@example.com">' in result

    # Test number field
    number_template = jinja2.Template("""
        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">
    """)
    env.get_template = lambda name: number_template
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input type="number" name="number_field" id="number-field" value="123">' in result

    # Test password field (value should be empty string)
    password_template = jinja2.Template("""
        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">
    """)
    env.get_template = lambda name: password_template
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input type="password" name="password_field" id="password-field" value="">' in result

    # Test choice field (should use select template)
    select_template = jinja2.Template("""
        <select name="{{ field_name }}" id="{{ field_id }}">
            {% for choice in field.choices %}
            <option value="{{ choice }}">{{ choice }}</option>
            {% endfor %}
        </select>
    """)
    env.get_template = lambda name: select_template
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'])
    assert '<select name="choice_field" id="choice-field">' in result
    assert '<option value="a">a</option>' in result
    assert '<option value="b">b</option>' in result
    assert '<option value="c">c</option>' in result

    # Test boolean field (should use checkbox template)
    checkbox_template = jinja2.Template("""
        <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>
    """)
    env.get_template = lambda name: checkbox_template
    result = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=True)
    assert '<input type="checkbox" name="boolean_field" id="boolean-field" checked>' in result

    # Test with error
    text_template_with_error = jinja2.Template("""
        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
    """)
    env.get_template = lambda name: text_template_with_error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid value')
    assert '<div class="error">Invalid value</div>' in result


# LLM-generated content at query #23
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)

    # Test autoescape is True
    assert env.autoescape is True


# LLM-generated content at query #24
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #25
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals['forms/input.html'] = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    env.globals['forms/select.html'] = '<select name="{{ field_name }}"></select>'
    env.globals['forms/checkbox.html'] = '<input type="checkbox" name="{{ field_name }}">'
    env.globals['forms/textarea.html'] = '<textarea name="{{ field_name }}">{{ value }}</textarea>'

    schema = Schema(fields={
        'text_field': String(),
        'choice_field': Choice(choices=['a', 'b']),
        'bool_field': Boolean(),
        'text_area': String(format='text'),
    })

    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input type="text" name="text_field" value="test">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'])
    assert '<select name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'])
    assert '<input type="checkbox" name="bool_field">' in result

    # Test textarea field
    result = form.render_field(field_name='text_area', field=schema.fields['text_area'], value='test')
    assert '<textarea name="text_area">test</textarea>' in result

    # Test password field
    password_field = String(format='password')
    result = form.render_field(field_name='password', field=password_field, value='secret')
    assert '<input type="password" name="password" value="">' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #26
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    }))
    schema = Schema(fields={
        'text_field': String(),
        'choice_field': Choice(choices=['a', 'b']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text')
    })
    form = Form(env=env, schema=schema, values={'text_field': 'test'})

    # Test String field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test Choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test Boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test Textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result

    # Test password field
    password_field = String(format='password')
    result = form.render_field(field_name='password', field=password_field, value='secret')
    assert '<input id="password" name="password" type="password" value="">' in result


# LLM-generated content at query #27
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'text_field': String(),
        'password_field': String(format='password'),
        'email_field': String(format='email'),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text')
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test password field (value should be empty string)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='42')
    assert '<input id="number-field" name="number_field" type="number" value="42">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #28
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #29
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'text_area': String(format='text'),
    })
    form = Form(env=env, schema=schema, values={})

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (value should be empty string)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='42')
    assert '<input id="number-field" name="number_field" type="number" value="42">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field (checked)
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test boolean field (unchecked)
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=False)
    assert '<input id="bool-field" name="bool_field" type="checkbox">' in result and 'checked' not in result

    # Test textarea field
    result = form.render_field(field_name='text_area', field=schema.fields['text_area'], value='long text')
    assert '<textarea id="text-area" name="text_area">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #30
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'email': String(format='email'),
        'bio': String(format='text'),
        'active': Boolean(),
        'gender': Choice(choices=['M', 'F', 'O'])
    })
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    }))
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'email': 'john@example.com', 'bio': 'Developer', 'active': True, 'gender': 'M'})

    # Test with no errors
    form.validate({'name': 'John', 'age': '30', 'email': 'john@example.com', 'bio': 'Developer', 'active': True, 'gender': 'M'})
    rendered = form.render_fields()
    assert '<input type="text" name="name" value="John">' in rendered
    assert '<input type="number" name="age" value="30">' in rendered
    assert '<input type="email" name="email" value="john@example.com">' in rendered
    assert '<textarea name="bio">Developer</textarea>' in rendered
    assert '<input type="checkbox" name="active" checked>' in rendered
    assert '<select name="gender"><option value="M">M</option><option value="F">F</option><option value="O">O</option></select>' in rendered

    # Test with errors
    form_with_errors = Form(env=env, schema=schema)
    form_with_errors.validate({'name': '', 'age': 'invalid', 'email': 'invalid', 'bio': '', 'active': 'invalid', 'gender': 'invalid'})
    rendered_with_errors = form_with_errors.render_fields()
    assert 'name="name"' in rendered_with_errors
    assert 'name="age"' in rendered_with_errors
    assert 'name="email"' in rendered_with_errors
    assert 'name="bio"' in rendered_with_errors
    assert 'name="active"' in rendered_with_errors
    assert 'name="gender"' in rendered_with_errors

    # Test with read-only field
    schema_with_readonly = Schema(fields={'readonly_field': String(read_only=True)})
    form_readonly = Form(env=env, schema=schema_with_readonly)
    form_readonly.validate({})
    rendered_readonly = form_readonly.render_fields()
    assert 'readonly_field' not in rendered_readonly


# LLM-generated content at query #31
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="templates", package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #33
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'number_field': String(format='number'),
        'password_field': String(format='password'),
        'textarea_field': String(format='text'),
        'select_field': Choice(choices=['a', 'b', 'c']),
        'checkbox_field': Boolean(),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test password field (value should be empty string)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test select field
    result = form.render_field(field_name='select_field', field=schema.fields['select_field'])
    assert '<select id="select-field" name="select_field"></select>' in result

    # Test checkbox field (unchecked)
    result = form.render_field(field_name='checkbox_field', field=schema.fields['checkbox_field'], value=False)
    assert '<input id="checkbox-field" name="checkbox_field" type="checkbox">' in result
    assert 'checked' not in result

    # Test checkbox field (checked)
    result = form.render_field(field_name='checkbox_field', field=schema.fields['checkbox_field'], value=True)
    assert 'checked' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #34
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with neither directory nor package raises AssertionError
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #35
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text')
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (value should be empty)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='42')
    assert '<input id="number-field" name="number_field" type="number" value="42">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #36
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization without directory or package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #37
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals["forms/select.html"] = "<select>{{ field_name }}</select>"
    env.globals["forms/checkbox.html"] = "<input type='checkbox' {{ field_name }} />"
    env.globals["forms/textarea.html"] = "<textarea>{{ field_name }}</textarea>"
    env.globals["forms/input.html"] = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}' />"

    schema = Schema(fields={
        "text_field": String(),
        "choice_field": Choice(choices=[("a", "A"), ("b", "B")]),
        "bool_field": Boolean(),
        "text_area": String(format="text"),
    })

    form = Form(env=env, schema=schema, values={"text_field": "test"})

    # Test String field
    result = form.render_field(
        field_name="text_field",
        field=schema.fields["text_field"],
        value="test_value",
        error=None
    )
    assert "<input type='text' name='text_field' value='test_value' />" in result

    # Test Choice field
    result = form.render_field(
        field_name="choice_field",
        field=schema.fields["choice_field"],
        value="a",
        error=None
    )
    assert "<select>choice_field</select>" in result

    # Test Boolean field
    result = form.render_field(
        field_name="bool_field",
        field=schema.fields["bool_field"],
        value=True,
        error=None
    )
    assert "<input type='checkbox' bool_field />" in result

    # Test TextArea field
    result = form.render_field(
        field_name="text_area",
        field=schema.fields["text_area"],
        value="long text",
        error=None
    )
    assert "<textarea>text_area</textarea>" in result

    # Test with error
    result = form.render_field(
        field_name="text_field",
        field=schema.fields["text_field"],
        value="test_value",
        error="Invalid value"
    )
    assert "Invalid value" in result

    # Test password field
    password_field = String(format="password")
    result = form.render_field(
        field_name="password",
        field=password_field,
        value="secret",
        error=None
    )
    assert "<input type='password' name='password' value='' />" in result


# LLM-generated content at query #38
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #39
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #40
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text'),
    })
    form = Form(env=env, schema=schema, values={})

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (value should be empty string)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='42')
    assert '<input id="number-field" name="number_field" type="number" value="42">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field (checked when True)
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test boolean field (not checked when False)
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=False)
    assert '<input id="bool-field" name="bool_field" type="checkbox">' in result and 'checked' not in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #41
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms1 = Jinja2Forms(directory="test_directory")
    assert forms1.env is not None
    assert isinstance(forms1.env.loader, jinja2.FileSystemLoader)

    # Test initialization with package
    forms2 = Jinja2Forms(package="test_package")
    assert forms2.env is not None
    assert isinstance(forms2.env.loader, jinja2.PackageLoader)

    # Test initialization with both directory and package
    forms3 = Jinja2Forms(directory="test_directory", package="test_package")
    assert forms3.env is not None
    assert isinstance(forms3.env.loader, jinja2.ChoiceLoader)

    # Test error when neither directory nor package is provided
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #42
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

    # Test with package only
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)


# LLM-generated content at query #43
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)


# LLM-generated content at query #44
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization without directory or package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #45
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'text_field': String(format='text'),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text')
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (value should be empty)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='42')
    assert '<input id="number-field" name="number_field" type="number" value="42">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #46
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test initialization with neither directory nor package raises AssertionError
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #47
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))

    schema = Schema(fields={
        'name': String(max_length=100),
        'age': String(format='number'),
        'description': String(format='text'),
        'active': Boolean(),
        'gender': Choice(choices=['M', 'F'])
    })

    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'description': 'Test', 'active': True, 'gender': 'M'})

    # Test without validation
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in html
    assert '<input id="age" name="age" type="number" value="30">' in html
    assert '<textarea id="description" name="description">Test</textarea>' in html
    assert '<input id="active" name="active" type="checkbox" checked>' in html
    assert '<select id="gender" name="gender"></select>' in html

    # Test with validation
    form.validate({'name': 'Jane', 'age': '25', 'description': 'New test', 'active': False, 'gender': 'F'})
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="Jane">' in html
    assert '<input id="age" name="age" type="number" value="25">' in html
    assert '<textarea id="description" name="description">New test</textarea>' in html
    assert '<input id="active" name="active" type="checkbox" >' in html
    assert '<select id="gender" name="gender"></select>' in html

    # Test with errors
    form.validate({'name': '', 'age': 'invalid', 'description': 'Test', 'active': True, 'gender': 'X'})
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="">' in html
    assert '<input id="age" name="age" type="number" value="invalid">' in html
    assert '<textarea id="description" name="description">Test</textarea>' in html
    assert '<input id="active" name="active" type="checkbox" checked>' in html
    assert '<select id="gender" name="gender"></select>' in html


# LLM-generated content at query #48
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)

    # Test autoescape is True
    assert env.autoescape is True


# LLM-generated content at query #49
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test initialization with neither directory nor package raises assertion error
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #50
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }), autoescape=True)

    schema = Schema(fields={
        'name': String(format='text'),
        'email': String(format='email'),
        'age': String(format='number'),
        'bio': String(format='text'),
        'active': Boolean()
    })

    form = Form(env=env, schema=schema, values={'name': 'John', 'email': 'john@example.com', 'age': '30', 'bio': 'Developer', 'active': True})

    # Test without validation
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in html
    assert '<input id="email" name="email" type="email" value="john@example.com">' in html
    assert '<input id="age" name="age" type="number" value="30">' in html
    assert '<textarea id="bio" name="bio">Developer</textarea>' in html
    assert '<input id="active" name="active" type="checkbox" checked>' in html

    # Test with validation errors
    form.validate({'name': '', 'email': 'invalid', 'age': 'abc', 'bio': '', 'active': 'not_bool'})
    html = form.render_fields()
    assert 'value=""' in html
    assert 'value="invalid"' in html
    assert 'value="abc"' in html
    assert '<textarea id="bio" name="bio"></textarea>' in html
    assert '<input id="active" name="active" type="checkbox">' in html


# LLM-generated content at query #51
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'bio': String(format='text'),
        'active': Boolean(),
        'gender': Choice(choices=['M', 'F', 'O'])
    })
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'bio': 'Test bio', 'active': True, 'gender': 'M'})

    # Test without validation
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in html
    assert '<input id="age" name="age" type="number" value="30">' in html
    assert '<textarea id="bio" name="bio">Test bio</textarea>' in html
    assert '<input id="active" name="active" type="checkbox" checked>' in html
    assert '<select id="gender" name="gender"></select>' in html

    # Test with validation and errors
    form.validate({'name': '', 'age': 'invalid', 'bio': '', 'active': 'not_bool', 'gender': 'X'})
    html = form.render_fields()
    assert 'value=""' in html  # Empty name
    assert 'value="invalid"' in html  # Invalid age
    assert '<textarea id="bio" name="bio"></textarea>' in html  # Empty bio
    assert '<input id="active" name="active" type="checkbox"' in html  # Invalid active
    assert '<select id="gender" name="gender"></select>' in html  # Invalid gender

    # Test with read-only field
    schema_readonly = Schema(fields={'readonly_field': String(read_only=True)})
    form_readonly = Form(env=env, schema=schema_readonly)
    html = form_readonly.render_fields()
    assert html == ''  # Read-only fields should not be rendered


# LLM-generated content at query #52
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #53
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="templates", package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #54
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #55
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="templates", package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #56
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'name': String(format='text'),
        'age': String(format='number'),
        'email': String(format='email'),
        'agree': Boolean(),
        'country': Choice(choices=['US', 'UK', 'CA']),
    })
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'email': 'john@example.com', 'agree': True, 'country': 'US'})

    # Test without validation
    result = form.render_fields()
    assert '<textarea id="name" name="name">John</textarea>' in result
    assert '<input id="age" name="age" type="number" value="30">' in result
    assert '<input id="email" name="email" type="email" value="john@example.com">' in result
    assert '<input id="agree" name="agree" type="checkbox" checked>' in result
    assert '<select id="country" name="country"></select>' in result

    # Test with validation
    form.validate({'name': 'Jane', 'age': '25', 'email': 'jane@example.com', 'agree': False, 'country': 'UK'})
    result = form.render_fields()
    assert '<textarea id="name" name="name">Jane</textarea>' in result
    assert '<input id="age" name="age" type="number" value="25">' in result
    assert '<input id="email" name="email" type="email" value="jane@example.com">' in result
    assert '<input id="agree" name="agree" type="checkbox">' in result
    assert '<select id="country" name="country"></select>' in result

    # Test with errors
    form.validate({'name': '', 'age': 'invalid', 'email': 'invalid', 'country': 'invalid'})
    result = form.render_fields()
    assert 'error' in result.lower()


# LLM-generated content at query #57
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #58
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))

    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'email': String(format='email'),
        'bio': String(format='text'),
        'active': Boolean(),
        'status': Choice(choices=['active', 'inactive'])
    })

    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'email': 'john@example.com', 'bio': 'Test bio', 'active': True, 'status': 'active'})

    # Test without validation
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in html
    assert '<input id="age" name="age" type="number" value="30">' in html
    assert '<input id="email" name="email" type="email" value="john@example.com">' in html
    assert '<textarea id="bio" name="bio">Test bio</textarea>' in html
    assert '<input id="active" name="active" type="checkbox" checked>' in html
    assert '<select id="status" name="status"></select>' in html

    # Test with validation errors
    form.validate({'name': '', 'age': 'invalid', 'email': 'invalid', 'bio': '', 'active': 'not-boolean', 'status': 'invalid'})
    html = form.render_fields()
    assert 'value=""' in html  # Empty name
    assert 'value="invalid"' in html  # Invalid age
    assert 'value="invalid"' in html  # Invalid email
    assert '<textarea id="bio" name="bio"></textarea>' in html  # Empty bio
    assert '<input id="active" name="active" type="checkbox">' in html  # Invalid boolean
    assert '<select id="status" name="status"></select>' in html  # Invalid choice

    # Test with read-only field
    schema_readonly = Schema(fields={'readonly_field': String(read_only=True)})
    form_readonly = Form(env=env, schema=schema_readonly, values={'readonly_field': 'test'})
    html = form_readonly.render_fields()
    assert 'readonly_field' not in html


# LLM-generated content at query #59
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>",
        "forms/select.html": "<select id='{{ field_id }}' name='{{ field_name }}'></select>",
        "forms/checkbox.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox'>",
        "forms/textarea.html": "<textarea id='{{ field_id }}' name='{{ field_name }}'></textarea>",
    }))
    schema = Schema(fields={
        "name": String(),
        "age": String(format="number"),
        "email": String(format="email"),
        "bio": String(format="text"),
        "active": Boolean(),
        "gender": Choice(choices=["M", "F", "O"])
    })
    form = Form(env=env, schema=schema, values={"name": "John", "age": "30", "email": "john@example.com", "bio": "Hello", "active": True, "gender": "M"})

    # Test without validation
    html = form.render_fields()
    assert "<input id='name' name='name' type='text' value='John'>" in html
    assert "<input id='age' name='age' type='number' value='30'>" in html
    assert "<input id='email' name='email' type='email' value='john@example.com'>" in html
    assert "<textarea id='bio' name='bio'></textarea>" in html
    assert "<input id='active' name='active' type='checkbox'>" in html
    assert "<select id='gender' name='gender'></select>" in html

    # Test with validation and errors
    form.validate({"name": "", "age": "invalid", "email": "invalid", "bio": "", "active": "not-boolean", "gender": "invalid"})
    html = form.render_fields()
    assert "error" in html.lower()


# LLM-generated content at query #60
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'number_field': String(format='number'),
        'password_field': String(format='password'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text'),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test password field (value should be empty)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #61
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals['forms/input.html'] = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    env.globals['forms/select.html'] = '<select name="{{ field_name }}"></select>'
    env.globals['forms/checkbox.html'] = '<input type="checkbox" name="{{ field_name }}">'
    env.globals['forms/textarea.html'] = '<textarea name="{{ field_name }}">{{ value }}</textarea>'

    schema = Schema(fields={
        'text_field': String(),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b']),
        'bool_field': Boolean(),
        'text_area': String(format='text'),
        'password_field': String(format='password'),
    })

    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(
        field_name='text_field',
        field=schema.fields['text_field'],
        value='test_value'
    )
    assert '<input type="text" name="text_field" value="test_value">' in result

    # Test number field
    result = form.render_field(
        field_name='number_field',
        field=schema.fields['number_field'],
        value='123'
    )
    assert '<input type="number" name="number_field" value="123">' in result

    # Test choice field
    result = form.render_field(
        field_name='choice_field',
        field=schema.fields['choice_field'],
        value='a'
    )
    assert '<select name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(
        field_name='bool_field',
        field=schema.fields['bool_field'],
        value=True
    )
    assert '<input type="checkbox" name="bool_field">' in result

    # Test textarea field
    result = form.render_field(
        field_name='text_area',
        field=schema.fields['text_area'],
        value='long text'
    )
    assert '<textarea name="text_area">long text</textarea>' in result

    # Test password field (value should be empty)
    result = form.render_field(
        field_name='password_field',
        field=schema.fields['password_field'],
        value='secret'
    )
    assert '<input type="password" name="password_field" value="">' in result

    # Test with error
    result = form.render_field(
        field_name='text_field',
        field=schema.fields['text_field'],
        value='test_value',
        error='Invalid value'
    )
    assert 'Invalid value' in result


# LLM-generated content at query #62
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'email': String(format='email'),
        'bio': String(format='text'),
        'agree': Boolean(),
        'country': Choice(choices=['US', 'UK', 'CA'])
    })
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'email': 'john@example.com', 'bio': 'Hello', 'agree': True, 'country': 'US'})

    # Test without validation
    rendered = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in rendered
    assert '<input id="age" name="age" type="number" value="30">' in rendered
    assert '<input id="email" name="email" type="email" value="john@example.com">' in rendered
    assert '<textarea id="bio" name="bio">Hello</textarea>' in rendered
    assert '<input id="agree" name="agree" type="checkbox" checked>' in rendered
    assert '<select id="country" name="country"><option value="US">US</option><option value="UK">UK</option><option value="CA">CA</option></select>' in rendered

    # Test with validation errors
    form.validate({'name': '', 'age': 'invalid', 'email': 'invalid', 'bio': '', 'agree': False, 'country': 'INVALID'})
    rendered = form.render_fields()
    assert '<input id="name" name="name" type="text" value="">' in rendered
    assert '<input id="age" name="age" type="number" value="invalid">' in rendered
    assert '<input id="email" name="email" type="email" value="invalid">' in rendered
    assert '<textarea id="bio" name="bio"></textarea>' in rendered
    assert '<input id="agree" name="agree" type="checkbox" >' in rendered
    assert '<select id="country" name="country"><option value="US">US</option><option value="UK">UK</option><option value="CA">CA</option></select>' in rendered


# LLM-generated content at query #63
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

    # Test with package only
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)


# LLM-generated content at query #64
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert isinstance(forms.env, jinja2.Environment)
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert isinstance(forms.env, jinja2.Environment)
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert isinstance(forms.env, jinja2.Environment)
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)
    assert len(forms.env.loader.loaders) == 2
    assert isinstance(forms.env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(forms.env.loader.loaders[1], jinja2.PackageLoader)

    # Test autoescape is enabled
    assert forms.env.autoescape is True


# LLM-generated content at query #65
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'description': String(format='text'),
        'active': Boolean(),
        'gender': Choice(choices=['M', 'F'])
    })
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'description': 'Test', 'active': True, 'gender': 'M'})

    # Test without validation
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in html
    assert '<input id="age" name="age" type="number" value="30">' in html
    assert '<textarea id="description" name="description">Test</textarea>' in html
    assert '<input id="active" name="active" type="checkbox" checked>' in html
    assert '<select id="gender" name="gender"></select>' in html

    # Test with validation errors
    form.validate({'name': '', 'age': 'invalid', 'description': '', 'active': 'not_bool', 'gender': 'X'})
    html = form.render_fields()
    assert 'value=""' in html  # Empty name
    assert 'value="invalid"' in html  # Invalid age
    assert '<textarea id="description" name="description"></textarea>' in html  # Empty description
    assert '<input id="active" name="active" type="checkbox">' in html  # Invalid boolean
    assert '<select id="gender" name="gender"></select>' in html  # Invalid choice

    # Test with read-only field
    schema_readonly = Schema(fields={'readonly_field': String(read_only=True)})
    form_readonly = Form(env=env, schema=schema_readonly)
    html = form_readonly.render_fields()
    assert html == ''  # Read-only fields should not be rendered


# LLM-generated content at query #66
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }), autoescape=True)

    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'bio': String(format='text'),
        'country': Choice(choices=['US', 'UK', 'CA']),
        'subscribe': Boolean()
    })

    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'bio': 'Developer', 'country': 'US', 'subscribe': True})

    # Test without validation
    result = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in result
    assert '<input id="age" name="age" type="number" value="30">' in result
    assert '<textarea id="bio" name="bio">Developer</textarea>' in result
    assert '<select id="country" name="country"></select>' in result
    assert '<input id="subscribe" name="subscribe" type="checkbox" checked>' in result

    # Test with validation errors
    form.validate({'name': '', 'age': 'invalid', 'bio': 'Short', 'country': 'XX', 'subscribe': 'not-boolean'})
    result = form.render_fields()
    assert '<input id="name" name="name" type="text" value="">' in result
    assert '<input id="age" name="age" type="number" value="invalid">' in result
    assert '<textarea id="bio" name="bio">Short</textarea>' in result
    assert '<select id="country" name="country"></select>' in result
    assert '<input id="subscribe" name="subscribe" type="checkbox">' in result


# LLM-generated content at query #67
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #68
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #69
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #70
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)

    # Test autoescape is enabled
    assert env.autoescape is True


# LLM-generated content at query #71
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals["forms/input.html"] = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    env.globals["forms/select.html"] = "<select id='{{ field_id }}' name='{{ field_name }}'>{% for choice in field.choices %}<option value='{{ choice.value }}'>{{ choice.display_name }}</option>{% endfor %}</select>"
    env.globals["forms/checkbox.html"] = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    env.globals["forms/textarea.html"] = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"

    schema = Schema(fields={
        "text_field": String(),
        "number_field": String(format="number"),
        "password_field": String(format="password"),
        "choice_field": Choice(choices=[("1", "Option 1"), ("2", "Option 2")]),
        "boolean_field": Boolean(),
        "text_area_field": String(format="text"),
    })

    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test")
    assert "<input id='text-field' name='text_field' type='text' value='test'>" in result

    # Test number field
    result = form.render_field(field_name="number_field", field=schema.fields["number_field"], value="123")
    assert "<input id='number-field' name='number_field' type='number' value='123'>" in result

    # Test password field
    result = form.render_field(field_name="password_field", field=schema.fields["password_field"], value="secret")
    assert "<input id='password-field' name='password_field' type='password' value=''>" in result

    # Test choice field
    result = form.render_field(field_name="choice_field", field=schema.fields["choice_field"], value="1")
    assert "<select id='choice-field' name='choice_field'>" in result
    assert "<option value='1'>Option 1</option>" in result
    assert "<option value='2'>Option 2</option>" in result

    # Test boolean field
    result = form.render_field(field_name="boolean_field", field=schema.fields["boolean_field"], value=True)
    assert "<input id='boolean-field' name='boolean_field' type='checkbox' checked>" in result

    # Test text area field
    result = form.render_field(field_name="text_area_field", field=schema.fields["text_area_field"], value="long text")
    assert "<textarea id='text-area-field' name='text_area_field'>long text</textarea>" in result

    # Test with error
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test", error="Invalid value")
    assert "Invalid value" in result


# LLM-generated content at query #72
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test initialization with package
    forms = Jinja2Forms(package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test initialization with neither directory nor package raises AssertionError
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #73
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #74
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }), autoescape=True)

    schema = Schema(fields={
        'name': String(max_length=100),
        'age': String(format='number'),
        'bio': String(format='text'),
        'gender': Choice(choices=['M', 'F', 'O']),
        'active': Boolean()
    })

    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'bio': 'Developer', 'gender': 'M', 'active': True})

    # Test
    result = form.render_fields()

    # Assert
    assert '<input id="name" name="name" type="text" value="John">' in result
    assert '<input id="age" name="age" type="number" value="30">' in result
    assert '<textarea id="bio" name="bio">Developer</textarea>' in result
    assert '<select id="gender" name="gender"></select>' in result
    assert '<input id="active" name="active" type="checkbox" checked>' in result


# LLM-generated content at query #75
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>",
        "forms/select.html": "<select id='{{ field_id }}' name='{{ field_name }}'></select>",
        "forms/checkbox.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>",
        "forms/textarea.html": "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>",
    }))
    schema = Schema(fields={
        "username": String(format="text"),
        "password": String(format="password"),
        "age": String(format="number"),
        "bio": String(format="text"),
        "country": Choice(choices=["US", "UK", "CA"]),
        "subscribe": Boolean(),
    })
    form = Form(env=env, schema=schema, values={"username": "test", "age": "25", "bio": "Hello", "country": "US", "subscribe": True})

    # Test String field with text format (textarea)
    result = form.render_field(field_name="username", field=schema.fields["username"], value="test")
    assert "<input id='username' name='username' type='text' value='test'>" in result

    # Test String field with password format
    result = form.render_field(field_name="password", field=schema.fields["password"], value="secret")
    assert "<input id='password' name='password' type='password' value=''>" in result

    # Test String field with number format
    result = form.render_field(field_name="age", field=schema.fields["age"], value="25")
    assert "<input id='age' name='age' type='number' value='25'>" in result

    # Test String field with text format (textarea)
    result = form.render_field(field_name="bio", field=schema.fields["bio"], value="Hello")
    assert "<textarea id='bio' name='bio'>Hello</textarea>" in result

    # Test Choice field (select)
    result = form.render_field(field_name="country", field=schema.fields["country"], value="US")
    assert "<select id='country' name='country'></select>" in result

    # Test Boolean field (checkbox)
    result = form.render_field(field_name="subscribe", field=schema.fields["subscribe"], value=True)
    assert "<input id='subscribe' name='subscribe' type='checkbox' checked>" in result

    # Test with error
    result = form.render_field(field_name="username", field=schema.fields["username"], value="test", error="Invalid username")
    assert "Invalid username" in result


# LLM-generated content at query #76
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'text_area_field': String(format='text')
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (value should be empty)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='42')
    assert '<input id="number-field" name="number_field" type="number" value="42">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='text_area_field', field=schema.fields['text_area_field'], value='long text')
    assert '<textarea id="text-area-field" name="text_area_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Form_input_type_for_field():
    # Setup
    form = Form(
        env=jinja2.Environment(),
        schema=Schema(fields={})
    )

    # Test default case (no format)
    field = String()
    assert form.input_type_for_field(field) == "text"

    # Test supported formats
    for format, input_type in form.FORMAT_TO_INPUTTYPE.items():
        field = String(format=format)
        assert form.input_type_for_field(field) == input_type

    # Test unsupported format
    field = String(format="unsupported")
    assert form.input_type_for_field(field) == "text"

    # Test non-String field with format
    field = Boolean(format="color")
    assert form.input_type_for_field(field) == "text"


# LLM-generated content at query #2
#--------------------------

```python
def test_Form_template_for_field():
    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    assert Form.template_for_field(choice_field) == "forms/select.html"

    # Test Boolean field
    boolean_field = Boolean()
    assert Form.template_for_field(boolean_field) == "forms/checkbox.html"

    # Test String field with text format
    text_field = String(format="text")
    assert Form.template_for_field(text_field) == "forms/textarea.html"

    # Test String field without format
    string_field = String()
    assert Form.template_for_field(string_field) == "forms/input.html"

    # Test other field types
    number_field = Field()
    assert Form.template_for_field(number_field) == "forms/input.html"

    # Test Object field raises assertion
    object_field = Object()
    with pytest.raises(AssertionError):
        Form.template_for_field(object_field)


# LLM-generated content at query #3
#--------------------------

```python
def test_Form___str__():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'description': String(format='text'),
        'active': Boolean()
    })
    form = Form(env=env, schema=schema, values={'name': 'Test', 'age': '25', 'description': 'Test description', 'active': True})

    # Exercise
    result = str(form)

    # Verify
    assert '<input id="name" name="name" type="text" value="Test">' in result
    assert '<input id="age" name="age" type="number" value="25">' in result
    assert '<textarea id="description" name="description">Test description</textarea>' in result
    assert '<input id="active" name="active" type="checkbox" checked>' in result


# LLM-generated content at query #4
#--------------------------

```python
def test_Form___html__():
    # Setup
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'bio': String(format='text'),
        'agree': Boolean(),
        'country': Choice(choices=['US', 'UK', 'CA'])
    })
    jinja_env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    }), autoescape=True)
    form = Form(env=jinja_env, schema=schema, values={'name': 'John', 'age': '30', 'bio': 'Developer', 'agree': True, 'country': 'US'})

    # Exercise
    result = form.__html__()

    # Verify
    assert isinstance(result, markupsafe.Markup)
    assert str(result) == form.render_fields()


# LLM-generated content at query #5
#--------------------------

```python
def test_Form_validate():
    # Setup
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
    })
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '25'})

    # Test valid data
    form.validate({'name': 'Jane', 'age': '30'})
    assert form.is_valid
    assert form.validated_data == {'name': 'Jane', 'age': '30'}
    assert form.errors is None

    # Test invalid data (missing required field)
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '25'})
    form.validate({'name': ''})
    assert not form.is_valid
    assert form.errors is not None
    assert 'age' in form.errors

    # Test validate() called twice
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '25'})
    form.validate({'name': 'Jane', 'age': '30'})
    with pytest.raises(AssertionError):
        form.validate({'name': 'Jane', 'age': '30'})


# LLM-generated content at query #6
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'textarea_field': String(format='text'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'boolean_field': Boolean()
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (value should be empty)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field (checked)
    result = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=True)
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" checked>' in result

    # Test boolean field (unchecked)
    result = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=False)
    assert '<input id="boolean-field" name="boolean_field" type="checkbox">' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid value')
    assert 'Invalid value' in result


# LLM-generated content at query #7
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'email': String(format='email'),
        'bio': String(format='text'),
        'agree': Boolean(),
        'country': Choice(choices=['US', 'UK', 'CA'])
    })
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'email': 'john@example.com', 'bio': 'Test bio', 'agree': True, 'country': 'US'})

    # Test without validation
    rendered = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in rendered
    assert '<input id="age" name="age" type="number" value="30">' in rendered
    assert '<input id="email" name="email" type="email" value="john@example.com">' in rendered
    assert '<textarea id="bio" name="bio">Test bio</textarea>' in rendered
    assert '<input id="agree" name="agree" type="checkbox" checked>' in rendered
    assert '<select id="country" name="country"><option value="US">US</option><option value="UK">UK</option><option value="CA">CA</option></select>' in rendered

    # Test with validation errors
    form.validate({'name': '', 'age': 'invalid', 'email': 'invalid', 'bio': '', 'agree': False, 'country': 'invalid'})
    rendered = form.render_fields()
    assert 'error' in rendered.lower()


# LLM-generated content at query #8
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>",
        "forms/textarea.html": "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>",
        "forms/select.html": "<select id='{{ field_id }}' name='{{ field_name }}'></select>",
        "forms/checkbox.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    }))
    schema = Schema(fields={
        "text_field": String(format="text"),
        "email_field": String(format="email"),
        "number_field": String(format="number"),
        "password_field": String(format="password"),
        "choice_field": Choice(choices=["a", "b", "c"]),
        "boolean_field": Boolean(),
        "textarea_field": String(format="text"),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test")
    assert "<input id='text-field' name='text_field' type='text' value='test'>" in result

    # Test email field
    result = form.render_field(field_name="email_field", field=schema.fields["email_field"], value="test@example.com")
    assert "<input id='email-field' name='email_field' type='email' value='test@example.com'>" in result

    # Test number field
    result = form.render_field(field_name="number_field", field=schema.fields["number_field"], value="123")
    assert "<input id='number-field' name='number_field' type='number' value='123'>" in result

    # Test password field (value should be empty string)
    result = form.render_field(field_name="password_field", field=schema.fields["password_field"], value="secret")
    assert "<input id='password-field' name='password_field' type='password' value=''>" in result

    # Test choice field
    result = form.render_field(field_name="choice_field", field=schema.fields["choice_field"], value="a")
    assert "<select id='choice-field' name='choice_field'></select>" in result

    # Test boolean field
    result = form.render_field(field_name="boolean_field", field=schema.fields["boolean_field"], value=True)
    assert "<input id='boolean-field' name='boolean_field' type='checkbox' checked>" in result

    # Test textarea field
    result = form.render_field(field_name="textarea_field", field=schema.fields["textarea_field"], value="long text")
    assert "<textarea id='textarea-field' name='textarea_field'>long text</textarea>" in result

    # Test with error
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test", error="Invalid")
    assert "Invalid" in result


# LLM-generated content at query #9
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals.update({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        'forms/select.html': '<select name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}">',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    })

    schema = Schema(fields={
        'text_field': String(),
        'choice_field': Choice(choices=['a', 'b']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text')
    })

    form = Form(env=env, schema=schema)

    # Test String field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input type="text" name="text_field" value="test">' in result

    # Test Choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'])
    assert '<select name="choice_field"></select>' in result

    # Test Boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'])
    assert '<input type="checkbox" name="bool_field">' in result

    # Test Textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='test')
    assert '<textarea name="textarea_field">test</textarea>' in result

    # Test password field
    password_field = String(format='password')
    result = form.render_field(field_name='password', field=password_field, value='secret')
    assert '<input type="password" name="password" value="">' in result

    # Test error handling
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], error='Error message')
    assert 'Error message' in result


# LLM-generated content at query #10
#--------------------------

```python
def test_Form_template_for_field():
    # Test Choice field
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    form = Form(env=None, schema=Schema(fields={"test": choice_field}))
    assert form.template_for_field(choice_field) == "forms/select.html"

    # Test Boolean field
    boolean_field = Boolean()
    form = Form(env=None, schema=Schema(fields={"test": boolean_field}))
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"

    # Test String field with text format
    text_field = String(format="text")
    form = Form(env=None, schema=Schema(fields={"test": text_field}))
    assert form.template_for_field(text_field) == "forms/textarea.html"

    # Test String field without text format
    string_field = String()
    form = Form(env=None, schema=Schema(fields={"test": string_field}))
    assert form.template_for_field(string_field) == "forms/input.html"

    # Test other field types
    number_field = Field()
    form = Form(env=None, schema=Schema(fields={"test": number_field}))
    assert form.template_for_field(number_field) == "forms/input.html"

    # Test Object field raises assertion
    object_field = Object(fields={})
    form = Form(env=None, schema=Schema(fields={"test": object_field}))
    with pytest.raises(AssertionError):
        form.template_for_field(object_field)


# LLM-generated content at query #11
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
    }))
    schema = Schema(fields={
        'text_field': String(format='text'),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'default_field': String(),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (should have empty value)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='42')
    assert '<input id="number-field" name="number_field" type="number" value="42">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test default field (should be text type)
    result = form.render_field(field_name='default_field', field=schema.fields['default_field'], value='default')
    assert '<input id="default-field" name="default_field" type="text" value="default">' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #12
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #13
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=True)
    schema = Schema(fields={
        "username": String(format="text"),
        "email": String(format="email"),
        "age": String(format="number"),
        "password": String(format="password"),
        "bio": String(format="text"),
        "agree": Boolean(),
        "country": Choice(choices=["US", "UK", "CA"]),
    })
    form = Form(env=env, schema=schema)

    # Test String field with text format (should use textarea template)
    assert form.template_for_field(schema.fields["username"]) == "forms/input.html"
    assert form.input_type_for_field(schema.fields["username"]) == "text"

    # Test String field with email format
    assert form.template_for_field(schema.fields["email"]) == "forms/input.html"
    assert form.input_type_for_field(schema.fields["email"]) == "email"

    # Test String field with number format
    assert form.template_for_field(schema.fields["age"]) == "forms/input.html"
    assert form.input_type_for_field(schema.fields["age"]) == "number"

    # Test String field with password format
    assert form.template_for_field(schema.fields["password"]) == "forms/input.html"
    assert form.input_type_for_field(schema.fields["password"]) == "password"

    # Test String field with text format (should use textarea)
    assert form.template_for_field(schema.fields["bio"]) == "forms/textarea.html"
    assert form.input_type_for_field(schema.fields["bio"]) == "text"

    # Test Boolean field (should use checkbox template)
    assert form.template_for_field(schema.fields["agree"]) == "forms/checkbox.html"
    assert form.input_type_for_field(schema.fields["agree"]) == "text"

    # Test Choice field (should use select template)
    assert form.template_for_field(schema.fields["country"]) == "forms/select.html"
    assert form.input_type_for_field(schema.fields["country"]) == "text"

    # Test render_field with mock template
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>",
        "forms/textarea.html": "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>",
        "forms/select.html": "<select id='{{ field_id }}' name='{{ field_name }}'></select>",
        "forms/checkbox.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {{ 'checked' if value else '' }}>",
    }), autoescape=True)
    form = Form(env=env, schema=schema)

    # Test rendering different field types
    assert "username" in form.render_field(field_name="username", field=schema.fields["username"], value="test")
    assert "email" in form.render_field(field_name="email", field=schema.fields["email"], value="test@example.com")
    assert "password" in form.render_field(field_name="password", field=schema.fields["password"], value="secret")
    assert "bio" in form.render_field(field_name="bio", field=schema.fields["bio"], value="Long text here")
    assert "agree" in form.render_field(field_name="agree", field=schema.fields["agree"], value=True)
    assert "country" in form.render_field(field_name="country", field=schema.fields["country"], value="US")

    # Test error rendering
    error_html = form.render_field(field_name="username", field=schema.fields["username"], value="", error="This field is required")
    assert "This field is required" in error_html


# LLM-generated content at query #14
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'email': String(format='email'),
        'description': String(format='text'),
        'agree': Boolean(),
    })
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'email': 'john@example.com', 'description': 'Test', 'agree': True})

    # Test without validation
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in html
    assert '<input id="age" name="age" type="number" value="30">' in html
    assert '<input id="email" name="email" type="email" value="john@example.com">' in html
    assert '<textarea id="description" name="description">Test</textarea>' in html
    assert '<input id="agree" name="agree" type="checkbox" checked>' in html

    # Test with validation
    form.validate({'name': 'Jane', 'age': '25', 'email': 'jane@example.com', 'description': 'Updated', 'agree': False})
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="Jane">' in html
    assert '<input id="age" name="age" type="number" value="25">' in html
    assert '<input id="email" name="email" type="email" value="jane@example.com">' in html
    assert '<textarea id="description" name="description">Updated</textarea>' in html
    assert '<input id="agree" name="agree" type="checkbox">' in html

    # Test with errors
    form.validate({'name': '', 'age': 'invalid', 'email': 'invalid', 'description': '', 'agree': False})
    html = form.render_fields()
    assert 'error' in html.lower()


# LLM-generated content at query #15
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test initialization with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #16
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'text_area': String(format='text'),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='text_area', field=schema.fields['text_area'], value='long text')
    assert '<textarea id="text-area" name="text_area">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #17
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #18
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b']),
        'bool_field': Boolean(),
        'text_area': String(format='text'),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='text_area', field=schema.fields['text_area'], value='long text')
    assert '<textarea id="text-area" name="text_area">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #19
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package (should raise assertion error)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #20
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    }), autoescape=True)

    schema = Schema(fields={
        'text_field': String(),
        'choice_field': Choice(choices=['a', 'b']),
        'boolean_field': Boolean(),
        'text_area_field': String(format='text'),
        'password_field': String(format='password')
    })

    form = Form(env=env, schema=schema, values={
        'text_field': 'test',
        'choice_field': 'a',
        'boolean_field': True,
        'text_area_field': 'long text',
        'password_field': 'secret'
    })

    # Test text field
    html = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in html

    # Test choice field
    html = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in html

    # Test boolean field
    html = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=True)
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" checked>' in html

    # Test text area field
    html = form.render_field(field_name='text_area_field', field=schema.fields['text_area_field'], value='long text')
    assert '<textarea id="text-area-field" name="text_area_field">long text</textarea>' in html

    # Test password field
    html = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in html

    # Test with error
    html = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in html


# LLM-generated content at query #21
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)

    # Test autoescape is True
    assert env.autoescape is True


# LLM-generated content at query #22
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals["forms/input.html"] = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    env.globals["forms/select.html"] = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    env.globals["forms/checkbox.html"] = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    env.globals["forms/textarea.html"] = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"

    schema = Schema(fields={
        "text_field": String(),
        "choice_field": Choice(choices=["a", "b"]),
        "boolean_field": Boolean(),
        "textarea_field": String(format="text"),
        "password_field": String(format="password"),
    })

    form = Form(env=env, schema=schema)

    # Test String field
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test")
    assert "<input id='text-field' name='text_field' type='text' value='test'>" == result

    # Test Choice field
    result = form.render_field(field_name="choice_field", field=schema.fields["choice_field"])
    assert "<select id='choice-field' name='choice_field'></select>" == result

    # Test Boolean field
    result = form.render_field(field_name="boolean_field", field=schema.fields["boolean_field"], value=True)
    assert "<input id='boolean-field' name='boolean_field' type='checkbox' checked>" == result

    # Test Textarea field
    result = form.render_field(field_name="textarea_field", field=schema.fields["textarea_field"], value="test")
    assert "<textarea id='textarea-field' name='textarea_field'>test</textarea>" == result

    # Test Password field
    result = form.render_field(field_name="password_field", field=schema.fields["password_field"], value="secret")
    assert "<input id='password-field' name='password_field' type='password' value=''>" == result

    # Test with error
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], error="Error message")
    assert "Error message" in result


# LLM-generated content at query #23
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'text_field': String(format='text'),
        'email_field': String(format='email'),
        'number_field': String(format='number'),
        'password_field': String(format='password'),
        'textarea_field': String(format='text'),
        'select_field': Choice(choices=['a', 'b', 'c']),
        'checkbox_field': Boolean(),
    })
    form = Form(env=env, schema=schema, values={})

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test password field (value should be empty)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test select field
    result = form.render_field(field_name='select_field', field=schema.fields['select_field'], value='a')
    assert '<select id="select-field" name="select_field"></select>' in result

    # Test checkbox field (checked)
    result = form.render_field(field_name='checkbox_field', field=schema.fields['checkbox_field'], value=True)
    assert '<input id="checkbox-field" name="checkbox_field" type="checkbox" checked>' in result

    # Test checkbox field (unchecked)
    result = form.render_field(field_name='checkbox_field', field=schema.fields['checkbox_field'], value=False)
    assert '<input id="checkbox-field" name="checkbox_field" type="checkbox">' in result


# LLM-generated content at query #24
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test initialization without directory or package raises assertion error
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #25
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)

    # Test autoescape is enabled
    assert env.autoescape is True


# LLM-generated content at query #26
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    schema = Schema(fields={
        "name": String(),
        "age": String(format="number"),
        "email": String(format="email"),
        "password": String(format="password"),
        "bio": String(format="text"),
        "agree": Boolean(),
        "country": Choice(choices=["US", "UK", "CA"])
    })
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>",
        "forms/textarea.html": "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>",
        "forms/select.html": "<select id='{{ field_id }}' name='{{ field_name }}'></select>",
        "forms/checkbox.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    }))
    form = Form(env=env, schema=schema, values={"name": "John", "age": "30", "email": "john@example.com", "password": "secret", "bio": "Hello", "agree": True, "country": "US"})

    # Test
    result = form.render_fields()

    # Assert
    assert "<input id='name' name='name' type='text' value='John'>" in result
    assert "<input id='age' name='age' type='number' value='30'>" in result
    assert "<input id='email' name='email' type='email' value='john@example.com'>" in result
    assert "<input id='password' name='password' type='password' value=''>" in result
    assert "<textarea id='bio' name='bio'>Hello</textarea>" in result
    assert "<input id='agree' name='agree' type='checkbox' checked>" in result
    assert "<select id='country' name='country'></select>" in result


# LLM-generated content at query #27
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #28
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'choice_field': Choice(choices=['a', 'b']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text'),
        'password_field': String(format='password'),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'])
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='test')
    assert '<textarea id="textarea-field" name="textarea_field">test</textarea>' in result

    # Test password field (value should be empty string)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #29
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals["forms/input.html"] = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    env.globals["forms/select.html"] = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    env.globals["forms/checkbox.html"] = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    env.globals["forms/textarea.html"] = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"

    schema = Schema(fields={
        "text_field": String(),
        "choice_field": Choice(choices=["a", "b"]),
        "bool_field": Boolean(),
        "textarea_field": String(format="text"),
        "password_field": String(format="password"),
    })

    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test")
    assert "<input id='text-field' name='text_field' type='text' value='test'>" in result

    # Test choice field
    result = form.render_field(field_name="choice_field", field=schema.fields["choice_field"])
    assert "<select id='choice-field' name='choice_field'></select>" in result

    # Test boolean field
    result = form.render_field(field_name="bool_field", field=schema.fields["bool_field"], value=True)
    assert "<input id='bool-field' name='bool_field' type='checkbox' checked>" in result

    # Test textarea field
    result = form.render_field(field_name="textarea_field", field=schema.fields["textarea_field"], value="long text")
    assert "<textarea id='textarea-field' name='textarea_field'>long text</textarea>" in result

    # Test password field
    result = form.render_field(field_name="password_field", field=schema.fields["password_field"], value="secret")
    assert "<input id='password-field' name='password_field' type='password' value=''>" in result

    # Test with error
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], error="Invalid")
    assert "Invalid" in result


# LLM-generated content at query #30
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)

    # Test autoescape is True
    assert env.autoescape is True


# LLM-generated content at query #31
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'number_field': String(format='number'),
        'textarea_field': String(format='text'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='test')
    assert '<textarea id="textarea-field" name="textarea_field">test</textarea>' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result

    # Test password field (value should be empty string)
    password_field = String(format='password')
    result = form.render_field(field_name='password', field=password_field, value='secret')
    assert '<input id="password" name="password" type="password" value="">' in result


# LLM-generated content at query #32
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #33
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'textarea_field': String(format='text'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'boolean_field': Boolean(),
    })
    form = Form(env=env, schema=schema, values={
        'text_field': 'test',
        'email_field': 'test@example.com',
        'password_field': 'secret',
        'number_field': '123',
        'textarea_field': 'long text',
        'choice_field': 'a',
        'boolean_field': True,
    })

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (value should be empty)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field (checked)
    result = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=True)
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" checked>' in result

    # Test boolean field (not checked)
    result = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=False)
    assert '<input id="boolean-field" name="boolean_field" type="checkbox">' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid value')
    assert 'Invalid value' in result

    # Test with None value
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value=None)
    assert '<input id="text-field" name="text_field" type="text" value="None">' in result


# LLM-generated content at query #34
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)


# LLM-generated content at query #35
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'choice_field': Choice(choices=['a', 'b']),
        'bool_field': Boolean(),
        'textarea_field': String(format='text'),
    })
    form = Form(env=env, schema=schema)

    # Test String field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test Choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test Boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test Textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='test')
    assert '<textarea id="textarea-field" name="textarea_field">test</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Error message')
    assert 'Error message' in result

    # Test password field (value should be empty string)
    password_field = String(format='password')
    result = form.render_field(field_name='password', field=password_field, value='secret')
    assert 'value=""' in result


# LLM-generated content at query #36
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="templates", package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #37
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'name': String(title="Name"),
        'age': String(format="number", title="Age"),
        'bio': String(format="text", title="Bio"),
        'active': Boolean(title="Active"),
        'gender': Choice(choices=[('M', 'Male'), ('F', 'Female')], title="Gender"),
        'readonly_field': String(read_only=True, title="Read Only")
    })
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'bio': 'Developer', 'active': True, 'gender': 'M'})

    # Test without validation
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in html
    assert '<input id="age" name="age" type="number" value="30">' in html
    assert '<textarea id="bio" name="bio">Developer</textarea>' in html
    assert '<input id="active" name="active" type="checkbox" checked>' in html
    assert '<select id="gender" name="gender"></select>' in html
    assert 'readonly_field' not in html

    # Test with validation and errors
    form.validate({'name': '', 'age': 'invalid', 'bio': '', 'active': '', 'gender': 'X'})
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="">' in html
    assert '<input id="age" name="age" type="number" value="invalid">' in html
    assert '<textarea id="bio" name="bio"></textarea>' in html
    assert '<input id="active" name="active" type="checkbox">' in html
    assert '<select id="gender" name="gender"></select>' in html


# LLM-generated content at query #38
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "forms/input.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>",
        "forms/textarea.html": "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>",
        "forms/select.html": "<select id='{{ field_id }}' name='{{ field_name }}'></select>",
        "forms/checkbox.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    }), autoescape=True)

    schema = Schema(fields={
        "text_field": String(title="Text Field"),
        "password_field": String(format="password", title="Password Field"),
        "email_field": String(format="email", title="Email Field"),
        "number_field": String(format="number", title="Number Field"),
        "textarea_field": String(format="text", title="Text Area"),
        "choice_field": Choice(choices=[("1", "Option 1"), ("2", "Option 2")], title="Choice Field"),
        "boolean_field": Boolean(title="Boolean Field"),
    })

    form = Form(env=env, schema=schema, values={
        "text_field": "test",
        "password_field": "secret",
        "email_field": "test@example.com",
        "number_field": "42",
        "textarea_field": "long text",
        "choice_field": "1",
        "boolean_field": True,
    })

    # Test text field
    html = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test")
    assert "<input id='text-field' name='text_field' type='text' value='test'>" in html

    # Test password field
    html = form.render_field(field_name="password_field", field=schema.fields["password_field"], value="secret")
    assert "<input id='password-field' name='password_field' type='password' value=''>" in html

    # Test email field
    html = form.render_field(field_name="email_field", field=schema.fields["email_field"], value="test@example.com")
    assert "<input id='email-field' name='email_field' type='email' value='test@example.com'>" in html

    # Test number field
    html = form.render_field(field_name="number_field", field=schema.fields["number_field"], value="42")
    assert "<input id='number-field' name='number_field' type='number' value='42'>" in html

    # Test textarea field
    html = form.render_field(field_name="textarea_field", field=schema.fields["textarea_field"], value="long text")
    assert "<textarea id='textarea-field' name='textarea_field'>long text</textarea>" in html

    # Test choice field
    html = form.render_field(field_name="choice_field", field=schema.fields["choice_field"], value="1")
    assert "<select id='choice-field' name='choice_field'></select>" in html

    # Test boolean field
    html = form.render_field(field_name="boolean_field", field=schema.fields["boolean_field"], value=True)
    assert "<input id='boolean-field' name='boolean_field' type='checkbox' checked>" in html

    # Test with error
    html = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test", error="Invalid value")
    assert "Invalid value" in html

    # Test with None value
    html = form.render_field(field_name="text_field", field=schema.fields["text_field"], value=None)
    assert "<input id='text-field' name='text_field' type='text' value=''>" in html


# LLM-generated content at query #39
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment()
    env.loader = jinja2.DictLoader({
        "forms/input.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>",
        "forms/select.html": "<select id='{{ field_id }}' name='{{ field_name }}'></select>",
        "forms/checkbox.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>",
        "forms/textarea.html": "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    })

    schema = Schema(fields={
        "text_field": String(),
        "email_field": String(format="email"),
        "password_field": String(format="password"),
        "number_field": String(format="number"),
        "choice_field": Choice(choices=["a", "b", "c"]),
        "bool_field": Boolean(),
        "textarea_field": String(format="text")
    })

    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test")
    assert "<input id='text-field' name='text_field' type='text' value='test'>" in result

    # Test email field
    result = form.render_field(field_name="email_field", field=schema.fields["email_field"], value="test@example.com")
    assert "<input id='email-field' name='email_field' type='email' value='test@example.com'>" in result

    # Test password field (value should be empty)
    result = form.render_field(field_name="password_field", field=schema.fields["password_field"], value="secret")
    assert "<input id='password-field' name='password_field' type='password' value=''>" in result

    # Test number field
    result = form.render_field(field_name="number_field", field=schema.fields["number_field"], value="42")
    assert "<input id='number-field' name='number_field' type='number' value='42'>" in result

    # Test choice field
    result = form.render_field(field_name="choice_field", field=schema.fields["choice_field"], value="a")
    assert "<select id='choice-field' name='choice_field'></select>" in result

    # Test boolean field
    result = form.render_field(field_name="bool_field", field=schema.fields["bool_field"], value=True)
    assert "<input id='bool-field' name='bool_field' type='checkbox' checked>" in result

    # Test textarea field
    result = form.render_field(field_name="textarea_field", field=schema.fields["textarea_field"], value="long text")
    assert "<textarea id='textarea-field' name='textarea_field'>long text</textarea>" in result

    # Test with error
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test", error="Invalid")
    assert "Invalid" in result


# LLM-generated content at query #40
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'description': String(format='text'),
        'agree': Boolean(),
        'country': Choice(choices=['US', 'UK', 'CA'])
    })
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'description': 'Test', 'agree': True, 'country': 'US'})

    # Test without validation
    result = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in result
    assert '<input id="age" name="age" type="number" value="30">' in result
    assert '<textarea id="description" name="description">Test</textarea>' in result
    assert '<input id="agree" name="agree" type="checkbox" checked>' in result
    assert '<select id="country" name="country"><option value="US">US</option><option value="UK">UK</option><option value="CA">CA</option></select>' in result

    # Test with validation and errors
    form.validate({'name': '', 'age': 'invalid', 'description': '', 'agree': False, 'country': 'INVALID'})
    result = form.render_fields()
    assert '<input id="name" name="name" type="text" value="">' in result
    assert '<input id="age" name="age" type="number" value="invalid">' in result
    assert '<textarea id="description" name="description"></textarea>' in result
    assert '<input id="agree" name="agree" type="checkbox" >' in result
    assert '<select id="country" name="country"><option value="US">US</option><option value="UK">UK</option><option value="CA">CA</option></select>' in result


# LLM-generated content at query #41
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=True)
    schema = Schema(fields={
        'username': String(format='text'),
        'password': String(format='password'),
        'age': String(format='number'),
        'email': String(format='email'),
        'description': String(format='text'),
        'agree': Boolean(),
        'country': Choice(choices=['US', 'UK', 'CA'])
    })
    form = Form(env=env, schema=schema)

    # Test text input
    template = env.from_string("""
        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}"
               value="{{ value }}" {% if required %}required{% endif %}>
    """)
    env.get_template = lambda name: template

    result = form.render_field(
        field_name='username',
        field=schema.fields['username'],
        value='testuser',
        error=None
    )
    assert '<input type="text" name="username" id="username" value="testuser" required>' in result

    # Test password input
    result = form.render_field(
        field_name='password',
        field=schema.fields['password'],
        value='secret',
        error=None
    )
    assert '<input type="password" name="password" id="password" value="" required>' in result

    # Test number input
    result = form.render_field(
        field_name='age',
        field=schema.fields['age'],
        value='25',
        error=None
    )
    assert '<input type="number" name="age" id="age" value="25" required>' in result

    # Test email input
    result = form.render_field(
        field_name='email',
        field=schema.fields['email'],
        value='test@example.com',
        error=None
    )
    assert '<input type="email" name="email" id="email" value="test@example.com" required>' in result

    # Test textarea
    textarea_template = env.from_string("""
        <textarea name="{{ field_name }}" id="{{ field_id }}"
                  {% if required %}required{% endif %}>{{ value }}</textarea>
    """)
    env.get_template = lambda name: textarea_template if name == 'forms/textarea.html' else template

    result = form.render_field(
        field_name='description',
        field=schema.fields['description'],
        value='Test description',
        error=None
    )
    assert '<textarea name="description" id="description" required>Test description</textarea>' in result

    # Test checkbox
    checkbox_template = env.from_string("""
        <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"
               {% if value %}checked{% endif %} {% if required %}required{% endif %}>
    """)
    env.get_template = lambda name: checkbox_template if name == 'forms/checkbox.html' else template

    result = form.render_field(
        field_name='agree',
        field=schema.fields['agree'],
        value=True,
        error=None
    )
    assert '<input type="checkbox" name="agree" id="agree" checked required>' in result

    # Test select
    select_template = env.from_string("""
        <select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>
            {% for choice in field.choices %}
            <option value="{{ choice }}" {% if choice == value %}selected{% endif %}>{{ choice }}</option>
            {% endfor %}
        </select>
    """)
    env.get_template = lambda name: select_template if name == 'forms/select.html' else template

    result = form.render_field(
        field_name='country',
        field=schema.fields['country'],
        value='UK',
        error=None
    )
    assert '<select name="country" id="country" required>' in result
    assert '<option value="US">US</option>' in result
    assert '<option value="UK" selected>UK</option>' in result
    assert '<option value="CA">CA</option>' in result

    # Test with error
    template_with_error = env.from_string("""
        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}"
               value="{{ value }}" {% if required %}required{% endif %}>
        {% if error %}<span class="error">{{ error }}</span>{% endif %}
    """)
    env.get_template = lambda name: template_with_error

    result = form.render_field(
        field_name='username',
        field=schema.fields['username'],
        value='',
        error='This field is required'
    )
    assert '<input type="text" name="username" id="username" value="" required>' in result
    assert '<span class="error">This field is required</span>' in result

    # Test optional field
    optional_schema = Schema(fields={
        'optional_field': String(allow_blank=True)
    })
    optional_form = Form(env=env, schema=optional_schema)

    result = optional_form.render_field(
        field_name='optional_field',
        field=optional_schema.fields['optional_field'],
        value='test',
        error=None
    )
    assert 'required' not in result


# LLM-generated content at query #42
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'bio': String(format='text'),
        'active': Boolean(),
        'gender': Choice(choices=['M', 'F', 'O'])
    })
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'bio': 'Test bio', 'active': True, 'gender': 'M'})

    # Test with no errors
    form.validate({'name': 'John', 'age': '30', 'bio': 'Test bio', 'active': True, 'gender': 'M'})
    rendered = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in rendered
    assert '<input id="age" name="age" type="number" value="30">' in rendered
    assert '<textarea id="bio" name="bio">Test bio</textarea>' in rendered
    assert '<input id="active" name="active" type="checkbox" checked>' in rendered
    assert '<select id="gender" name="gender"></select>' in rendered

    # Test with errors
    form_with_errors = Form(env=env, schema=schema)
    form_with_errors.validate({'name': '', 'age': 'invalid', 'bio': '', 'active': 'not_bool', 'gender': 'X'})
    rendered_with_errors = form_with_errors.render_fields()
    assert 'value=""' in rendered_with_errors  # Empty value for name
    assert 'value="invalid"' in rendered_with_errors  # Invalid value for age
    assert '<textarea id="bio" name="bio"></textarea>' in rendered_with_errors  # Empty textarea
    assert '<input id="active" name="active" type="checkbox"' in rendered_with_errors  # Checkbox without checked
    assert '<select id="gender" name="gender"></select>' in rendered_with_errors  # Select with invalid choice

    # Test with read-only field
    schema_with_readonly = Schema(fields={'readonly_field': String(read_only=True)})
    form_readonly = Form(env=env, schema=schema_with_readonly)
    form_readonly.validate({})
    rendered_readonly = form_readonly.render_fields()
    assert rendered_readonly == ''  # Read-only fields should not be rendered


# LLM-generated content at query #43
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'text_field': String(),
        'number_field': String(format='number'),
        'textarea_field': String(format='text'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'boolean_field': Boolean()
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='test')
    assert '<textarea id="textarea-field" name="textarea_field">test</textarea>' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=True)
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" checked>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Error message')
    assert 'Error message' in result

    # Test password field (value should be empty string)
    password_field = String(format='password')
    result = form.render_field(field_name='password_field', field=password_field, value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result


# LLM-generated content at query #44
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'password_field': String(format='password'),
        'number_field': String(format='number'),
        'textarea_field': String(format='text'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'boolean_field': Boolean(),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(
        field_name='text_field',
        field=schema.fields['text_field'],
        value='test_value',
        error=None
    )
    assert '<input id="text-field" name="text_field" type="text" value="test_value">' in result

    # Test email field
    result = form.render_field(
        field_name='email_field',
        field=schema.fields['email_field'],
        value='test@example.com',
        error=None
    )
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test password field (value should be empty)
    result = form.render_field(
        field_name='password_field',
        field=schema.fields['password_field'],
        value='secret',
        error=None
    )
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test number field
    result = form.render_field(
        field_name='number_field',
        field=schema.fields['number_field'],
        value='123',
        error=None
    )
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test textarea field
    result = form.render_field(
        field_name='textarea_field',
        field=schema.fields['textarea_field'],
        value='multiline text',
        error=None
    )
    assert '<textarea id="textarea-field" name="textarea_field">multiline text</textarea>' in result

    # Test choice field
    result = form.render_field(
        field_name='choice_field',
        field=schema.fields['choice_field'],
        value='a',
        error=None
    )
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field (checked)
    result = form.render_field(
        field_name='boolean_field',
        field=schema.fields['boolean_field'],
        value=True,
        error=None
    )
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" checked>' in result

    # Test boolean field (unchecked)
    result = form.render_field(
        field_name='boolean_field',
        field=schema.fields['boolean_field'],
        value=False,
        error=None
    )
    assert '<input id="boolean-field" name="boolean_field" type="checkbox">' in result

    # Test with error
    result = form.render_field(
        field_name='text_field',
        field=schema.fields['text_field'],
        value='test_value',
        error='Invalid value'
    )
    assert 'Invalid value' in result


# LLM-generated content at query #45
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)

    # Test autoescape is True
    assert env.autoescape is True


# LLM-generated content at query #46
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'description': String(format='text'),
        'active': Boolean(),
        'gender': Choice(choices=['male', 'female', 'other'])
    })
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'description': 'Test', 'active': True, 'gender': 'male'})

    # Test without validation
    result = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in result
    assert '<input id="age" name="age" type="number" value="30">' in result
    assert '<textarea id="description" name="description">Test</textarea>' in result
    assert '<input id="active" name="active" type="checkbox" checked>' in result
    assert '<select id="gender" name="gender"></select>' in result

    # Test with validation and errors
    form.validate({'name': '', 'age': 'invalid', 'description': '', 'active': False, 'gender': 'invalid'})
    result = form.render_fields()
    assert '<input id="name" name="name" type="text" value="">' in result
    assert '<input id="age" name="age" type="number" value="invalid">' in result
    assert '<textarea id="description" name="description"></textarea>' in result
    assert '<input id="active" name="active" type="checkbox">' in result
    assert '<select id="gender" name="gender"></select>' in result

    # Test with read-only field
    schema_readonly = Schema(fields={'readonly_field': String(read_only=True)})
    form_readonly = Form(env=env, schema=schema_readonly, values={'readonly_field': 'value'})
    result = form_readonly.render_fields()
    assert 'readonly_field' not in result


# LLM-generated content at query #47
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)

    # Test autoescape is enabled
    assert env.autoescape is True


# LLM-generated content at query #48
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with neither directory nor package raises AssertionError
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #49
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals["forms/select.html"] = "<select name='{{ field_name }}'></select>"
    env.globals["forms/checkbox.html"] = "<input type='checkbox' name='{{ field_name }}' />"
    env.globals["forms/textarea.html"] = "<textarea name='{{ field_name }}'></textarea>"
    env.globals["forms/input.html"] = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}' />"

    schema = Schema(fields={
        "text_field": String(),
        "choice_field": Choice(choices=["a", "b"]),
        "bool_field": Boolean(),
        "text_area": String(format="text"),
        "password_field": String(format="password"),
    })

    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test")
    assert "<input type='text' name='text_field' value='test' />" in result

    # Test choice field
    result = form.render_field(field_name="choice_field", field=schema.fields["choice_field"], value="a")
    assert "<select name='choice_field'></select>" in result

    # Test boolean field
    result = form.render_field(field_name="bool_field", field=schema.fields["bool_field"], value=True)
    assert "<input type='checkbox' name='bool_field' />" in result

    # Test textarea field
    result = form.render_field(field_name="text_area", field=schema.fields["text_area"], value="long text")
    assert "<textarea name='text_area'></textarea>" in result

    # Test password field
    result = form.render_field(field_name="password_field", field=schema.fields["password_field"], value="secret")
    assert "<input type='password' name='password_field' value='' />" in result

    # Test with error
    result = form.render_field(field_name="text_field", field=schema.fields["text_field"], value="test", error="Invalid")
    assert "Invalid" in result


# LLM-generated content at query #50
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #51
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
    }), autoescape=True)

    schema = Schema(fields={
        'text_field': String(format='text'),
        'email_field': String(format='email'),
        'number_field': String(format='number'),
        'password_field': String(format='password'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'boolean_field': Boolean(),
    })

    form = Form(env=env, schema=schema, values={})

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test password field (value should be empty)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=True)
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" checked>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid value')
    assert 'Invalid value' in result


# LLM-generated content at query #52
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    }))

    schema = Schema(fields={
        'username': String(max_length=100),
        'email': String(format='email'),
        'age': String(format='number'),
        'bio': String(format='text'),
        'country': Choice(choices=['US', 'UK', 'CA']),
        'active': Boolean()
    })

    form = Form(env=env, schema=schema, values={'username': 'test', 'email': 'test@example.com', 'age': '25', 'bio': 'Test bio', 'country': 'US', 'active': True})

    # Test String field with default format
    result = form.render_field(field_name='username', field=schema.fields['username'], value='test')
    assert '<input id="username" name="username" type="text" value="test">' in result

    # Test String field with email format
    result = form.render_field(field_name='email', field=schema.fields['email'], value='test@example.com')
    assert '<input id="email" name="email" type="email" value="test@example.com">' in result

    # Test String field with number format
    result = form.render_field(field_name='age', field=schema.fields['age'], value='25')
    assert '<input id="age" name="age" type="number" value="25">' in result

    # Test String field with text format (should use textarea template)
    result = form.render_field(field_name='bio', field=schema.fields['bio'], value='Test bio')
    assert '<textarea id="bio" name="bio">Test bio</textarea>' in result

    # Test Choice field (should use select template)
    result = form.render_field(field_name='country', field=schema.fields['country'], value='US')
    assert '<select id="country" name="country"></select>' in result

    # Test Boolean field (should use checkbox template)
    result = form.render_field(field_name='active', field=schema.fields['active'], value=True)
    assert '<input id="active" name="active" type="checkbox" checked>' in result

    # Test with error
    result = form.render_field(field_name='username', field=schema.fields['username'], value='test', error='This field is required')
    assert 'This field is required' in result

    # Test password field (value should be empty string)
    password_field = String(format='password')
    result = form.render_field(field_name='password', field=password_field, value='secret')
    assert '<input id="password" name="password" type="password" value="">' in result


# LLM-generated content at query #53
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals.update({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    })

    schema = Schema(fields={
        'text_field': String(),
        'choice_field': Choice(choices=['a', 'b']),
        'bool_field': Boolean(),
        'text_area_field': String(format='text')
    })

    form = Form(env=env, schema=schema, values={'text_field': 'test'})

    # Test String field
    result = form.render_field(
        field_name='text_field',
        field=schema.fields['text_field'],
        value='test',
        error=None
    )
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test Choice field
    result = form.render_field(
        field_name='choice_field',
        field=schema.fields['choice_field'],
        value='a',
        error=None
    )
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test Boolean field
    result = form.render_field(
        field_name='bool_field',
        field=schema.fields['bool_field'],
        value=True,
        error=None
    )
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test TextArea field
    result = form.render_field(
        field_name='text_area_field',
        field=schema.fields['text_area_field'],
        value='long text',
        error=None
    )
    assert '<textarea id="text-area-field" name="text_area_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(
        field_name='text_field',
        field=schema.fields['text_field'],
        value='test',
        error='Invalid value'
    )
    assert 'Invalid value' in result

    # Test password field
    password_field = String(format='password')
    result = form.render_field(
        field_name='password',
        field=password_field,
        value='secret',
        error=None
    )
    assert '<input id="password" name="password" type="password" value="">' in result


# LLM-generated content at query #54
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    schema = Schema(fields={
        'name': String(),
        'age': String(format='number'),
        'email': String(format='email'),
        'bio': String(format='text'),
        'agree': Boolean(),
        'country': Choice(choices=['US', 'UK', 'CA'])
    })
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    }))
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': '30', 'email': 'john@example.com', 'bio': 'Hello', 'agree': True, 'country': 'US'})

    # Test without validation
    html = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in html
    assert '<input id="age" name="age" type="number" value="30">' in html
    assert '<input id="email" name="email" type="email" value="john@example.com">' in html
    assert '<textarea id="bio" name="bio">Hello</textarea>' in html
    assert '<input id="agree" name="agree" type="checkbox" checked>' in html
    assert '<select id="country" name="country"><option value="US">US</option><option value="UK">UK</option><option value="CA">CA</option></select>' in html

    # Test with validation errors
    form.validate({'name': '', 'age': 'invalid', 'email': 'invalid', 'bio': '', 'agree': False, 'country': 'INVALID'})
    html = form.render_fields()
    assert 'value=""' in html  # Empty name
    assert 'value="invalid"' in html  # Invalid age
    assert 'value="invalid"' in html  # Invalid email
    assert '<textarea id="bio" name="bio"></textarea>' in html  # Empty bio
    assert '<input id="agree" name="agree" type="checkbox"' in html  # Unchecked agree
    assert '<select id="country" name="country">' in html  # Invalid country


# LLM-generated content at query #55
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'email_field': String(format='email'),
        'number_field': String(format='number'),
        'password_field': String(format='password'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'boolean_field': Boolean(),
        'textarea_field': String(format='text'),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test password field (value should be empty)
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=True)
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid')
    assert 'Invalid' in result


# LLM-generated content at query #56
#--------------------------

```python
def test_Form_render_fields():
    # Setup
    schema = Schema(fields={
        'name': String(max_length=100),
        'age': Integer(min_value=0),
        'email': String(format='email'),
        'bio': String(format='text'),
        'gender': Choice(choices=['M', 'F', 'O']),
        'subscribe': Boolean()
    })

    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))

    form = Form(env=env, schema=schema, values={'name': 'John', 'age': 30, 'email': 'john@example.com', 'bio': 'Test bio', 'gender': 'M', 'subscribe': True})

    # Test without validation
    result = form.render_fields()
    assert '<input id="name" name="name" type="text" value="John">' in result
    assert '<input id="age" name="age" type="number" value="30">' in result
    assert '<input id="email" name="email" type="email" value="john@example.com">' in result
    assert '<textarea id="bio" name="bio">Test bio</textarea>' in result
    assert '<select id="gender" name="gender"><option value="M">M</option><option value="F">F</option><option value="O">O</option></select>' in result
    assert '<input id="subscribe" name="subscribe" type="checkbox" checked>' in result

    # Test with validation and errors
    form.validate({'name': '', 'age': -1, 'email': 'invalid', 'bio': '', 'gender': 'X', 'subscribe': 'not_bool'})
    result = form.render_fields()
    assert '<input id="name" name="name" type="text" value="">' in result
    assert '<input id="age" name="age" type="number" value="-1">' in result
    assert '<input id="email" name="email" type="email" value="invalid">' in result
    assert '<textarea id="bio" name="bio"></textarea>' in result
    assert '<select id="gender" name="gender"><option value="M">M</option><option value="F">F</option><option value="O">O</option></select>' in result
    assert '<input id="subscribe" name="subscribe" type="checkbox" >' in result

    # Test with read-only field
    schema_readonly = Schema(fields={
        'readonly_field': String(read_only=True),
        'normal_field': String()
    })
    form_readonly = Form(env=env, schema=schema_readonly, values={'readonly_field': 'readonly', 'normal_field': 'normal'})
    result = form_readonly.render_fields()
    assert 'readonly_field' not in result
    assert '<input id="normal-field" name="normal_field" type="text" value="normal">' in result


# LLM-generated content at query #57
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test initialization with package
    forms = Jinja2Forms(package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="path/to/templates", package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test initialization with neither directory nor package raises AssertionError
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #58
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory only
    forms = Jinja2Forms(directory="test_templates")
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package only
    forms = Jinja2Forms(package="test_package")
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)
    assert len(forms.env.loader.loaders) == 2
    assert isinstance(forms.env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(forms.env.loader.loaders[1], jinja2.PackageLoader)

    # Test autoescape is enabled
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env.autoescape is True


# LLM-generated content at query #59
#--------------------------

```python
def test_Form_render_field():
    # Setup
    schema = Schema(fields={
        'name': String(max_length=100),
        'age': Integer(minimum=0),
        'bio': String(format='text'),
        'active': Boolean(),
        'gender': Choice(choices=['M', 'F', 'O'])
    })
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>',
        'forms/checkbox.html': '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    }))
    form = Form(env=env, schema=schema, values={'name': 'John', 'age': 30, 'bio': 'Developer', 'active': True, 'gender': 'M'})

    # Test String field
    result = form.render_field(field_name='name', field=schema.fields['name'], value='John')
    assert '<input type="text" name="name" value="John">' in result

    # Test Integer field
    result = form.render_field(field_name='age', field=schema.fields['age'], value=30)
    assert '<input type="text" name="age" value="30">' in result

    # Test Textarea field
    result = form.render_field(field_name='bio', field=schema.fields['bio'], value='Developer')
    assert '<textarea name="bio">Developer</textarea>' in result

    # Test Boolean field
    result = form.render_field(field_name='active', field=schema.fields['active'], value=True)
    assert '<input type="checkbox" name="active" checked>' in result

    # Test Choice field
    result = form.render_field(field_name='gender', field=schema.fields['gender'], value='M')
    assert '<select name="gender">' in result
    assert '<option value="M">M</option>' in result

    # Test with error
    result = form.render_field(field_name='name', field=schema.fields['name'], value='John', error='Invalid name')
    assert 'Invalid name' in result

    # Test with None value
    result = form.render_field(field_name='name', field=schema.fields['name'], value=None)
    assert '<input type="text" name="name" value="">' in result

    # Test with password format
    password_field = String(format='password')
    result = form.render_field(field_name='password', field=password_field, value='secret')
    assert '<input type="password" name="password" value="">' in result


# LLM-generated content at query #60
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #61
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #62
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #63
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization without directory or package raises assertion error
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #64
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    }))
    schema = Schema(fields={
        'text_field': String(),
        'number_field': String(format='number'),
        'password_field': String(format='password'),
        'textarea_field': String(format='text'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'boolean_field': Boolean()
    })
    form = Form(env=env, schema=schema)

    # Test text field
    assert '<input id="text-field" name="text_field" type="text" value="test">' in form.render_field(
        field_name='text_field',
        field=schema.fields['text_field'],
        value='test'
    )

    # Test number field
    assert '<input id="number-field" name="number_field" type="number" value="123">' in form.render_field(
        field_name='number_field',
        field=schema.fields['number_field'],
        value='123'
    )

    # Test password field (value should be empty)
    assert '<input id="password-field" name="password_field" type="password" value="">' in form.render_field(
        field_name='password_field',
        field=schema.fields['password_field'],
        value='secret'
    )

    # Test textarea field
    assert '<textarea id="textarea-field" name="textarea_field">multiline</textarea>' in form.render_field(
        field_name='textarea_field',
        field=schema.fields['textarea_field'],
        value='multiline'
    )

    # Test choice field
    assert '<select id="choice-field" name="choice_field"></select>' in form.render_field(
        field_name='choice_field',
        field=schema.fields['choice_field']
    )

    # Test boolean field
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" checked>' in form.render_field(
        field_name='boolean_field',
        field=schema.fields['boolean_field'],
        value=True
    )
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" >' in form.render_field(
        field_name='boolean_field',
        field=schema.fields['boolean_field'],
        value=False
    )


# LLM-generated content at query #65
#--------------------------

```python
def test_Jinja2Forms():
    # Test initialization with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test initialization with neither directory nor package raises AssertionError
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #66
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env.loader, jinja2.ChoiceLoader)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()

    # Test with jinja2 not installed
    with pytest.raises(AssertionError):
        import sys
        sys.modules['jinja2'] = None
        Jinja2Forms(directory="path/to/templates")


# LLM-generated content at query #67
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env, jinja2.Environment)
    assert isinstance(env.loader, jinja2.ChoiceLoader)

    # Test autoescape is True
    assert env.autoescape is True


# LLM-generated content at query #68
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.globals["forms"] = {
        "input.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>",
        "select.html": "<select id='{{ field_id }}' name='{{ field_name }}'></select>",
        "checkbox.html": "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>",
        "textarea.html": "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>",
    }

    schema = Schema(fields={
        "text_field": String(format="text"),
        "email_field": String(format="email"),
        "choice_field": Choice(choices=["a", "b"]),
        "bool_field": Boolean(),
    })

    form = Form(env=env, schema=schema, values={})

    # Test text field
    result = form.render_field(
        field_name="text_field",
        field=schema.fields["text_field"],
        value="test_value",
        error=None
    )
    assert "<input id='text-field' name='text_field' type='text' value='test_value'>" in result

    # Test email field
    result = form.render_field(
        field_name="email_field",
        field=schema.fields["email_field"],
        value="test@example.com",
        error=None
    )
    assert "<input id='email-field' name='email_field' type='email' value='test@example.com'>" in result

    # Test choice field
    result = form.render_field(
        field_name="choice_field",
        field=schema.fields["choice_field"],
        value="a",
        error=None
    )
    assert "<select id='choice-field' name='choice_field'></select>" in result

    # Test boolean field
    result = form.render_field(
        field_name="bool_field",
        field=schema.fields["bool_field"],
        value=True,
        error=None
    )
    assert "<input id='bool-field' name='bool_field' type='checkbox' checked>" in result

    # Test password field (should have empty value)
    password_field = String(format="password")
    result = form.render_field(
        field_name="password",
        field=password_field,
        value="secret",
        error=None
    )
    assert "<input id='password' name='password' type='password' value=''>" in result

    # Test with error
    result = form.render_field(
        field_name="text_field",
        field=schema.fields["text_field"],
        value="",
        error="This field is required"
    )
    assert "This field is required" in result


# LLM-generated content at query #69
#--------------------------

```python
def test_Jinja2Forms_load_template_env():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)

    # Test autoescape is enabled
    assert env.autoescape is True


# LLM-generated content at query #70
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="path/to/templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #71
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #72
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'password_field': String(format='password'),
        'email_field': String(format='email'),
        'number_field': String(format='number'),
        'textarea_field': String(format='text'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'boolean_field': Boolean(),
    })
    form = Form(env=env, schema=schema, values={})

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test_value')
    assert '<input id="text-field" name="text_field" type="text" value="test_value">' in result

    # Test password field
    result = form.render_field(field_name='password_field', field=schema.fields['password_field'], value='secret')
    assert '<input id="password-field" name="password_field" type="password" value="">' in result

    # Test email field
    result = form.render_field(field_name='email_field', field=schema.fields['email_field'], value='test@example.com')
    assert '<input id="email-field" name="email_field" type="email" value="test@example.com">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test textarea field
    result = form.render_field(field_name='textarea_field', field=schema.fields['textarea_field'], value='long text')
    assert '<textarea id="textarea-field" name="textarea_field">long text</textarea>' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='boolean_field', field=schema.fields['boolean_field'], value=True)
    assert '<input id="boolean-field" name="boolean_field" type="checkbox" checked>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Invalid value')
    assert 'Invalid value' in result


# LLM-generated content at query #73
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="templates", package="my_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #74
#--------------------------

```python
def test_Jinja2Forms():
    # Test with directory
    forms = Jinja2Forms(directory="test_templates")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with package
    forms = Jinja2Forms(package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with both directory and package
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    assert forms.env is not None
    assert isinstance(forms.env, jinja2.Environment)

    # Test with neither directory nor package (should raise AssertionError)
    with pytest.raises(AssertionError):
        Jinja2Forms()


# LLM-generated content at query #75
#--------------------------

```python
def test_Form_render_field():
    # Setup
    env = jinja2.Environment(loader=jinja2.DictLoader({
        'forms/input.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">',
        'forms/select.html': '<select id="{{ field_id }}" name="{{ field_name }}"></select>',
        'forms/checkbox.html': '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>',
        'forms/textarea.html': '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>',
    }))
    schema = Schema(fields={
        'text_field': String(),
        'number_field': String(format='number'),
        'choice_field': Choice(choices=['a', 'b', 'c']),
        'bool_field': Boolean(),
        'text_area': String(format='text'),
    })
    form = Form(env=env, schema=schema)

    # Test text field
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test')
    assert '<input id="text-field" name="text_field" type="text" value="test">' in result

    # Test number field
    result = form.render_field(field_name='number_field', field=schema.fields['number_field'], value='123')
    assert '<input id="number-field" name="number_field" type="number" value="123">' in result

    # Test choice field
    result = form.render_field(field_name='choice_field', field=schema.fields['choice_field'], value='a')
    assert '<select id="choice-field" name="choice_field"></select>' in result

    # Test boolean field
    result = form.render_field(field_name='bool_field', field=schema.fields['bool_field'], value=True)
    assert '<input id="bool-field" name="bool_field" type="checkbox" checked>' in result

    # Test textarea field
    result = form.render_field(field_name='text_area', field=schema.fields['text_area'], value='long text')
    assert '<textarea id="text-area" name="text_area">long text</textarea>' in result

    # Test with error
    result = form.render_field(field_name='text_field', field=schema.fields['text_field'], value='test', error='Error message')
    assert 'Error message' in result

    # Test password field (value should be empty string)
    password_field = String(format='password')
    result = form.render_field(field_name='password', field=password_field, value='secret')
    assert '<input id="password" name="password" type="password" value="">' in result


