####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_valid_data():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema, values={"name": "test"})
    form.validate({"name": "valid"})
    assert form.is_valid
    assert form.validated_data == {"name": "valid"}
    assert form._validate_called

def test_validate_with_invalid_data():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String(min_length=5)})
    form = Form(env=env, schema=schema, values={"name": "test"})
    form.validate({"name": "bad"})
    assert not form.is_valid
    assert form.errors is not None
    assert form._validate_called

def test_validate_called_twice_raises_assertion():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema, values={"name": "test"})
    form.validate({"name": "valid"})
    try:
        form.validate({"name": "valid"})
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "validate() has already been called."

def test_validate_without_data():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema, values={"name": "test"})
    form.validate()
    assert form.is_valid
    assert form.validated_data == {"name": "test"}
    assert form._validate_called


# LLM-generated content at query #2
#--------------------------

```python
def test_form_constructor_with_valid_schema():
    env = jinja2.Environment()
    schema = Schema(fields={})
    form = Form(env=env, schema=schema)
    assert form.env == env
    assert form.schema == schema
    assert form.values is None
    assert form.errors is None
    assert not form._validate_called

def test_form_constructor_with_values():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    values = {"name": "John"}
    form = Form(env=env, schema=schema, values=values)
    assert form.values == values


# LLM-generated content at query #3
#--------------------------

```python
def test_template_for_field_with_choice():
    form = Form(env=None, schema=None)
    field = Choice()
    assert form.template_for_field(field) == "forms/select.html"

def test_template_for_field_with_boolean():
    form = Form(env=None, schema=None)
    field = Boolean()
    assert form.template_for_field(field) == "forms/checkbox.html"

def test_template_for_field_with_string_text_format():
    form = Form(env=None, schema=None)
    field = String(format="text")
    assert form.template_for_field(field) == "forms/textarea.html"

def test_template_for_field_with_string_non_text_format():
    form = Form(env=None, schema=None)
    field = String(format="email")
    assert form.template_for_field(field) == "forms/input.html"

def test_template_for_field_with_non_specialized_field():
    form = Form(env=None, schema=None)
    field = Integer()
    assert form.template_for_field(field) == "forms/input.html"

def test_template_for_field_with_object_raises_assertion():
    form = Form(env=None, schema=None)
    field = Object()
    with pytest.raises(AssertionError):
        form.template_for_field(field)


# LLM-generated content at query #4
#--------------------------

```python
def test_template_for_field_with_choice():
    field = Choice()
    form = Form(env=None, schema=None)
    assert form.template_for_field(field) == "forms/select.html"


# LLM-generated content at query #5
#--------------------------

```python
def test_constructor_with_directory():
    instance = Jinja2Forms(directory="test_dir")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.FileSystemLoader)

def test_constructor_with_package():
    instance = Jinja2Forms(package="test_package")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.PackageLoader)

def test_constructor_with_both_directory_and_package():
    instance = Jinja2Forms(directory="test_dir", package="test_package")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.ChoiceLoader)
    assert len(instance.env.loader.loaders) == 2
    assert isinstance(instance.env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(instance.env.loader.loaders[1], jinja2.PackageLoader)

def test_constructor_without_directory_and_package():
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "Either 'directory' or 'package' must be specified."


# LLM-generated content at query #6
#--------------------------

```python
def test_render_field_with_string_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "{{ field_id }}"}))
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema, values={"name": "test"})
    form.validate({"name": "test"})
    assert form.render_field(field_name="name", field=schema.fields["name"], value="test", error=None) == "name"

def test_render_field_with_choice_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/select.html": "{{ field_id }}"}))
    schema = Schema(fields={"choice": Choice(choices=[("a", "A"), ("b", "B")])})
    form = Form(env=env, schema=schema, values={"choice": "a"})
    form.validate({"choice": "a"})
    assert form.render_field(field_name="choice", field=schema.fields["choice"], value="a", error=None) == "choice"

def test_render_field_with_boolean_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/checkbox.html": "{{ field_id }}"}))
    schema = Schema(fields={"flag": Boolean()})
    form = Form(env=env, schema=schema, values={"flag": True})
    form.validate({"flag": True})
    assert form.render_field(field_name="flag", field=schema.fields["flag"], value=True, error=None) == "flag"

def test_render_field_with_textarea():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/textarea.html": "{{ field_id }}"}))
    schema = Schema(fields={"description": String(format="text")})
    form = Form(env=env, schema=schema, values={"description": "test"})
    form.validate({"description": "test"})
    assert form.render_field(field_name="description", field=schema.fields["description"], value="test", error=None) == "description"

def test_render_field_with_error():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "{{ error }}"}))
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema, values={"name": "test"})
    form.validate({"name": ""})
    assert form.render_field(field_name="name", field=schema.fields["name"], value="", error="This field is required.") == "This field is required."

def test_render_field_with_password_input():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "{{ value }}"}))
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=env, schema=schema, values={"password": "secret"})
    form.validate({"password": "secret"})
    assert form.render_field(field_name="password", field=schema.fields["password"], value="secret", error=None) == ""

def test_render_field_with_custom_format():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "{{ input_type }}"}))
    schema = Schema(fields={"color": String(format="color")})
    form = Form(env=env, schema=schema, values={"color": "#ff0000"})
    form.validate({"color": "#ff0000"})
    assert form.render_field(field_name="color", field=schema.fields["color"], value="#ff0000", error=None) == "color"

def test_render_field_with_unknown_format():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "{{ input_type }}"}))
    schema = Schema(fields={"unknown": String(format="unknown")})
    form = Form(env=env, schema=schema, values={"unknown": "test"})
    form.validate({"unknown": "test"})
    assert form.render_field(field_name="unknown", field=schema.fields["unknown"], value="test", error=None) == "text"


# LLM-generated content at query #7
#--------------------------

```python
def test_render_field_uses_title_as_label():
    env = jinja2.Environment()
    schema = Schema(fields={"test_field": String(title="Test Label")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="test_field", field=schema.fields["test_field"])
    assert "Test Label" in result


# LLM-generated content at query #8
#--------------------------

```python
def test_allow_empty_with_allow_null():
    field = Field(allow_null=True)
    assert field.allow_null or getattr(field, "allow_blank", False) == True

def test_allow_empty_with_allow_blank():
    field = Field()
    field.allow_blank = True
    assert field.allow_null or getattr(field, "allow_blank", False) == True

def test_allow_empty_with_neither():
    field = Field()
    assert field.allow_null or getattr(field, "allow_blank", False) == False


# LLM-generated content at query #9
#--------------------------

```python
def test_load_template_env_with_directory():
    jinja2_forms = Jinja2Forms(directory="/path/to/templates")
    env = jinja2_forms.load_template_env(directory="/path/to/templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["/path/to/templates"]
    assert env.autoescape is True

def test_load_template_env_with_package():
    jinja2_forms = Jinja2Forms(package="package.name")
    env = jinja2_forms.load_template_env(package="package.name")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "package.name"
    assert env.loader.package_path == "templates"
    assert env.autoescape is True

def test_load_template_env_with_directory_and_package():
    jinja2_forms = Jinja2Forms(directory="/path/to/templates", package="package.name")
    env = jinja2_forms.load_template_env(directory="/path/to/templates", package="package.name")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)
    assert env.autoescape is True


# LLM-generated content at query #10
#--------------------------

```python
def test_load_template_env_with_both_none():
    jinja2_forms = Jinja2Forms()
    with pytest.raises(AssertionError):
        jinja2_forms.load_template_env(directory=None, package=None)


# LLM-generated content at query #11
#--------------------------

```python
def test_template_for_field_with_object_field():
    env = jinja2.Environment()
    schema = Schema(fields={"test": Object()})
    form = Form(env=env, schema=schema)
    with pytest.raises(AssertionError, match="Forms do not support rendering Object fields"):
        form.template_for_field(Object())


# LLM-generated content at query #12
#--------------------------

```python
def test_load_template_env_with_directory():
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

def test_load_template_env_with_package():
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

def test_load_template_env_with_directory_and_package():
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)


# LLM-generated content at query #13
#--------------------------

```python
def test_load_template_env_with_directory():
    forms = Jinja2Forms(directory="test_templates")
    env = forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]

def test_load_template_env_with_package():
    forms = Jinja2Forms(package="test_package")
    env = forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"

def test_load_template_env_with_both_directory_and_package():
    forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)
    assert env.loader.loaders[0].searchpath == ["test_templates"]
    assert env.loader.loaders[1].package_name == "test_package"
    assert env.loader.loaders[1].package_path == "templates"


# LLM-generated content at query #14
#--------------------------

```python
def test_required_predicate_false():
    field = Field(allow_null=True)
    assert not (not field.has_default() and not (field.allow_null or getattr(field, "allow_blank", False)))


# LLM-generated content at query #15
#--------------------------

```python
def test_render_fields_with_no_errors_and_no_values():
    env = jinja2.Environment()
    schema = Schema(fields={})
    form = Form(env=env, schema=schema)
    assert form.render_fields() == ""

def test_render_fields_with_errors():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema)
    form.validate({"name": "invalid"})
    assert form.render_fields() == form.render_field(field_name="name", field=schema.fields["name"], value="invalid", error=form.errors["name"])

def test_render_fields_with_values():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema, values={"name": "valid"})
    assert form.render_fields() == form.render_field(field_name="name", field=schema.fields["name"], value="valid", error=None)

def test_render_fields_with_read_only_field():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String(read_only=True)})
    form = Form(env=env, schema=schema)
    assert form.render_fields() == ""

def test_render_fields_with_multiple_fields():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String(), "age": Integer()})
    form = Form(env=env, schema=schema)
    assert form.render_fields() == form.render_field(field_name="name", field=schema.fields["name"], value=None, error=None) + form.render_field(field_name="age", field=schema.fields["age"], value=None, error=None)


# LLM-generated content at query #16
#--------------------------

```python
def test_constructor_with_directory():
    jinja2_forms = Jinja2Forms(directory="path/to/templates")
    assert isinstance(jinja2_forms.env, jinja2.Environment)
    assert isinstance(jinja2_forms.env.loader, jinja2.FileSystemLoader)

def test_constructor_with_package():
    jinja2_forms = Jinja2Forms(package="package.name")
    assert isinstance(jinja2_forms.env, jinja2.Environment)
    assert isinstance(jinja2_forms.env.loader, jinja2.PackageLoader)

def test_constructor_with_both_directory_and_package():
    jinja2_forms = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert isinstance(jinja2_forms.env, jinja2.Environment)
    assert isinstance(jinja2_forms.env.loader, jinja2.ChoiceLoader)

def test_constructor_without_directory_and_package():
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_render_field_with_valid_field():
    env = jinja2.Environment()
    schema = Schema(fields={"test_field": String()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="test_field", field=String(), value="test_value")
    assert result is not None

def test_render_field_with_null_value():
    env = jinja2.Environment()
    schema = Schema(fields={"test_field": String(allow_null=True)})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="test_field", field=String(allow_null=True), value=None)
    assert result is not None

def test_render_field_with_error():
    env = jinja2.Environment()
    schema = Schema(fields={"test_field": String()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="test_field", field=String(), value="test_value", error="Error message")
    assert result is not None

def test_render_field_with_password_input_type():
    env = jinja2.Environment()
    schema = Schema(fields={"password_field": String(format="password")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="password_field", field=String(format="password"), value="secret")
    assert result is not None

def test_render_field_with_custom_template():
    env = jinja2.Environment()
    schema = Schema(fields={"choice_field": Choice(choices=[("1", "Option 1")])})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="choice_field", field=Choice(choices=[("1", "Option 1")]))
    assert result is not None

def test_render_field_with_boolean_field():
    env = jinja2.Environment()
    schema = Schema(fields={"bool_field": Boolean()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="bool_field", field=Boolean(), value=True)
    assert result is not None

def test_render_field_with_textarea():
    env = jinja2.Environment()
    schema = Schema(fields={"text_field": String(format="text")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="text_field", field=String(format="text"), value="Long text")
    assert result is not None

def test_render_field_with_read_only_field():
    env = jinja2.Environment()
    schema = Schema(fields={"read_only_field": String(read_only=True)})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="read_only_field", field=String(read_only=True), value="read_only")
    assert result is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_jinja2forms_constructor_with_directory():
    form = Jinja2Forms(directory="path/to/templates")
    assert form.env.loader is not None

def test_jinja2forms_constructor_with_package():
    form = Jinja2Forms(package="package.name")
    assert form.env.loader is not None

def test_jinja2forms_constructor_with_both():
    form = Jinja2Forms(directory="path/to/templates", package="package.name")
    assert form.env.loader is not None

def test_jinja2forms_constructor_without_args():
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_jinja2_not_installed():
    global jinja2
    original_jinja2 = jinja2
    jinja2 = None
    try:
        Jinja2Forms(directory="test_dir")
    except AssertionError as e:
        assert str(e) == "jinja2 must be installed to use Jinja2Forms."
    finally:
        jinja2 = original_jinja2


# LLM-generated content at query #20
#--------------------------

```python
def test_jinja2_not_installed():
    global jinja2
    jinja2 = None
    with pytest.raises(AssertionError, match="jinja2 must be installed to use Jinja2Forms."):
        Jinja2Forms(directory="test_dir")


# LLM-generated content at query #21
#--------------------------

```python
def test_load_template_env_both_none():
    with pytest.raises(AssertionError):
        Jinja2Forms().load_template_env(directory=None, package=None)


# LLM-generated content at query #22
#--------------------------

```python
def test_jinja2_not_installed():
    global jinja2
    jinja2 = None
    with pytest.raises(AssertionError, match="jinja2 must be installed to use Jinja2Forms."):
        Jinja2Forms(directory="test")


# LLM-generated content at query #23
#--------------------------

```python
def test_load_template_env_both_none():
    with pytest.raises(AssertionError):
        Jinja2Forms().load_template_env(directory=None, package=None)


# LLM-generated content at query #24
#--------------------------

```python
def test_jinja2_not_installed():
    global jinja2
    original_jinja2 = jinja2
    jinja2 = None
    try:
        Jinja2Forms(directory="some_dir")
    except AssertionError as e:
        assert str(e) == "jinja2 must be installed to use Jinja2Forms."
    finally:
        jinja2 = original_jinja2


# LLM-generated content at query #25
#--------------------------

```python
def test_load_template_env_with_directory():
    jinja2_forms = Jinja2Forms(directory="test_templates")
    env = jinja2_forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]
    assert env.autoescape is True

def test_load_template_env_with_package():
    jinja2_forms = Jinja2Forms(package="test_package")
    env = jinja2_forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"
    assert env.autoescape is True

def test_load_template_env_with_directory_and_package():
    jinja2_forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = jinja2_forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)
    assert env.autoescape is True


# LLM-generated content at query #26
#--------------------------

```python
def test_load_template_env_with_both_directory_and_package():
    jinja2_forms = Jinja2Forms(directory="test_dir", package="test_pkg")
    env = jinja2_forms.load_template_env(directory="test_dir", package="test_pkg")
    assert isinstance(env.loader, jinja2.ChoiceLoader)


# LLM-generated content at query #27
#--------------------------

```python
def test_jinja2_not_installed():
    global jinja2
    jinja2 = None
    with pytest.raises(AssertionError, match="jinja2 must be installed to use Jinja2Forms."):
        Jinja2Forms(directory="some_dir")


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_form_html_method():
    env = jinja2.Environment()
    schema = Schema(fields={"test": String()})
    form = Form(env=env, schema=schema, values={"test": "value"})
    result = form.__html__()
    assert isinstance(result, markupsafe.Markup)
    assert str(result) == form.render_fields()


# LLM-generated content at query #2
#--------------------------

```python
def test_render_fields_with_no_errors_and_no_values():
    env = jinja2.Environment()
    schema = Schema(fields={})
    form = Form(env=env, schema=schema)
    assert form.render_fields() == ""

def test_render_fields_with_errors():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema)
    form.validate({"name": "invalid"})
    assert form.render_fields() == form.render_field(field_name="name", field=schema.fields["name"], value="invalid", error="invalid")

def test_render_fields_with_values():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema, values={"name": "valid"})
    assert form.render_fields() == form.render_field(field_name="name", field=schema.fields["name"], value="valid", error=None)

def test_render_fields_with_read_only_field():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String(read_only=True)})
    form = Form(env=env, schema=schema)
    assert form.render_fields() == ""

def test_render_fields_with_multiple_fields():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String(), "age": Integer()})
    form = Form(env=env, schema=schema)
    form.validate({"name": "John", "age": 30})
    expected = form.render_field(field_name="name", field=schema.fields["name"], value="John", error=None)
    expected += form.render_field(field_name="age", field=schema.fields["age"], value=30, error=None)
    assert form.render_fields() == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_valid_data():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema)
    form.validate({"name": "test"})
    assert form.is_valid
    assert form.validated_data == {"name": "test"}
    assert form._validate_called

def test_validate_with_invalid_data():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String(min_length=5)})
    form = Form(env=env, schema=schema)
    form.validate({"name": "test"})
    assert not form.is_valid
    assert form.errors is not None
    assert form._validate_called

def test_validate_called_twice_raises_assertion():
    env = jinja2.Environment()
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema)
    form.validate({"name": "test"})
    with pytest.raises(AssertionError):
        form.validate({"name": "test"})


# LLM-generated content at query #4
#--------------------------

```python
def test_render_field_with_string_field():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="name", field=schema.fields["name"], value="John Doe")
    assert "<input" in result
    assert 'name="name"' in result
    assert 'type="text"' in result
    assert 'value="John Doe"' in result

def test_render_field_with_password_field():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="password", field=schema.fields["password"], value="secret")
    assert 'type="password"' in result
    assert 'value=""' in result

def test_render_field_with_email_field():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"email": String(format="email")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="email", field=schema.fields["email"], value="user@example.com")
    assert 'type="email"' in result
    assert 'value="user@example.com"' in result

def test_render_field_with_boolean_field():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"active": Boolean()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="active", field=schema.fields["active"], value=True)
    assert "forms/checkbox.html" in str(env.loader.templates)
    assert 'name="active"' in result

def test_render_field_with_choice_field():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"status": Choice(choices=[("active", "Active"), ("inactive", "Inactive")])})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="status", field=schema.fields["status"], value="active")
    assert "forms/select.html" in str(env.loader.templates)
    assert 'name="status"' in result
    assert '<option value="active"' in result

def test_render_field_with_error():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="name", field=schema.fields["name"], value="", error="This field is required")
    assert "This field is required" in result

def test_render_field_with_required_field():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="name", field=schema.fields["name"], value="")
    assert 'required' in result

def test_render_field_with_optional_field():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"name": String(allow_null=True)})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="name", field=schema.fields["name"], value="")
    assert 'required' not in result

def test_render_field_with_textarea():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"description": String(format="text")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="description", field=schema.fields["description"], value="Long text")
    assert "forms/textarea.html" in str(env.loader.templates)
    assert 'name="description"' in result
    assert "Long text" in result

def test_render_field_with_custom_title():
    env = jinja2.Environment(loader=jinja2.BaseLoader())
    schema = Schema(fields={"full_name": String(title="Full Name")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="full_name", field=schema.fields["full_name"], value="John Doe")
    assert "Full Name" in result


# LLM-generated content at query #5
#--------------------------

```python
def test_template_for_field_with_choice():
    form = Form(env=None, schema=None)
    choice_field = Choice()
    assert form.template_for_field(choice_field) == "forms/select.html"

def test_template_for_field_with_boolean():
    form = Form(env=None, schema=None)
    boolean_field = Boolean()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"

def test_template_for_field_with_string_format_text():
    form = Form(env=None, schema=None)
    string_field = String(format="text")
    assert form.template_for_field(string_field) == "forms/textarea.html"

def test_template_for_field_with_string_format_not_text():
    form = Form(env=None, schema=None)
    string_field = String(format="email")
    assert form.template_for_field(string_field) == "forms/input.html"

def test_template_for_field_with_object():
    form = Form(env=None, schema=None)
    object_field = Object()
    try:
        form.template_for_field(object_field)
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "Forms do not support rendering Object fields"

def test_template_for_field_with_other_field_types():
    form = Form(env=None, schema=None)
    integer_field = Integer()
    assert form.template_for_field(integer_field) == "forms/input.html"


# LLM-generated content at query #6
#--------------------------

```python
def test_template_for_field_with_choice():
    form = Form(env=None, schema=None)
    choice_field = Choice(any_of=[])
    result = form.template_for_field(choice_field)
    assert result == "forms/select.html"


# LLM-generated content at query #7
#--------------------------

```python
def test_template_for_field_with_choice():
    field = Choice(any_of=[])
    form = Form(env=jinja2.Environment(), schema=Schema(fields={}), values={})
    assert form.template_for_field(field) == "forms/select.html"


# LLM-generated content at query #8
#--------------------------

```python
def test_template_for_field_with_choice():
    form = Form(env=None, schema=None)
    choice_field = Choice()
    assert form.template_for_field(choice_field) == "forms/select.html"

def test_template_for_field_with_boolean():
    form = Form(env=None, schema=None)
    boolean_field = Boolean()
    assert form.template_for_field(boolean_field) == "forms/checkbox.html"

def test_template_for_field_with_string_format_text():
    form = Form(env=None, schema=None)
    string_field = String(format="text")
    assert form.template_for_field(string_field) == "forms/textarea.html"

def test_template_for_field_with_other_field():
    form = Form(env=None, schema=None)
    other_field = String(format="email")
    assert form.template_for_field(other_field) == "forms/input.html"

def test_template_for_field_with_object_raises_assertion():
    form = Form(env=None, schema=None)
    object_field = Object()
    try:
        form.template_for_field(object_field)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_template_for_field_with_choice():
    form = Form(env=None, schema=None)
    field = Choice()
    assert form.template_for_field(field) == "forms/select.html"

def test_template_for_field_with_boolean():
    form = Form(env=None, schema=None)
    field = Boolean()
    assert form.template_for_field(field) == "forms/checkbox.html"

def test_template_for_field_with_string_format_text():
    form = Form(env=None, schema=None)
    field = String(format="text")
    assert form.template_for_field(field) == "forms/textarea.html"

def test_template_for_field_with_string_format_not_text():
    form = Form(env=None, schema=None)
    field = String(format="email")
    assert form.template_for_field(field) == "forms/input.html"

def test_template_for_field_with_object():
    form = Form(env=None, schema=None)
    field = Object()
    try:
        form.template_for_field(field)
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "Forms do not support rendering Object fields"

def test_template_for_field_with_other_field():
    form = Form(env=None, schema=None)
    field = Number()
    assert form.template_for_field(field) == "forms/input.html"


# LLM-generated content at query #10
#--------------------------

```python
def test_template_for_field_with_boolean():
    form = Form(env=jinja2.Environment(), schema=Schema(fields={}), values={})
    field = Boolean()
    assert form.template_for_field(field) == "forms/checkbox.html"


# LLM-generated content at query #11
#--------------------------

```python
def test_required_false_when_field_has_default_and_allow_null():
    field = Field(allow_null=True, default="default_value")
    assert not (not field.has_default() and not (field.allow_null or getattr(field, "allow_blank", False)))


# LLM-generated content at query #12
#--------------------------

```python
def test_load_template_env_with_directory():
    jinja2_forms = Jinja2Forms(directory="test_templates")
    env = jinja2_forms.load_template_env(directory="test_templates")
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == ["test_templates"]
    assert env.autoescape is True

def test_load_template_env_with_package():
    jinja2_forms = Jinja2Forms(package="test_package")
    env = jinja2_forms.load_template_env(package="test_package")
    assert isinstance(env.loader, jinja2.PackageLoader)
    assert env.loader.package_name == "test_package"
    assert env.loader.package_path == "templates"
    assert env.autoescape is True

def test_load_template_env_with_both():
    jinja2_forms = Jinja2Forms(directory="test_templates", package="test_package")
    env = jinja2_forms.load_template_env(directory="test_templates", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)
    assert len(env.loader.loaders) == 2
    assert isinstance(env.loader.loaders[0], jinja2.FileSystemLoader)
    assert isinstance(env.loader.loaders[1], jinja2.PackageLoader)
    assert env.autoescape is True


# LLM-generated content at query #13
#--------------------------

```python
def test_allow_empty_with_allow_null():
    field = Field(allow_null=True)
    form = Form(env=jinja2.Environment(), schema=Schema(fields={}), values={})
    allow_empty = field.allow_null or getattr(field, "allow_blank", False)
    assert allow_empty is True


# LLM-generated content at query #14
#--------------------------

```python
def test_render_field_with_string_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}'>"}))
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="name", field=schema.fields["name"], value="test")
    assert "<input type='text' name='name' value='test'>" in result

def test_render_field_with_number_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}'>"}))
    schema = Schema(fields={"age": Number()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="age", field=schema.fields["age"], value=25)
    assert "<input type='number' name='age' value='25'>" in result

def test_render_field_with_boolean_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/checkbox.html": "<input type='checkbox' name='{{ field_name }}' {{ 'checked' if value else '' }}>"}))
    schema = Schema(fields={"active": Boolean()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="active", field=schema.fields["active"], value=True)
    assert "<input type='checkbox' name='active' checked>" in result

def test_render_field_with_choice_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/select.html": "<select name='{{ field_name }}'><option value='{{ value }}'>{{ value }}</option></select>"}))
    schema = Schema(fields={"status": Choice(choices=["active", "inactive"])})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="status", field=schema.fields["status"], value="active")
    assert "<select name='status'><option value='active'>active</option></select>" in result

def test_render_field_with_textarea():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/textarea.html": "<textarea name='{{ field_name }}'>{{ value }}</textarea>"}))
    schema = Schema(fields={"description": String(format="text")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="description", field=schema.fields["description"], value="long text")
    assert "<textarea name='description'>long text</textarea>" in result

def test_render_field_with_error():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}'><div class='error'>{{ error }}</div>"}))
    schema = Schema(fields={"name": String()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="name", field=schema.fields["name"], value="test", error="Invalid name")
    assert "<div class='error'>Invalid name</div>" in result

def test_render_field_with_password_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}'>"}))
    schema = Schema(fields={"password": String(format="password")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="password", field=schema.fields["password"], value="secret")
    assert "<input type='password' name='password' value=''>" in result

def test_render_field_with_required_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}' required='{{ required }}'>"}))
    schema = Schema(fields={"email": String()})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="email", field=schema.fields["email"], value="test@example.com")
    assert "<input type='text' name='email' value='test@example.com' required='True'>" in result

def test_render_field_with_optional_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}' required='{{ required }}'>"}))
    schema = Schema(fields={"email": String(default="")})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="email", field=schema.fields["email"], value="test@example.com")
    assert "<input type='text' name='email' value='test@example.com' required='False'>" in result

def test_render_field_with_nullable_field():
    env = jinja2.Environment(loader=jinja2.DictLoader({"forms/input.html": "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}' required='{{ required }}'>"}))
    schema = Schema(fields={"email": String(allow_null=True)})
    form = Form(env=env, schema=schema)
    result = form.render_field(field_name="email", field=schema.fields["email"], value="test@example.com")
    assert "<input type='text' name='email' value='test@example.com' required='False'>" in result


# LLM-generated content at query #15
#--------------------------

```python
def test_load_template_env_with_both_directory_and_package():
    obj = Jinja2Forms(directory="test_dir", package="test_pkg")
    assert obj.env.loader.loaders[0].searchpath == ["test_dir"]
    assert obj.env.loader.loaders[1].package_name == "test_pkg"


# LLM-generated content at query #16
#--------------------------

```python
def test_constructor_with_directory():
    instance = Jinja2Forms(directory="some/directory")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.FileSystemLoader)

def test_constructor_with_package():
    instance = Jinja2Forms(package="some.package")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.PackageLoader)

def test_constructor_with_both_directory_and_package():
    instance = Jinja2Forms(directory="some/directory", package="some.package")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.ChoiceLoader)

def test_constructor_without_directory_or_package():
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_load_template_env_with_both_directory_and_package():
    jinja2_forms = Jinja2Forms(directory="test_dir", package="test_package")
    env = jinja2_forms.load_template_env(directory="test_dir", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)


# LLM-generated content at query #18
#--------------------------

```python
def test_template_for_field_with_object_field():
    form = Form(env=None, schema=None)
    with pytest.raises(AssertionError, match="Forms do not support rendering Object fields"):
        form.template_for_field(Object())


# LLM-generated content at query #19
#--------------------------

```python
def test_template_for_field_with_object_field():
    form = Form(env=None, schema=None)
    with pytest.raises(AssertionError, match="Forms do not support rendering Object fields"):
        form.template_for_field(Object())


# LLM-generated content at query #20
#--------------------------

```python
def test_jinja2forms_constructor_with_directory():
    instance = Jinja2Forms(directory="test_dir")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.FileSystemLoader)

def test_jinja2forms_constructor_with_package():
    instance = Jinja2Forms(package="test_package")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.PackageLoader)

def test_jinja2forms_constructor_with_both_directory_and_package():
    instance = Jinja2Forms(directory="test_dir", package="test_package")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.ChoiceLoader)


# LLM-generated content at query #21
#--------------------------

```python
def test_jinja2forms_constructor_with_directory():
    assert Jinja2Forms(directory="test_dir").env.loader is not None

def test_jinja2forms_constructor_with_package():
    assert Jinja2Forms(package="test_package").env.loader is not None

def test_jinja2forms_constructor_with_both_directory_and_package():
    assert Jinja2Forms(directory="test_dir", package="test_package").env.loader is not None

def test_jinja2forms_constructor_without_directory_or_package():
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_jinja2_not_installed():
    global jinja2
    original_jinja2 = jinja2
    jinja2 = None
    try:
        Jinja2Forms(directory="test_dir")
    except AssertionError as e:
        assert str(e) == "jinja2 must be installed to use Jinja2Forms."
    finally:
        jinja2 = original_jinja2


# LLM-generated content at query #23
#--------------------------

```python
def test_template_for_field_with_object_field():
    form = Form(env=jinja2.Environment(), schema=Schema(fields={}), values={})
    with pytest.raises(AssertionError, match="Forms do not support rendering Object fields"):
        form.template_for_field(Object(fields={}))


# LLM-generated content at query #24
#--------------------------

```python
def test_load_template_env_with_both_none():
    jinja2_forms = Jinja2Forms()
    with pytest.raises(AssertionError):
        jinja2_forms.load_template_env(directory=None, package=None)


# LLM-generated content at query #25
#--------------------------

```python
def test_jinja2_not_installed():
    global jinja2
    jinja2 = None
    with pytest.raises(AssertionError, match="jinja2 must be installed to use Jinja2Forms."):
        Jinja2Forms(directory="test_dir")


# LLM-generated content at query #26
#--------------------------

```python
def test_template_for_field_with_object_field_raises_assertion_error():
    form = Form(env=None, schema=None)
    with pytest.raises(AssertionError, match="Forms do not support rendering Object fields"):
        form.template_for_field(Object())


# LLM-generated content at query #27
#--------------------------

```python
def test_load_template_env_with_both_none():
    with pytest.raises(AssertionError):
        Jinja2Forms(directory=None, package=None)


# LLM-generated content at query #28
#--------------------------

```python
def test_template_for_field_raises_assertion_error_for_object_field():
    form = Form(env=jinja2.Environment(), schema=Schema(fields={}), values={})
    with pytest.raises(AssertionError):
        form.template_for_field(Object(fields={}))


# LLM-generated content at query #29
#--------------------------

```python
def test_jinja2_not_installed():
    global jinja2
    jinja2 = None
    with pytest.raises(AssertionError, match="jinja2 must be installed to use Jinja2Forms."):
        Jinja2Forms(directory="some_dir")


# LLM-generated content at query #30
#--------------------------

```python
def test_load_template_env_with_both_directory_and_package():
    jinja2_forms = Jinja2Forms(directory="test_dir", package="test_pkg")
    env = jinja2_forms.load_template_env(directory="test_dir", package="test_pkg")
    assert isinstance(env.loader, jinja2.ChoiceLoader)


# LLM-generated content at query #31
#--------------------------

```python
def test_constructor_with_directory():
    instance = Jinja2Forms(directory="test_dir")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.FileSystemLoader)

def test_constructor_with_package():
    instance = Jinja2Forms(package="test_package")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.PackageLoader)

def test_constructor_with_both_directory_and_package():
    instance = Jinja2Forms(directory="test_dir", package="test_package")
    assert isinstance(instance.env, jinja2.Environment)
    assert isinstance(instance.env.loader, jinja2.ChoiceLoader)

def test_constructor_without_directory_or_package():
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_load_template_env_with_both_directory_and_package():
    jinja2_forms = Jinja2Forms(directory="test_dir", package="test_pkg")
    assert jinja2_forms.env.loader is not None


# LLM-generated content at query #33
#--------------------------

```python
def test_load_template_env_with_both_directory_and_package():
    jinja2_forms = Jinja2Forms(directory="test_dir", package="test_package")
    env = jinja2_forms.load_template_env(directory="test_dir", package="test_package")
    assert isinstance(env.loader, jinja2.ChoiceLoader)


# LLM-generated content at query #34
#--------------------------

```python
def test_constructor_with_directory():
    jf = Jinja2Forms(directory="test_dir")
    assert jf.env.loader is not None

def test_constructor_with_package():
    jf = Jinja2Forms(package="test_package")
    assert jf.env.loader is not None

def test_constructor_with_both_directory_and_package():
    jf = Jinja2Forms(directory="test_dir", package="test_package")
    assert jf.env.loader is not None

def test_constructor_without_directory_and_package():
    try:
        Jinja2Forms()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


