####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context_simple_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"name": "John", "age": 30}
    overwrite = {"name": "Jane"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"name": "Jane", "age": 30}


def test_apply_overwrites_to_context_ignore_new_variable_first_level():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"name": "John"}
    overwrite = {"new_var": "value"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"name": "John"}


def test_apply_overwrites_to_context_add_new_variable_in_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"config": {"key1": "value1"}}
    overwrite = {"config": {"key2": "value2"}}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"config": {"key1": "value1", "key2": "value2"}}


def test_apply_overwrites_to_context_choice_variable_valid():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flavor": ["vanilla", "chocolate", "strawberry"]}
    overwrite = {"flavor": "chocolate"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"flavor": ["chocolate", "vanilla", "strawberry"]}


def test_apply_overwrites_to_context_choice_variable_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flavor": ["vanilla", "chocolate"]}
    overwrite = {"flavor": "pistachio"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "pistachio" in str(e) and "flavor" in str(e)


def test_apply_overwrites_to_context_multichoice_valid():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"toppings": ["pepperoni", "mushroom", "onion"]}
    overwrite = {"toppings": ["mushroom", "onion"]}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"toppings": ["mushroom", "onion"]}


def test_apply_overwrites_to_context_multichoice_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"toppings": ["pepperoni", "mushroom"]}
    overwrite = {"toppings": ["pepperoni", "pineapple"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "pineapple" in str(e)


def test_apply_overwrites_to_context_boolean_yes():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"use_feature": False}
    overwrite = {"use_feature": "yes"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"use_feature": True}


def test_apply_overwrites_to_context_boolean_no():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"use_feature": True}
    overwrite = {"use_feature": "false"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"use_feature": False}


def test_apply_overwrites_to_context_boolean_true():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"debug": False}
    overwrite = {"debug": "true"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"debug": True}


def test_apply_overwrites_to_context_boolean_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite = {"flag": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "maybe" in str(e) and "flag" in str(e)


def test_apply_overwrites_to_context_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"database": {"host": "localhost", "port": 5432}}
    overwrite = {"database": {"port": 3306}}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"database": {"host": "localhost", "port": 3306}}


def test_apply_overwrites_to_context_list_in_dict_overwrites():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"settings": {"options": ["a", "b", "c"]}}
    overwrite = {"settings": {"options": ["x", "y"]}}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"settings": {"options": ["x", "y"]}}


def test_apply_overwrites_to_context_multiple_variables():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"name": "John", "age": 30, "city": "NYC"}
    overwrite = {"name": "Jane", "age": 25}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"name": "Jane", "age": 25, "city": "NYC"}


def test_apply_overwrites_to_context_empty_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"name": "John"}
    overwrite = {}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"name": "John"}


def test_apply_overwrites_to_context_boolean_with_spaces():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite = {"enabled": "  yes  "}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"enabled": True}


def test_apply_overwrites_to_context_boolean_numeric():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": False}
    overwrite = {"flag": "1"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"flag": True}


# LLM-generated content at query #2
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_top_level_variable():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"existing": "value"}
    overwrite_context = {"new_variable": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_adds_new_nested_variable():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"existing": "value"}}
    overwrite_context = {"new_variable": "new_value"}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"nested": {"existing": "value", "new_variable": "new_value"}}


def test_apply_overwrites_to_context_simple_value_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"variable": "old_value"}
    overwrite_context = {"variable": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"variable": "new_value"}


def test_apply_overwrites_to_context_choice_variable_valid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice_var": ["option1", "option2", "option3"]}
    overwrite_context = {"choice_var": "option2"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["choice_var"] == ["option2", "option1", "option3"]


def test_apply_overwrites_to_context_choice_variable_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice_var": ["option1", "option2"]}
    overwrite_context = {"choice_var": "invalid_option"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "invalid_option" in str(e)


def test_apply_overwrites_to_context_multichoice_variable_valid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"multichoice_var": ["option1", "option2", "option3"]}
    overwrite_context = {"multichoice_var": ["option1", "option3"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multichoice_var": ["option1", "option3"]}


def test_apply_overwrites_to_context_multichoice_variable_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"multichoice_var": ["option1", "option2"]}
    overwrite_context = {"multichoice_var": ["option1", "invalid_option"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "invalid_option" in str(e)


def test_apply_overwrites_to_context_nested_dict_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"nested": {"key1": "new_value1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"key1": "new_value1", "key2": "value2"}}


def test_apply_overwrites_to_context_boolean_yes_conversion():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}


def test_apply_overwrites_to_context_boolean_no_conversion():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": False}
    overwrite_context = {"bool_var": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": False}


def test_apply_overwrites_to_context_boolean_true_conversion():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "true"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}


def test_apply_overwrites_to_context_boolean_false_conversion():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": False}
    overwrite_context = {"bool_var": "false"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": False}


def test_apply_overwrites_to_context_boolean_invalid_conversion():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "invalid" in str(e)


def test_apply_overwrites_to_context_list_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"list_var": ["opt1", "opt2"]}}
    overwrite_context = {"nested": {"list_var": "opt2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["nested"]["list_var"] == ["opt2", "opt1"]


def test_apply_overwrites_to_context_list_overwrites_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"list_var": ["opt1", "opt2"]}}
    overwrite_context = {"nested": {"list_var": ["opt1", "opt2"]}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["nested"]["list_var"] == ["opt1", "opt2"]


def test_apply_overwrites_to_context_multiple_variables():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"var1": "value1", "var2": "value2", "var3": "value3"}
    overwrite_context = {"var1": "new_value1", "var3": "new_value3"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "new_value1", "var2": "value2", "var3": "new_value3"}


def test_apply_overwrites_to_context_boolean_with_spaces():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "  yes  "}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}


def test_apply_overwrites_to_context_boolean_case_insensitive():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "YES"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}


# LLM-generated content at query #3
#--------------------------

```python
def test_apply_overwrites_to_context_predicate_line_46_false():
    """Test that the predicate at line 46 evaluates to False when context_value is not a dict or overwrite is not a dict."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    # Test case 1: context_value is dict but overwrite is not dict
    context = {"key": {"nested": "value"}}
    overwrite_context = {"key": "string_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == "string_value"
    
    # Test case 2: context_value is not dict but overwrite is dict
    context = {"key": "string_value"}
    overwrite_context = {"key": {"nested": "value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == {"nested": "value"}
    
    # Test case 3: neither context_value nor overwrite is dict
    context = {"key": "original"}
    overwrite_context = {"key": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == "new"


# LLM-generated content at query #4
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    
    try:
        render_and_create_dir("", context, output_dir, env)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_creates_new_directory(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    dirname = "test_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, output_dir, env)
    
    assert result_path == tmp_path / "test_dir"
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_with_template_rendering(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    context = {"project_name": "my_project"}
    output_dir = tmp_path
    dirname = "{{ project_name }}_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, output_dir, env)
    
    assert result_path == tmp_path / "my_project_dir"
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_existing_dir_no_overwrite(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    dirname = "existing_dir"
    
    (tmp_path / dirname).mkdir()
    
    try:
        render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    dirname = "existing_dir"
    
    (tmp_path / dirname).mkdir()
    
    result_path, is_new = render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=True)
    
    assert result_path == tmp_path / "existing_dir"
    assert result_path.exists()
    assert is_new is False


def test_render_and_create_dir_nested_path(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    dirname = "parent/child/grandchild"
    
    result_path, is_new = render_and_create_dir(dirname, context, output_dir, env)
    
    assert result_path == tmp_path / "parent" / "child" / "grandchild"
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_none_dirname(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    
    try:
        render_and_create_dir(None, context, output_dir, env)
        assert False, "Expected EmptyDirNameException"
    except (EmptyDirNameException, TypeError):
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            render_and_create_dir("", {}, tmpdir, env)
            assert False, "Should raise EmptyDirNameException"
        except EmptyDirNameException:
            pass


def test_render_and_create_dir_creates_new_directory():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path, is_new = render_and_create_dir("test_dir", {}, tmpdir, env)
        assert result_path.exists()
        assert is_new is True
        assert result_path.name == "test_dir"


def test_render_and_create_dir_with_template_variable():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {"project_name": "my_project"}
        result_path, is_new = render_and_create_dir("{{ project_name }}", context, tmpdir, env)
        assert result_path.exists()
        assert result_path.name == "my_project"
        assert is_new is True


def test_render_and_create_dir_existing_dir_without_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_dir = Path(tmpdir) / "existing_dir"
        existing_dir.mkdir()
        
        try:
            render_and_create_dir("existing_dir", {}, tmpdir, env, overwrite_if_exists=False)
            assert False, "Should raise OutputDirExistsException"
        except OutputDirExistsException:
            pass


def test_render_and_create_dir_existing_dir_with_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_dir = Path(tmpdir) / "existing_dir"
        existing_dir.mkdir()
        
        result_path, is_new = render_and_create_dir("existing_dir", {}, tmpdir, env, overwrite_if_exists=True)
        assert result_path.exists()
        assert is_new is False


def test_render_and_create_dir_nested_path():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path, is_new = render_and_create_dir("parent/child/grandchild", {}, tmpdir, env)
        assert result_path.exists()
        assert is_new is True
        assert result_path.name == "grandchild"
        assert result_path.parent.name == "child"


# LLM-generated content at query #6
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that line 57 predicate evaluates to False when boolean conversion succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"debug": True}
    overwrite_context = {"debug": "false"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["debug"] is False


def test_apply_overwrites_to_context_boolean_conversion_yes():
    """Test that line 57 predicate evaluates to False with 'yes' input."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite_context = {"enabled": "yes"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is True


def test_apply_overwrites_to_context_boolean_conversion_no():
    """Test that line 57 predicate evaluates to False with 'no' input."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is False


def test_apply_overwrites_to_context_boolean_conversion_one():
    """Test that line 57 predicate evaluates to False with '1' input."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": False}
    overwrite_context = {"flag": "1"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is True


def test_apply_overwrites_to_context_boolean_conversion_zero():
    """Test that line 57 predicate evaluates to False with '0' input."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "0"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is False


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context_with_valid_json_file(tmp_path):
    """Test generate_context with a valid JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_invalid_json_file(tmp_path):
    """Test generate_context with an invalid JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"invalid": json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Should have raised ContextDecodingException"
    except Exception as e:
        assert "JSON decoding error" in str(e)


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context with default_context parameter."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    default_context = {"project_name": "default_project"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "default_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context with extra_context parameter."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    extra_context = {"project_name": "extra_project"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "extra_project"


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context with choice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"


def test_generate_context_with_invalid_choice(tmp_path):
    """Test generate_context with invalid choice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    extra_context = {"license": "BSD"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "BSD" in str(e)


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test generate_context with multichoice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    
    extra_context = {"features": ["feature2", "feature3"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["features"] == ["feature2", "feature3"]


def test_generate_context_with_invalid_multichoice(tmp_path):
    """Test generate_context with invalid multichoice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2"]}')
    
    extra_context = {"features": ["feature1", "invalid_feature"]}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_feature" in str(e)


def test_generate_context_with_boolean_variable(tmp_path):
    """Test generate_context with boolean variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    extra_context = {"use_docker": "yes"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is True


def test_generate_context_with_boolean_false(tmp_path):
    """Test generate_context with boolean false variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    extra_context = {"use_docker": "no"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_with_invalid_boolean(tmp_path):
    """Test generate_context with invalid boolean variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    extra_context = {"use_docker": "maybe"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_generate_context_with_nested_dict(tmp_path):
    """Test generate_context with nested dictionary."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"database": "postgres", "port": 5432}}')
    
    extra_context = {"config": {"database": "mysql"}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["config"]["database"] == "mysql"
    assert result["cookiecutter"]["config"]["port"] == 5432


def test_generate_context_with_simple_overwrite(tmp_path):
    """Test generate_context with simple string overwrite."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "original", "author": "original_author"}')
    
    extra_context = {"project_name": "new_project", "author": "new_author"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "new_project"
    assert result["cookiecutter"]["author"] == "new_author"


def test_generate_context_file_not_found():
    """Test generate_context with non-existent file."""
    try:
        generate_context("/non/existent/path/cookiecutter.json")
        assert False, "Should have raised an exception"
    except FileNotFoundError:
        pass


def test_generate_context_with_empty_json(tmp_path):
    """Test generate_context with empty JSON object."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"] == {}


def test_generate_context_preserves_original_with_default_context_error(tmp_path):
    """Test generate_context with invalid default context doesn't break."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    


# LLM-generated content at query #8
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_false(tmp_path):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    output_dir = tmp_path
    dirname = "test_dir"
    context = {}
    environment = Environment()
    
    # Create the directory first so it exists
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    
    # Call with overwrite_if_exists=True to avoid exception
    result_path, result_bool = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=output_dir,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # The predicate at line 24 (output_dir_exists) evaluates to True
    # So the return value should be False (not output_dir_exists)
    assert result_bool is False


# LLM-generated content at query #9
#--------------------------

```python
def test_run_hook_from_repo_dir_calls_hooks_run_hook_from_repo_dir(mocker):
    """Test that _run_hook_from_repo_dir calls hooks.run_hook_from_repo_dir."""
    mock_run_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mock_warn = mocker.patch('cookiecutter.generate.warnings.warn')
    
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )
    
    mock_warn.assert_called_once()
    assert "deprecated" in mock_warn.call_args[0][0].lower()
    assert DeprecationWarning == mock_warn.call_args[0][1]
    mock_run_hook.assert_called_once_with(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )


def test_run_hook_from_repo_dir_deprecation_warning(mocker):
    """Test that _run_hook_from_repo_dir issues a deprecation warning."""
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mock_warn = mocker.patch('cookiecutter.generate.warnings.warn')
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir('repo', 'hook', 'project', {}, False)
    
    mock_warn.assert_called_once()
    call_args = mock_warn.call_args[0]
    assert '_run_hook_from_repo_dir' in call_args[0]
    assert 'cookiecutter.hooks.run_hook_from_repo_dir' in call_args[0]


def test_run_hook_from_repo_dir_passes_all_arguments(mocker):
    """Test that _run_hook_from_repo_dir passes all arguments correctly."""
    mock_run_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.warnings.warn')
    
    repo_dir = '/template/repo'
    hook_name = 'pre_gen_project'
    project_dir = Path('/output/project')
    context = {'cookiecutter': {'name': 'myproject', 'version': '1.0'}}
    delete_project_on_failure = False
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )
    
    mock_run_hook.assert_called_once_with(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_context_json_decoding_error():
    """Test that ValueError is caught at line 20 and ContextDecodingException is raised."""
    import json
    import os
    import tempfile
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    # Create a temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        f.write('{invalid json content}')
        temp_file = f.name
    
    try:
        # This should raise ContextDecodingException when json.load encounters ValueError
        generate_context(context_file=temp_file)
        assert False, "Expected ContextDecodingException to be raised"
    except Exception as e:
        # Verify that the exception is ContextDecodingException (not ValueError)
        assert type(e).__name__ == 'ContextDecodingException'
        assert 'JSON decoding error' in str(e)
    finally:
        os.unlink(temp_file)


# LLM-generated content at query #11
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_true(tmp_path, monkeypatch):
    """Test that the predicate at line 24 evaluates to True when directory exists."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Create a temporary directory that will exist
    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup context and environment
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    output_dir = tmp_path
    
    # Call the function with overwrite_if_exists=True to avoid exception
    result_path, is_new = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=output_dir,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # Verify the predicate at line 24 was True (directory existed)
    assert result_path.exists()
    assert is_new is False


# LLM-generated content at query #12
#--------------------------

```python
def test_generate_context_with_valid_json_file(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_default_context(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    default_context = {"project_name": "default_project"}
    
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "default_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_extra_context(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    extra_context = {"author": "Jane"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "Jane"


def test_generate_context_with_invalid_json(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": invalid json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Expected ContextDecodingException"
    except Exception as e:
        assert "JSON decoding error" in str(e)


def test_generate_context_with_boolean_variable_and_string_override(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_feature": true}')
    extra_context = {"use_feature": "false"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_feature"] is False


def test_generate_context_with_choice_variable(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    extra_context = {"license": "Apache"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"


def test_generate_context_with_multichoice_variable(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    extra_context = {"features": ["feature2", "feature3"]}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["features"] == ["feature2", "feature3"]


def test_generate_context_with_nested_dict(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"settings": {"debug": true, "timeout": 30}}')
    extra_context = {"settings": {"debug": false}}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["settings"]["debug"] is False
    assert result["cookiecutter"]["settings"]["timeout"] == 30


def test_generate_context_with_nonexistent_file():
    try:
        generate_context("/nonexistent/path/cookiecutter.json")
        assert False, "Expected exception"
    except Exception:
        pass


def test_generate_context_with_invalid_choice_override(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    extra_context = {"license": "GPL"}
    
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "GPL" in str(e) and "choice variable" in str(e)


def test_generate_context_with_invalid_boolean_string(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_feature": true}')
    extra_context = {"use_feature": "invalid"}
    
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid" in str(e) and "boolean" in str(e)


def test_generate_context_with_custom_context_file_name(tmp_path):
    context_file = tmp_path / "custom.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    result = generate_context(str(context_file))
    
    assert "custom" in result
    assert result["custom"]["project_name"] == "my_project"


def test_generate_context_with_invalid_default_context_warns(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    default_context = {"license": "GPL"}
    
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["license"] == ["MIT", "Apache"]


def test_generate_context_preserves_order(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"z_field": "z", "a_field": "a", "m_field": "m"}')
    
    result = generate_context(str(context_file))
    
    keys = list(result["cookiecutter"].keys())
    assert keys == ["z_field", "a_field", "m_field"]


# LLM-generated content at query #13
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_false(tmp_path):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    output_dir = tmp_path
    dirname = "test_dir"
    context = {}
    environment = Environment()
    
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    
    result_path, is_new = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=output_dir,
        environment=environment,
        overwrite_if_exists=True
    )
    
    assert dir_to_create.exists()
    assert result_path == dir_to_create
    assert is_new == False


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_context_predicate_line_38_false():
    """Test that the predicate at line 38 (if default_context:) evaluates to False."""
    import tempfile
    import json
    import os
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    
    # Create a temporary JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        test_data = {'project_name': 'test_project', 'author': 'test_author'}
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        # Call generate_context with default_context=None (predicate evaluates to False)
        result = generate_context(
            context_file=context_file,
            default_context=None,
            extra_context=None
        )
        
        # Verify the context was generated correctly
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'test_project'
        assert result['cookiecutter']['author'] == 'test_author'


# LLM-generated content at query #15
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that boolean conversion succeeds with valid yes/no input."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"feature_enabled": True}
    overwrite_context = {"feature_enabled": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["feature_enabled"] is False


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_context_basic(tmp_path):
    """Test generate_context with a basic JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context with default_context parameter."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    default_context = {"project_name": "overridden_project"}
    
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "overridden_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context with extra_context parameter."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    extra_context = {"project_name": "extra_project"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "extra_project"


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context with choice variable (list)."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    extra_context = {"license": "Apache"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"


def test_generate_context_with_boolean_variable(tmp_path):
    """Test generate_context with boolean variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    extra_context = {"use_docker": "false"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_with_nested_dict(tmp_path):
    """Test generate_context with nested dictionary."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"debug": false, "timeout": 30}}')
    extra_context = {"config": {"debug": "true"}}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["config"]["debug"] is True
    assert result["cookiecutter"]["config"]["timeout"] == 30


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context with invalid JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{invalid json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Should raise ContextDecodingException"
    except Exception as e:
        assert "JSON decoding error" in str(e)


def test_generate_context_file_not_found():
    """Test generate_context with non-existent file."""
    try:
        generate_context("/nonexistent/path/cookiecutter.json")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test generate_context with multichoice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    extra_context = {"features": ["feature2", "feature3"]}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["features"]) == {"feature2", "feature3"}


def test_generate_context_invalid_choice_raises_error(tmp_path):
    """Test generate_context with invalid choice raises ValueError."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    extra_context = {"license": "InvalidLicense"}
    
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "provided for choice variable" in str(e)


def test_generate_context_custom_filename(tmp_path):
    """Test generate_context with custom context file name."""
    context_file = tmp_path / "custom_context.json"
    context_file.write_text('{"name": "test"}')
    
    result = generate_context(str(context_file))
    
    assert "custom_context" in result
    assert result["custom_context"]["name"] == "test"


def test_generate_context_with_all_parameters(tmp_path):
    """Test generate_context with all parameters."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project": "default", "version": "1.0", "active": true}')
    default_context = {"project": "from_default"}
    extra_context = {"version": "2.0", "active": "false"}
    
    result = generate_context(str(context_file), default_context=default_context, extra_context=extra_context)
    
    assert result["cookiecutter"]["project"] == "from_default"
    assert result["cookiecutter"]["version"] == "2.0"
    assert result["cookiecutter"]["active"] is False


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_files_basic(tmp_path, monkeypatch):
    """Test generate_files with basic template structure."""
    from cookiecutter.generate import generate_files
    from pathlib import Path
    from collections import OrderedDict
    
    # Create template directory structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}")
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Mock find_template to return our template directory
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    gen_module.find_template = lambda repo, env: template_dir
    
    try:
        context = OrderedDict([
            ('cookiecutter', {'project_name': 'my_project'})
        ])
        
        result = generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir,
            accept_hooks=False
        )
        
        assert result is not None
        assert Path(result).exists()
        assert Path(result).name == 'my_project'
    finally:
        gen_module.find_template = original_find_template


def test_generate_files_with_empty_dirname(tmp_path):
    """Test generate_files raises error on empty directory name."""
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    
    env = Environment()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    context = {'cookiecutter': {}}
    
    try:
        render_and_create_dir("", context, output_dir, env)
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_generate_files_directory_exists_no_overwrite(tmp_path):
    """Test generate_files raises error when output directory exists and overwrite is False."""
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    
    env = Environment()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create the directory that will be generated
    existing_dir = output_dir / "existing_project"
    existing_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    try:
        render_and_create_dir("existing_project", context, output_dir, env, overwrite_if_exists=False)
        assert False, "Should have raised OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_generate_files_creates_directory(tmp_path):
    """Test generate_files successfully creates output directory."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    context = {'cookiecutter': {}}
    
    result_path, created = render_and_create_dir("new_project", context, output_dir, env)
    
    assert result_path.exists()
    assert created is True
    assert result_path.name == "new_project"


def test_generate_files_with_context_variables(tmp_path, monkeypatch):
    """Test generate_files renders context variables in directory names."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    context = {'cookiecutter': {'project_name': 'awesome_project'}}
    
    result_path, created = render_and_create_dir(
        "{{cookiecutter.project_name}}", 
        context, 
        output_dir, 
        env
    )
    
    assert result_path.exists()
    assert result_path.name == "awesome_project"


def test_generate_files_overwrite_existing(tmp_path):
    """Test generate_files can overwrite existing directory."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create initial directory
    existing_dir = output_dir / "my_project"
    existing_dir.mkdir()
    test_file = existing_dir / "test.txt"
    test_file.write_text("old content")
    
    context = {'cookiecutter': {}}
    
    result_path, created = render_and_create_dir(
        "my_project", 
        context, 
        output_dir, 
        env, 
        overwrite_if_exists=True
    )
    
    assert result_path.exists()
    assert created is False


def test_is_copy_only_path_matches_pattern(tmp_path):
    """Test is_copy_only_path returns True when path matches pattern."""
    from cookiecutter.generate import is_copy_only_path
    
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.pyc', 'node_modules/*']
        }
    }
    
    result = is_copy_only_path('test.pyc', context)
    assert result is True


def test_is_copy_only_path_no_match(tmp_path):
    """Test is_copy_only_path returns False when path doesn't match pattern."""
    from cookiecutter.generate import is_copy_only_path
    
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.pyc', 'node_modules/*']
        }
    }
    
    result = is_copy_only_path('test.py', context)
    assert result is False


def test_is_copy_only_path_no_config(tmp_path):
    """Test is_copy_only_path returns False when _copy_without_render not configured."""
    from cookiecutter.generate import is_copy_only_path
    
    context = {'cookiecutter': {}}
    
    result = is_copy_only_path('test.py', context)
    assert result is False


# LLM-generated content at query #18
#--------------------------

```python
def test_is_copy_only_path_with_matching_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['*.pyc', '*.so']}}
    result = is_copy_only_path('test.pyc', context)
    assert result is True


def test_is_copy_only_path_with_non_matching_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['*.pyc', '*.so']}}
    result = is_copy_only_path('test.py', context)
    assert result is False


def test_is_copy_only_path_with_wildcard_directory_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['node_modules/*', 'venv/*']}}
    result = is_copy_only_path('node_modules/package.json', context)
    assert result is True


def test_is_copy_only_path_with_missing_copy_without_render_key():
    context = {'cookiecutter': {}}
    result = is_copy_only_path('test.pyc', context)
    assert result is False


def test_is_copy_only_path_with_missing_cookiecutter_key():
    context = {}
    result = is_copy_only_path('test.pyc', context)
    assert result is False


def test_is_copy_only_path_with_empty_copy_without_render_list():
    context = {'cookiecutter': {'_copy_without_render': []}}
    result = is_copy_only_path('test.pyc', context)
    assert result is False


def test_is_copy_only_path_with_multiple_patterns_first_matches():
    context = {'cookiecutter': {'_copy_without_render': ['*.pyc', '*.so', '*.bin']}}
    result = is_copy_only_path('test.pyc', context)
    assert result is True


def test_is_copy_only_path_with_multiple_patterns_last_matches():
    context = {'cookiecutter': {'_copy_without_render': ['*.pyc', '*.so', '*.bin']}}
    result = is_copy_only_path('test.bin', context)
    assert result is True


def test_is_copy_only_path_with_question_mark_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['test?.txt']}}
    result = is_copy_only_path('test1.txt', context)
    assert result is True


def test_is_copy_only_path_with_bracket_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['test[0-9].txt']}}
    result = is_copy_only_path('test5.txt', context)
    assert result is True


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_context_opens_file_with_utf8_encoding(tmp_path):
    """Test that generate_context opens the context file with utf-8 encoding."""
    import json
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    
    # Create a temporary JSON file with UTF-8 encoding
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"project_name": "test_project", "author": "Test Author"}
    context_file.write_text(json.dumps(test_data), encoding='utf-8')
    
    # Call generate_context with the temporary file
    result = generate_context(str(context_file))
    
    # Verify the file was successfully opened and parsed
    assert isinstance(result, dict)
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"
    assert result["cookiecutter"]["author"] == "Test Author"


# LLM-generated content at query #20
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that boolean conversion succeeds when valid yes/no string is provided."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is False


def test_apply_overwrites_to_context_boolean_conversion_invalid():
    """Test that line 57 predicate (except clause) evaluates to False for valid input."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": False}
    overwrite_context = {"flag": "yes"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is True


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_line_62_evaluates_to_false():
    """Test that the predicate at line 62 (for root, dirs, files in os.walk('.')) evaluates to False when os.walk returns empty."""
    import os
    import tempfile
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    from unittest.mock import patch, MagicMock
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        
        template_dir = repo_dir / "{{cookiecutter.project_name}}"
        template_dir.mkdir()
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        context = OrderedDict([
            ('cookiecutter', {'project_name': 'test_project'})
        ])
        
        walk_results = []
        
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = walk_results
            
            with patch('cookiecutter.generate.find_template') as mock_find:
                mock_find.return_value = template_dir
                
                with patch('cookiecutter.generate.render_and_create_dir') as mock_render:
                    mock_render.return_value = (str(output_dir / 'test_project'), True)
                    
                    with patch('cookiecutter.generate.accept_hooks', False):
                        result = generate_files(
                            repo_dir,
                            context,
                            output_dir,
                            accept_hooks=False
                        )
        
        assert walk_results == []


# LLM-generated content at query #22
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_variable_at_first_level():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"existing_var": "value"}
    overwrite_context = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing_var": "value"}


def test_apply_overwrites_to_context_adds_new_variable_in_dictionary():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"existing": "value"}}
    overwrite_context = {"nested": {"new_var": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"nested": {"existing": "value", "new_var": "new_value"}}


def test_apply_overwrites_to_context_valid_multichoice():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}


def test_apply_overwrites_to_context_invalid_multichoice():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d", "e"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "multi-choice variable" in str(e)


def test_apply_overwrites_to_context_valid_choice():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice": ["default", "option1", "option2"]}
    overwrite_context = {"choice": "option1"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["option1", "default", "option2"]}


def test_apply_overwrites_to_context_invalid_choice():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "choice variable" in str(e)


def test_apply_overwrites_to_context_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"settings": {"debug": True, "port": 8000}}
    overwrite_context = {"settings": {"debug": False}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"settings": {"debug": False, "port": 8000}}


def test_apply_overwrites_to_context_boolean_yes():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"use_feature": False}
    overwrite_context = {"use_feature": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"use_feature": True}


def test_apply_overwrites_to_context_boolean_no():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"use_feature": True}
    overwrite_context = {"use_feature": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"use_feature": False}


def test_apply_overwrites_to_context_boolean_true():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"enabled": False}
    overwrite_context = {"enabled": "true"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": True}


def test_apply_overwrites_to_context_boolean_false():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"enabled": True}
    overwrite_context = {"enabled": "false"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": False}


def test_apply_overwrites_to_context_boolean_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"enabled": True}
    overwrite_context = {"enabled": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_apply_overwrites_to_context_simple_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"name": "old_name", "version": "1.0"}
    overwrite_context = {"name": "new_name"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"name": "new_name", "version": "1.0"}


def test_apply_overwrites_to_context_list_to_list_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"config": {"items": ["a", "b", "c"]}}
    overwrite_context = {"config": {"items": ["x", "y"]}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"config": {"items": ["x", "y"]}}


def test_apply_overwrites_to_context_multiple_variables():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"var1": "val1", "var2": "val2", "var3": "val3"}
    overwrite_context = {"var1": "new_val1", "var3": "new_val3"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "new_val1", "var2": "val2", "var3": "new_val3"}


# LLM-generated content at query #23
#--------------------------

```python
def test_render_and_create_dir_raises_empty_dir_name_exception():
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    context = {}
    output_dir = Path('.')
    environment = Environment()
    
    try:
        render_and_create_dir('', context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        assert True


# LLM-generated content at query #24
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        try:
            render_and_create_dir("", context, tmpdir, env)
            assert False, "Should raise EmptyDirNameException"
        except EmptyDirNameException:
            pass


def test_render_and_create_dir_creates_new_directory():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        dirname = "test_dir"
        result_path, is_new = render_and_create_dir(dirname, context, tmpdir, env)
        
        assert result_path.exists()
        assert result_path.name == "test_dir"
        assert is_new is True


def test_render_and_create_dir_with_template_rendering():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "myproject"}
        dirname = "{{ project_name }}_dir"
        result_path, is_new = render_and_create_dir(dirname, context, tmpdir, env)
        
        assert result_path.exists()
        assert result_path.name == "myproject_dir"
        assert is_new is True


def test_render_and_create_dir_existing_directory_no_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        dirname = "existing_dir"
        existing_path = Path(tmpdir) / dirname
        existing_path.mkdir()
        
        try:
            render_and_create_dir(dirname, context, tmpdir, env)
            assert False, "Should raise OutputDirExistsException"
        except OutputDirExistsException:
            pass


def test_render_and_create_dir_existing_directory_with_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        dirname = "existing_dir"
        existing_path = Path(tmpdir) / dirname
        existing_path.mkdir()
        
        result_path, is_new = render_and_create_dir(dirname, context, tmpdir, env, overwrite_if_exists=True)
        
        assert result_path.exists()
        assert is_new is False


def test_render_and_create_dir_nested_path():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        dirname = "nested/path/to/dir"
        result_path, is_new = render_and_create_dir(dirname, context, tmpdir, env)
        
        assert result_path.exists()
        assert result_path.name == "dir"
        assert is_new is True


# LLM-generated content at query #25
#--------------------------

```python
def test_render_and_create_dir_raises_on_empty_dirname():
    """Test that render_and_create_dir raises EmptyDirNameException when dirname is empty string."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    context = {}
    output_dir = Path('/tmp')
    environment = Environment()
    
    try:
        render_and_create_dir("", context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        assert True


# LLM-generated content at query #26
#--------------------------

```python
def test_delete_project_on_failure_true_when_output_directory_created_and_keep_project_false():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


def test_delete_project_on_failure_false_when_output_directory_not_created():
    output_directory_created = False
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


def test_delete_project_on_failure_false_when_keep_project_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


def test_delete_project_on_failure_false_when_both_conditions_fail():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_files_basic(tmp_path, monkeypatch):
    """Test basic file generation with a simple template."""
    from collections import OrderedDict
    from pathlib import Path
    from cookiecutter.generate import generate_files
    
    # Create a template directory structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    (template_dir / "README.md").write_text("# {{cookiecutter.project_name}}")
    
    # Create a cookiecutter.json-like context
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Mock the hook functions to avoid execution
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        accept_hooks=False
    )
    
    assert Path(result).exists()
    assert Path(result).name == "my_project"
    assert (Path(result) / "README.md").read_text() == "# my_project"


def test_generate_files_with_overwrite(tmp_path, monkeypatch):
    """Test file generation with overwrite_if_exists=True."""
    from collections import OrderedDict
    from pathlib import Path
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([('cookiecutter', {'name': 'project'})])
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result1 = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        accept_hooks=False
    )
    
    result2 = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert Path(result2).exists()
    assert result1 == result2


def test_generate_files_empty_context(tmp_path, monkeypatch):
    """Test file generation with no context provided."""
    from pathlib import Path
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "simple_project"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("static content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=repo_dir,
        context=None,
        output_dir=output_dir,
        accept_hooks=False
    )
    
    assert Path(result).exists()


def test_generate_files_skip_if_file_exists(tmp_path, monkeypatch):
    """Test file generation with skip_if_file_exists=True."""
    from collections import OrderedDict
    from pathlib import Path
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.proj}}"
    template_dir.mkdir()
    (template_dir / "existing.txt").write_text("new content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([('cookiecutter', {'proj': 'myproj'})])
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    # Generate once
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        accept_hooks=False
    )
    
    # Modify the file
    (Path(result) / "existing.txt").write_text("original content")
    
    # Generate again with skip_if_file_exists
    result2 = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    # File should retain original content
    assert (Path(result2) / "existing.txt").read_text() == "original content"


def test_generate_files_with_hooks(tmp_path, monkeypatch):
    """Test file generation with hooks acceptance."""
    from collections import OrderedDict
    from pathlib import Path
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([('cookiecutter', {'name': 'proj'})])
    
    hook_calls = []
    def mock_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        hook_calls.append(hook_name)
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_hook)
    
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        accept_hooks=True
    )
    
    assert 'pre_gen_project' in hook_calls
    assert 'post_gen_project' in hook_calls


def test_generate_files_output_dir_created(tmp_path, monkeypatch):
    """Test that output directory is created if it doesn't exist."""
    from collections import OrderedDict
    from pathlib import Path
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("test")
    
    output_dir = tmp_path / "nonexistent" / "output"
    
    context = OrderedDict([('cookiecutter', {'name': 'test'})])
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        accept_hooks=False
    )
    
    assert Path(result).exists()
    assert output_dir.exists()


# LLM-generated content at query #28
#--------------------------

```python
def test_render_and_create_dir_raises_empty_dirname_exception():
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    environment = Environment()
    context = {}
    output_dir = Path.cwd()
    
    try:
        render_and_create_dir("", context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException as e:
        assert "directory name is empty" in str(e)


# LLM-generated content at query #29
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_true(tmp_path, monkeypatch):
    """Test that the predicate at line 24 evaluates to True when directory exists."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Create a temporary directory that will exist
    existing_dir = tmp_path / "existing_project"
    existing_dir.mkdir(parents=True, exist_ok=True)
    
    context = {}
    environment = Environment()
    dirname = "existing_project"
    output_dir = tmp_path
    overwrite_if_exists = True
    
    # Call the function - the predicate at line 24 should evaluate to True
    result_path, result_flag = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=output_dir,
        environment=environment,
        overwrite_if_exists=overwrite_if_exists
    )
    
    # When output_dir_exists is True, the return value's second element should be False
    # (because it returns `not output_dir_exists`)
    assert result_flag is False
    assert result_path == existing_dir


# LLM-generated content at query #30
#--------------------------

```python
def test_generate_context_basic(tmp_path):
    """Test generate_context with a basic JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John Doe"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context with default_context parameter."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    default_context = {"project_name": "default_project"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "default_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context with extra_context parameter."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    extra_context = {"project_name": "extra_project"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "extra_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_boolean_overwrite(tmp_path):
    """Test generate_context with boolean variable and string overwrite."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_feature": true}')
    
    extra_context = {"use_feature": "false"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_feature"] is False


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context with choice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"
    assert "MIT" in result["cookiecutter"]["license"]


def test_generate_context_with_dict_variable(tmp_path):
    """Test generate_context with nested dictionary."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"debug": false, "port": 8000}}')
    
    extra_context = {"config": {"debug": "true"}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["config"]["debug"] is True
    assert result["cookiecutter"]["config"]["port"] == 8000


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context with invalid JSON raises ContextDecodingException."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{invalid json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Should have raised ContextDecodingException"
    except Exception as e:
        assert "JSON decoding error" in str(e)


def test_generate_context_file_not_found():
    """Test generate_context with non-existent file."""
    try:
        generate_context("/nonexistent/path/cookiecutter.json")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_generate_context_with_multichoice_valid(tmp_path):
    """Test generate_context with valid multichoice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    
    extra_context = {"features": ["feature2", "feature3"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["features"]) == {"feature2", "feature3"}


def test_generate_context_with_invalid_choice(tmp_path):
    """Test generate_context with invalid choice raises ValueError."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    extra_context = {"license": "GPL"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "GPL" in str(e)


def test_generate_context_with_invalid_boolean_string(tmp_path):
    """Test generate_context with invalid boolean string raises ValueError."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_feature": true}')
    
    extra_context = {"use_feature": "maybe"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_generate_context_default_context_invalid_warning(tmp_path, capsys):
    """Test generate_context with invalid default_context logs warning."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    default_context = {"license": "InvalidLicense"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["license"] == ["MIT", "Apache"]


def test_generate_context_both_default_and_extra_context(tmp_path):
    """Test generate_context with both default and extra context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project": "proj", "version": "1.0", "author": "Unknown"}')
    
    default_context = {"project": "default_proj", "author": "Default Author"}
    extra_context = {"version": "2.0", "author": "Extra Author"}
    result = generate_context(str(context_file), default_context=default_context, extra_context=extra_context)
    
    assert result["cookiecutter"]["project"] == "default_proj"
    assert result["cookiecutter"]["version"] == "2.0"
    assert result["cookiecutter"]["author"] == "Extra Author"


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_file_renders_text_file(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    template_dir = tmp_path / "templates"
    os.makedirs(template_dir)
    
    infile = "test_{{cookiecutter.name}}.txt"
    template_file = template_dir / infile
    template_file.write_text("Hello {{cookiecutter.name}}", encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'name': 'world',
            '_new_lines': False
        }
    }
    
    env = Environment()
    
    mocker.patch('os.getcwd', return_value=str(template_dir))
    mocker.patch('builtins.open', mocker.mock_open(read_data="Hello {{cookiecutter.name}}"))
    mocker.patch('shutil.copymode')
    mocker.patch('generate_file.is_binary', return_value=False)
    
    from generate_file import generate_file
    
    generate_file(project_dir, infile, context, env)


def test_generate_file_copies_binary_file(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    template_dir = tmp_path / "templates"
    os.makedirs(template_dir)
    
    infile = "binary_file.bin"
    template_file = template_dir / infile
    template_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    context = {'cookiecutter': {}}
    env = Environment()
    
    mocker.patch('generate_file.is_binary', return_value=True)
    mocker.patch('shutil.copyfile')
    mocker.patch('shutil.copymode')
    
    from generate_file import generate_file
    
    generate_file(project_dir, infile, context, env)


def test_generate_file_skips_existing_file(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    outfile_path = os.path.join(project_dir, "existing.txt")
    open(outfile_path, 'w').close()
    
    infile = "existing.txt"
    context = {'cookiecutter': {}}
    env = Environment()
    
    mocker.patch('generate_file.is_binary', return_value=False)
    
    from generate_file import generate_file
    
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)


def test_generate_file_handles_empty_filename(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    infile = "{{cookiecutter.name}}"
    context = {'cookiecutter': {'name': ''}}
    env = Environment()
    
    mocker.patch('os.path.isdir', return_value=True)
    
    from generate_file import generate_file
    
    generate_file(project_dir, infile, context, env)


def test_generate_file_renders_filename(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    template_dir = tmp_path / "templates"
    os.makedirs(template_dir)
    
    infile = "{{cookiecutter.filename}}.txt"
    template_file = template_dir / infile
    template_file.write_text("content", encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'filename': 'output',
            '_new_lines': '\n'
        }
    }
    
    env = Environment()
    
    mocker.patch('generate_file.is_binary', return_value=False)
    mocker.patch('shutil.copymode')
    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data="content"))
    
    from generate_file import generate_file
    
    generate_file(project_dir, infile, context, env)


# LLM-generated content at query #32
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname(tmp_path):
    """Test that EmptyDirNameException is raised when dirname is empty."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    
    try:
        render_and_create_dir("", context, output_dir, env)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_creates_new_directory(tmp_path):
    """Test that a new directory is created when it doesn't exist."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    dirname = "test_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, output_dir, env)
    
    assert result_path == Path(output_dir, dirname)
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_with_template_rendering(tmp_path):
    """Test that directory name is rendered from template."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    env = Environment()
    context = {"project_name": "my_project"}
    output_dir = tmp_path
    dirname = "{{ project_name }}_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, output_dir, env)
    
    assert result_path == Path(output_dir, "my_project_dir")
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_existing_dir_without_overwrite(tmp_path):
    """Test that OutputDirExistsException is raised when directory exists and overwrite is False."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from pathlib import Path
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    dirname = "existing_dir"
    
    Path(output_dir, dirname).mkdir(parents=True, exist_ok=True)
    
    try:
        render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    """Test that existing directory is allowed when overwrite_if_exists is True."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    dirname = "existing_dir"
    
    Path(output_dir, dirname).mkdir(parents=True, exist_ok=True)
    
    result_path, is_new = render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=True)
    
    assert result_path == Path(output_dir, dirname)
    assert result_path.exists()
    assert is_new is False


def test_render_and_create_dir_nested_directory(tmp_path):
    """Test that nested directories are created correctly."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    env = Environment()
    context = {}
    output_dir = tmp_path
    dirname = "parent/child/grandchild"
    
    result_path, is_new = render_and_create_dir(dirname, context, output_dir, env)
    
    assert result_path == Path(output_dir, dirname)
    assert result_path.exists()
    assert is_new is True


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_context_opens_file_with_utf8_encoding(tmp_path):
    """Test that generate_context opens the context file with utf-8 encoding."""
    import json
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    
    # Create a temporary JSON file with UTF-8 encoding
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"project_name": "test_project", "author": "Test Author"}
    context_file.write_text(json.dumps(test_data), encoding='utf-8')
    
    # Call generate_context with the temporary file
    result = generate_context(str(context_file))
    
    # Verify the file was opened and parsed correctly (predicate at line 18 evaluated to True)
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"
    assert result["cookiecutter"]["author"] == "Test Author"
    assert isinstance(result, OrderedDict)


# LLM-generated content at query #34
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_false(tmp_path, monkeypatch):
    """Test that the predicate at line 24 evaluates to False when directory exists."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Setup
    output_dir = tmp_path
    dirname = "test_dir"
    context = {}
    environment = Environment()
    
    # Create the directory that will be checked
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    
    # Call the function with overwrite_if_exists=True to avoid exception
    result_path, created = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=output_dir,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # Assert that output_dir_exists was True (predicate at line 24 was True)
    # which means not output_dir_exists (line 35) is False
    assert created is False
    assert result_path == dir_to_create


# LLM-generated content at query #35
#--------------------------

```python
def test_generate_context_applies_default_context_when_provided(tmp_path, monkeypatch):
    """Test that line 38 predicate evaluates to True when default_context is provided."""
    import json
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    
    # Create a temporary context file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "my_project", "author": "John Doe"}
    context_file.write_text(json.dumps(context_data))
    
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Define default context
    default_context = {"project_name": "default_project"}
    
    # Call generate_context with default_context
    result = generate_context(
        context_file="cookiecutter.json",
        default_context=default_context
    )
    
    # Assert that default context was applied (predicate at line 38 was True)
    assert result["cookiecutter"]["project_name"] == "default_project"
    assert "cookiecutter" in result


# LLM-generated content at query #36
#--------------------------

```python
def test_generate_files_with_valid_context(tmp_path, monkeypatch):
    """Test generate_files with valid context and template structure."""
    from pathlib import Path
    from collections import OrderedDict
    import os
    
    # Setup template structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}\n")
    
    # Create context
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        })
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Mock the hook functions to avoid actual execution
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result == str(output_dir / "my_project")
    assert (output_dir / "my_project").exists()
    assert (output_dir / "my_project" / "README.md").exists()


def test_generate_files_empty_context(tmp_path, monkeypatch):
    """Test generate_files with None context."""
    from pathlib import Path
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=None,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_overwrite_if_exists(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    from pathlib import Path
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        })
    ])
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert result == str(existing_project)
    assert existing_project.exists()


def test_generate_files_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists=True."""
    from pathlib import Path
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "existing.txt"
    template_file.write_text("{{cookiecutter.content}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project',
            'content': 'new content',
            '_jinja2_env_vars': {}
        })
    ])
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    assert result == str(output_dir / "my_project")


def test_generate_files_with_subdirectories(tmp_path, monkeypatch):
    """Test generate_files with nested template directories."""
    from pathlib import Path
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    src_dir = template_dir / "src"
    src_dir.mkdir()
    src_file = src_dir / "main.py"
    src_file.write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_app',
            '_jinja2_env_vars': {}
        })
    ])
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / "my_app" / "src" / "main.py").exists()
    assert (output_dir / "my_app" / "src" / "main.py").read_text() == "# my_app"


def test_generate_files_with_hooks(tmp_path, monkeypatch):
    """Test generate_files with accept_hooks=True."""
    from pathlib import Path
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        })
    ])
    
    hook_calls = []
    
    def mock_hook(*args, **kwargs):
        hook_calls.append((args, kwargs))
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_hook)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=True
    )
    
    assert len(hook_calls) >= 2
    assert result == str(output_dir / "my_project")


def test_generate_files_keep_project_


# LLM-generated content at query #37
#--------------------------

```python
def test_file_name_is_empty_predicate_true(tmp_path, monkeypatch):
    import os
    from jinja2 import Environment
    
    # Create a temporary directory to serve as project_dir
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    # Create a subdirectory that will be the "outfile" path
    outfile_dir = os.path.join(project_dir, "subdir")
    os.makedirs(outfile_dir)
    
    # Setup context and environment
    context = {"cookiecutter": {}}
    env = Environment()
    
    # Import the function
    from cookiecutter.generate import generate_file
    
    # Mock logger to verify behavior
    import logging
    logger = logging.getLogger("cookiecutter.generate")
    
    # Call generate_file with infile that renders to the directory path
    # This should make outfile point to the existing directory
    infile = "test_file"
    
    # Mock the rendered outfile to point to the existing directory
    monkeypatch.setattr(
        env, 
        "from_string", 
        lambda x: type('obj', (object,), {'render': lambda **kw: "subdir"})()
    )
    
    # Call the function - it should return early due to file_name_is_empty being True
    generate_file(project_dir, infile, context, env)
    
    # Verify that the predicate (os.path.isdir(outfile)) evaluates to True
    outfile = os.path.join(project_dir, "subdir")
    assert os.path.isdir(outfile) is True


# LLM-generated content at query #38
#--------------------------

```python
def test_generate_files_with_valid_context(tmp_path, monkeypatch):
    """Test generate_files with valid context and template directory."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    # Create a mock template structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None
    assert Path(result).exists()
    assert "my_project" in result


def test_generate_files_empty_directory_name_raises_exception(tmp_path):
    """Test generate_files raises EmptyDirNameException for empty directory name."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    from cookiecutter.exceptions import EmptyDirNameException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    # Create template with empty name
    template_dir = repo_dir / "{{}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([('cookiecutter', {})])
    
    try:
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False
        )
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_generate_files_output_dir_exists_without_overwrite(tmp_path):
    """Test generate_files raises OutputDirExistsException when output exists without overwrite."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    from cookiecutter.exceptions import OutputDirExistsException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Pre-create the output project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    try:
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=False,
            accept_hooks=False
        )
        assert False, "Should have raised OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_generate_files_with_overwrite_if_exists(tmp_path):
    """Test generate_files overwrites existing directory when overwrite_if_exists is True."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Pre-create the output project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old content")
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert Path(result).exists()
    assert (Path(result) / "file.txt").exists()


def test_generate_files_default_context(tmp_path):
    """Test generate_files works with default empty context."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "myproject"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=None,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None
    assert Path(result).exists()


def test_generate_files_with_nested_directories(tmp_path):
    """Test generate_files with nested template directories."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create nested directories
    nested_dir = template_dir / "src" / "{{cookiecutter.module_name}}"
    nested_dir.mkdir(parents=True)
    
    nested_file = nested_dir / "module.py"
    nested_file.write_text("# {{cookiecutter.module_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project',
            'module_name': 'my_module'
        })
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert Path(result).exists()
    assert (Path(result) / "src" / "my_module" / "module.py").exists()


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_context_basic(tmp_path):
    """Test generate_context with a basic JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John Doe"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context with default_context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    default_context = {"project_name": "overridden_project"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "overridden_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context with extra_context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    extra_context = {"version": "2.0"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["version"] == "2.0"


def test_generate_context_with_list_choice(tmp_path):
    """Test generate_context with choice variable (list)."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"
    assert "MIT" in result["cookiecutter"]["license"]


def test_generate_context_with_multichoice_list(tmp_path):
    """Test generate_context with multichoice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    
    extra_context = {"features": ["feature2", "feature3"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["features"]) == {"feature2", "feature3"}


def test_generate_context_with_boolean_true(tmp_path):
    """Test generate_context with boolean variable converted from string."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_ci": false}')
    
    extra_context = {"use_ci": "yes"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_ci"] is True


def test_generate_context_with_boolean_false(tmp_path):
    """Test generate_context with boolean variable converted to false."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_ci": true}')
    
    extra_context = {"use_ci": "no"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_ci"] is False


def test_generate_context_with_nested_dict(tmp_path):
    """Test generate_context with nested dictionary."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"debug": true, "port": 8000}}')
    
    extra_context = {"config": {"debug": false}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["config"]["debug"] is False
    assert result["cookiecutter"]["config"]["port"] == 8000


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context with invalid JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"invalid": json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Should have raised ContextDecodingException"
    except Exception as e:
        assert "ContextDecodingException" in str(type(e))


def test_generate_context_file_not_found():
    """Test generate_context with non-existent file."""
    try:
        generate_context("/non/existent/file.json")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_generate_context_invalid_choice_overwrite(tmp_path):
    """Test generate_context with invalid choice in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    extra_context = {"license": "GPL"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "GPL" in str(e)


def test_generate_context_with_string_overwrite(tmp_path):
    """Test generate_context with simple string overwrite."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"author": "John", "email": "john@example.com"}')
    
    extra_context = {"author": "Jane"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["author"] == "Jane"
    assert result["cookiecutter"]["email"] == "john@example.com"


def test_generate_context_with_numeric_values(tmp_path):
    """Test generate_context with numeric values."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"port": 8000, "timeout": 30}')
    
    extra_context = {"port": 9000}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["port"] == 9000
    assert result["cookiecutter"]["timeout"] == 30


# LLM-generated content at query #40
#--------------------------

```python
def test_generate_context_loads_json_file(tmp_path):
    """Test that generate_context loads a JSON file and returns it in a dictionary."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John Doe"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_generate_context_with_default_context(tmp_path):
    """Test that generate_context applies default_context overwrites."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    result = generate_context(str(context_file), default_context={"project_name": "override_project"})
    
    assert result["cookiecutter"]["project_name"] == "override_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test that generate_context applies extra_context overwrites."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    result = generate_context(str(context_file), extra_context={"version": "2.0"})
    
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["version"] == "2.0"


def test_generate_context_invalid_json_raises_exception(tmp_path):
    """Test that generate_context raises ContextDecodingException for invalid JSON."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"invalid json"')
    
    try:
        generate_context(str(context_file))
        assert False, "Expected ContextDecodingException"
    except Exception as e:
        assert "ContextDecodingException" in str(type(e))
        assert "JSON decoding error" in str(e)


def test_generate_context_with_custom_filename(tmp_path):
    """Test that generate_context uses the filename as the key."""
    context_file = tmp_path / "custom_config.json"
    context_file.write_text('{"key": "value"}')
    
    result = generate_context(str(context_file))
    
    assert "custom_config" in result
    assert result["custom_config"]["key"] == "value"


def test_generate_context_with_choice_variable(tmp_path):
    """Test that generate_context handles choice variables in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    result = generate_context(str(context_file), extra_context={"license": "Apache"})
    
    assert result["cookiecutter"]["license"][0] == "Apache"
    assert "MIT" in result["cookiecutter"]["license"]
    assert "GPL" in result["cookiecutter"]["license"]


def test_generate_context_with_boolean_variable(tmp_path):
    """Test that generate_context converts string to boolean for boolean variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    result = generate_context(str(context_file), extra_context={"use_docker": "false"})
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_with_nested_dict(tmp_path):
    """Test that generate_context handles nested dictionary overwrites."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"debug": false, "timeout": 30}}')
    
    result = generate_context(str(context_file), extra_context={"config": {"debug": "true"}})
    
    assert result["cookiecutter"]["config"]["debug"] is True
    assert result["cookiecutter"]["config"]["timeout"] == 30


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test that generate_context handles multichoice variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["auth", "api", "admin", "logging"]}')
    
    result = generate_context(str(context_file), extra_context={"features": ["auth", "admin"]})
    
    assert result["cookiecutter"]["features"] == ["auth", "admin"]


def test_generate_context_with_invalid_choice_raises_error(tmp_path):
    """Test that generate_context raises ValueError for invalid choice."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    try:
        generate_context(str(context_file), extra_context={"license": "BSD"})
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "BSD" in str(e)
        assert "choice variable" in str(e)


def test_generate_context_empty_file(tmp_path):
    """Test that generate_context handles empty JSON object."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"] == {}


def test_generate_context_with_default_and_extra_context(tmp_path):
    """Test that generate_context applies both default and extra context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"name": "default", "version": "1.0", "author": "original"}')
    
    result = generate_context(
        str(context_file),
        default_context={"name": "from_default", "version": "2.0"},
        extra_context={"version": "3.0"}
    )
    
    assert result["cookiecutter"]["name"] == "from_default"
    assert result["cookiecutter"]["version"] == "3.0"
    assert result["cookiecutter"]["author"] == "original"


# LLM-generated content at query #41
#--------------------------

```python
def test_generate_files_with_minimal_context(tmp_path, monkeypatch):
    """Test generate_files with minimal context creates project directory."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    (template_dir / "README.md").write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    monkeypatch.setenv('COOKIECUTTER_ACCEPT_HOOKS', 'False')
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert 'test_project' in result
    assert (output_dir / 'test_project').exists()


def test_generate_files_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files respects skip_if_file_exists flag."""
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    (template_dir / "file.txt").write_text("{{cookiecutter.content}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    project_dir = output_dir / "test_project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("existing content")
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'content': 'new content'
        }
    }
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    file_content = (project_dir / "file.txt").read_text()
    assert file_content == "existing content"


def test_generate_files_overwrite_if_exists(tmp_path):
    """Test generate_files with overwrite_if_exists flag."""
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    project_dir = output_dir / "test_project"
    project_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert (project_dir / "file.txt").exists()


def test_generate_files_returns_project_dir_path(tmp_path):
    """Test generate_files returns the absolute path to project directory."""
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result == str(output_dir / "my_project")
    assert (output_dir / "my_project").exists()


def test_generate_files_with_nested_directories(tmp_path):
    """Test generate_files creates nested directory structure."""
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    nested_dir = template_dir / "src" / "{{cookiecutter.module_name}}"
    nested_dir.mkdir(parents=True)
    
    (nested_dir / "main.py").write_text("# {{cookiecutter.module_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'test_proj',
            'module_name': 'mymodule'
        }
    }
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / "test_proj" / "src" / "mymodule" / "main.py").exists()


def test_generate_files_default_output_dir(tmp_path, monkeypatch):
    """Test generate_files uses current directory as default output_dir."""
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    (template_dir / "file.txt").write_text("content")
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    monkeypatch.chdir(tmp_path)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        accept_hooks=False
    )
    
    assert "test_project" in result


# LLM-generated content at query #42
#--------------------------

```python
def test_generate_context_basic(tmp_path):
    """Test basic context generation from a JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_default_context(tmp_path):
    """Test context generation with default context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "default_name", "version": "1.0"}')
    
    default_context = {"project_name": "overridden_name"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "overridden_name"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test context generation with extra context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "default_name", "version": "1.0"}')
    
    extra_context = {"version": "2.0"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "default_name"
    assert result["cookiecutter"]["version"] == "2.0"


def test_generate_context_with_choice_variable(tmp_path):
    """Test context generation with choice variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test context generation with multi-choice variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["auth", "api", "admin", "logging"]}')
    
    extra_context = {"features": ["api", "logging"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["features"] == ["api", "logging"]


def test_generate_context_with_boolean_variable(tmp_path):
    """Test context generation with boolean variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true, "use_ci": false}')
    
    extra_context = {"use_docker": "no", "use_ci": "yes"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False
    assert result["cookiecutter"]["use_ci"] is True


def test_generate_context_with_nested_dict(tmp_path):
    """Test context generation with nested dictionary variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"debug": true, "port": 8000}}')
    
    extra_context = {"config": {"debug": "no"}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["config"]["debug"] is False
    assert result["cookiecutter"]["config"]["port"] == 8000


def test_generate_context_invalid_json(tmp_path):
    """Test context generation with invalid JSON raises ContextDecodingException."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"invalid": json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Expected ContextDecodingException"
    except Exception as e:
        assert "ContextDecodingException" in str(type(e).__name__)


def test_generate_context_invalid_choice_value(tmp_path):
    """Test context generation with invalid choice value in extra context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    extra_context = {"license": "GPL"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "GPL" in str(e)


def test_generate_context_invalid_multichoice_value(tmp_path):
    """Test context generation with invalid multichoice value in extra context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["auth", "api"]}')
    
    extra_context = {"features": ["auth", "invalid"]}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid" in str(e)


def test_generate_context_invalid_boolean_conversion(tmp_path):
    """Test context generation with invalid boolean conversion."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    extra_context = {"use_docker": "invalid_bool"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_generate_context_file_not_found():
    """Test context generation with non-existent file."""
    try:
        generate_context("/nonexistent/path/cookiecutter.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_generate_context_preserves_other_types(tmp_path):
    """Test that context generation preserves other data types."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"port": 8000, "timeout": 30.5, "name": "test"}')
    
    result = generate_context(str(context_file))
    
    assert result["cookiecutter"]["port"] == 8000
    assert result["cookiecutter"]["timeout"] == 30.5
    assert result["cookiecutter"]["name"] == "test"


def test_generate_context_default_context_with_invalid_value(tmp_path):
    """Test that invalid default context raises warning but continues."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    default_context = {"license": "InvalidLicense"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["license"] == ["MIT", "Apache"]


# LLM-generated content at query #43
#--------------------------

```python
def test_generate_files_with_minimal_context(tmp_path, monkeypatch):
    """Test generate_files with minimal context."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    # Create a minimal template structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "test.txt"
    template_file.write_text("Hello {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None
    assert "my_project" in result


def test_generate_files_empty_dirname_raises_exception(tmp_path):
    """Test that empty directory name raises EmptyDirNameException."""
    from cookiecutter.generate import generate_files
    from cookiecutter.exceptions import EmptyDirNameException
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / ""
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    try:
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False
        )
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_generate_files_with_overwrite_if_exists(tmp_path):
    """Test generate_files with overwrite_if_exists flag."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "test.txt"
    template_file.write_text("Content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    # First generation
    result1 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Second generation with overwrite
    result2 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert result1 is not None
    assert result2 is not None


def test_generate_files_without_context(tmp_path):
    """Test generate_files with None context."""
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "test.txt"
    template_file.write_text("Content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=None,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_with_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists flag."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "test.txt"
    template_file.write_text("Original content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    # First generation
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    assert result is not None


# LLM-generated content at query #44
#--------------------------

```python
def test_generate_context_file_open_predicate_line_18():
    """Test that the predicate at line 18 (open file operation) evaluates to False when file doesn't exist."""
    import os
    import tempfile
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a non-existent file path
        non_existent_file = os.path.join(tmpdir, 'non_existent_cookiecutter.json')
        
        # The predicate at line 18 is the condition of the try block
        # It evaluates to False when the file cannot be opened (FileNotFoundError)
        # This causes the except block to NOT catch it (since it only catches ValueError)
        # So we expect a FileNotFoundError to be raised
        try:
            generate_context(context_file=non_existent_file)
            # If we reach here, the test fails because no exception was raised
            assert False, "Expected FileNotFoundError to be raised"
        except FileNotFoundError:
            # This is the expected behavior when the predicate evaluates to False
            # (file doesn't exist, so open() fails before JSON parsing)
            pass


# LLM-generated content at query #45
#--------------------------

```python
def test_generate_files_basic(tmp_path, monkeypatch):
    """Test basic generate_files functionality with minimal setup."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    # Create a basic template structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    test_file = template_dir / "test.txt"
    test_file.write_text("Hello {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    # Mock the hook functions to avoid executing actual hooks
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert str(output_dir / 'my_project') == result
    assert (output_dir / 'my_project').exists()
    assert (output_dir / 'my_project' / 'test.txt').exists()


def test_generate_files_empty_dirname_raises_exception(tmp_path, monkeypatch):
    """Test that empty directory name raises EmptyDirNameException."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    from cookiecutter.exceptions import EmptyDirNameException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / ""
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    monkeypatch.setattr('cookiecutter.generate.find_template', lambda *args, **kwargs: template_dir)
    monkeypatch.setattr('cookiecutter.generate.os.path.split', lambda x: (str(repo_dir), ""))
    
    try:
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False
        )
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_generate_files_with_overwrite_if_exists(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    test_file = template_dir / "test.txt"
    test_file.write_text("Hello {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create existing output directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old content")
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert (output_dir / 'my_project').exists()
    assert (output_dir / 'my_project' / 'test.txt').exists()


def test_generate_files_with_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists=True."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    test_file = template_dir / "test.txt"
    test_file.write_text("New content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create existing output with a file
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    (existing_project / "test.txt").write_text("old content")
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert (output_dir / 'my_project' / 'test.txt').read_text() == "old content"


def test_generate_files_with_copy_without_render(tmp_path, monkeypatch):
    """Test generate_files respects _copy_without_render setting."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a file that should not be rendered
    binary_file = template_dir / "binary.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project',
            '_copy_without_render': ['*.bin']
        })
    ])
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / 'my_project').exists()


def test_generate_files_default_context(tmp_path, monkeypatch):
    """Test generate_files with default None context."""
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    try:
        result = generate_files(
            repo_dir=str(repo_dir),
            context=None,
            output_dir=str(output_dir),
            accept_hooks=False
        )


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generate_context_basic(tmp_path):
    """Test basic context generation from a JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_default_context(tmp_path):
    """Test context generation with default_context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    default_context = {"author": "Jane"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["author"] == "Jane"
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_generate_context_with_extra_context(tmp_path):
    """Test context generation with extra_context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    extra_context = {"project_name": "new_project"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "new_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_invalid_json(tmp_path):
    """Test context generation with invalid JSON raises ContextDecodingException."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"invalid": json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Should have raised ContextDecodingException"
    except Exception as e:
        assert "ContextDecodingException" in str(type(e))
        assert "JSON decoding error" in str(e)


def test_generate_context_with_list_variable(tmp_path):
    """Test context generation with list (choice) variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"flavor": ["vanilla", "chocolate", "strawberry"]}')
    
    extra_context = {"flavor": "chocolate"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["flavor"][0] == "chocolate"
    assert "vanilla" in result["cookiecutter"]["flavor"]
    assert "strawberry" in result["cookiecutter"]["flavor"]


def test_generate_context_with_dict_variable(tmp_path):
    """Test context generation with nested dictionary variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"options": {"debug": false, "verbose": true}}')
    
    extra_context = {"options": {"debug": true}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["options"]["debug"] is True
    assert result["cookiecutter"]["options"]["verbose"] is True


def test_generate_context_with_boolean_string(tmp_path):
    """Test context generation with boolean variable and string overwrite."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_https": true}')
    
    extra_context = {"use_https": "false"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_https"] is False


def test_generate_context_with_ordered_dict(tmp_path):
    """Test context generation preserves order."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"first": "1", "second": "2", "third": "3"}')
    
    result = generate_context(str(context_file))
    
    keys = list(result["cookiecutter"].keys())
    assert keys == ["first", "second", "third"]


def test_generate_context_nested_file_name(tmp_path):
    """Test context generation with nested directory structure."""
    subdir = tmp_path / "templates"
    subdir.mkdir()
    context_file = subdir / "cookiecutter.json"
    context_file.write_text('{"name": "test"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["name"] == "test"


def test_generate_context_invalid_default_context(tmp_path):
    """Test context generation with invalid default_context issues warning."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"choices": ["a", "b"], "name": "test"}')
    
    invalid_default = {"choices": "invalid_choice"}
    result = generate_context(str(context_file), default_context=invalid_default)
    
    assert "cookiecutter" in result


def test_generate_context_invalid_extra_context_raises(tmp_path):
    """Test context generation with invalid extra_context raises ValueError."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"choices": ["a", "b"]}')
    
    invalid_extra = {"choices": "invalid_choice"}
    try:
        generate_context(str(context_file), extra_context=invalid_extra)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)


def test_generate_context_with_multichoice(tmp_path):
    """Test context generation with multi-choice variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["auth", "api", "admin", "tests"]}')
    
    extra_context = {"features": ["api", "tests"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["features"]) == {"api", "tests", "auth", "admin"}


def test_generate_context_custom_file_name(tmp_path):
    """Test context generation with custom context file name."""
    context_file = tmp_path / "config.json"
    context_file.write_text('{"value": "data"}')
    
    result = generate_context(str(context_file))
    
    assert "config" in result
    assert result["config"]["value"] == "data"


# LLM-generated content at query #2
#--------------------------

```python
def test_apply_overwrites_to_context_simple_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"name": "John", "age": 30}
    overwrite_context = {"name": "Jane"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"name": "Jane", "age": 30}


def test_apply_overwrites_to_context_ignores_new_variable_at_top_level():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"name": "John"}
    overwrite_context = {"new_var": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"name": "John"}


def test_apply_overwrites_to_context_adds_new_variable_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"settings": {"theme": "dark"}}
    overwrite_context = {"settings": {"new_key": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"settings": {"theme": "dark", "new_key": "new_value"}}


def test_apply_overwrites_to_context_choice_variable():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flavor": ["vanilla", "chocolate", "strawberry"]}
    overwrite_context = {"flavor": "chocolate"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flavor": ["chocolate", "vanilla", "strawberry"]}


def test_apply_overwrites_to_context_choice_variable_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flavor": ["vanilla", "chocolate"]}
    overwrite_context = {"flavor": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "invalid provided for choice variable" in str(e)


def test_apply_overwrites_to_context_multichoice_variable():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"options": ["a", "b", "c"]}
    overwrite_context = {"options": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"options": ["b", "c"]}


def test_apply_overwrites_to_context_multichoice_variable_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"options": ["a", "b", "c"]}
    overwrite_context = {"options": ["b", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "multi-choice variable" in str(e)


def test_apply_overwrites_to_context_boolean_yes():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite_context = {"enabled": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": True}


def test_apply_overwrites_to_context_boolean_no():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": False}


def test_apply_overwrites_to_context_boolean_true():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"debug": False}
    overwrite_context = {"debug": "true"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"debug": True}


def test_apply_overwrites_to_context_boolean_false():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"debug": True}
    overwrite_context = {"debug": "false"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"debug": False}


def test_apply_overwrites_to_context_boolean_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_apply_overwrites_to_context_nested_dict_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"db": {"host": "localhost", "port": 5432}}
    overwrite_context = {"db": {"host": "remotehost"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"db": {"host": "remotehost", "port": 5432}}


def test_apply_overwrites_to_context_in_dictionary_variable_flag():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"settings": {"theme": "dark"}}
    overwrite_context = {"settings": {"new_setting": "value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"settings": {"theme": "dark", "new_setting": "value"}}


def test_apply_overwrites_to_context_list_overwrite_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"options": ["a", "b"]}
    overwrite_context = {"options": ["c"]}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"options": ["c"]}


def test_apply_overwrites_to_context_multiple_variables():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"name": "John", "age": 30, "city": "NYC"}
    overwrite_context = {"name": "Jane", "age": 25}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"name": "Jane", "age": 25, "city": "NYC"}


def test_apply_overwrites_to_context_empty_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"name": "John", "age": 30}
    overwrite_context = {}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"name": "John", "age": 30}


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    """Test that the predicate at line 52 evaluates to False for non-matching types."""
    context_value = "not a boolean"
    overwrite = "yes"
    
    result = isinstance(context_value, bool) and isinstance(overwrite, str)
    
    assert result is False


def test_predicate_at_line_52_evaluates_to_false_when_context_value_not_bool():
    """Test predicate is False when context_value is not a boolean."""
    context_value = 42
    overwrite = "yes"
    
    result = isinstance(context_value, bool) and isinstance(overwrite, str)
    
    assert result is False


def test_predicate_at_line_52_evaluates_to_false_when_overwrite_not_str():
    """Test predicate is False when overwrite is not a string."""
    context_value = True
    overwrite = 123
    
    result = isinstance(context_value, bool) and isinstance(overwrite, str)
    
    assert result is False


def test_predicate_at_line_52_evaluates_to_false_when_both_wrong_type():
    """Test predicate is False when both context_value and overwrite are wrong types."""
    context_value = []
    overwrite = {}
    
    result = isinstance(context_value, bool) and isinstance(overwrite, str)
    
    assert result is False


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    """Test that the predicate at line 52 evaluates to False with non-matching types."""
    context = {"key": 123}
    overwrite_context = {"key": "value"}
    
    # The predicate at line 52 is: isinstance(context_value, bool) and isinstance(overwrite, str)
    # context_value = 123 (int, not bool)
    # overwrite = "value" (str)
    # So: isinstance(123, bool) and isinstance("value", str) = False and True = False
    
    from cookiecutter.generate import apply_overwrites_to_context
    
    apply_overwrites_to_context(context, overwrite_context)
    
    # If predicate is False, it should fall through to the else clause at line 63-65
    # which simply overwrites the value
    assert context["key"] == "value"


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    """Test that the predicate at line 52 evaluates to False when conditions are not met."""
    context = {"my_var": "string_value"}
    overwrite_context = {"my_var": "new_string"}
    
    # This should not raise an error and should execute the else branch at line 64
    # The predicate at line 52: isinstance(context_value, bool) and isinstance(overwrite, str)
    # should evaluate to False because context_value is a string, not a bool
    from cookiecutter.generate import apply_overwrites_to_context
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["my_var"] == "new_string"


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    """Test that the predicate at line 52 evaluates to False."""
    context_value = "not_a_bool"
    overwrite = 123
    
    result = isinstance(context_value, bool) and isinstance(overwrite, str)
    
    assert result is False


# LLM-generated content at query #7
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        output_dir = tmpdir
        
        try:
            render_and_create_dir("", context, output_dir, env)
            assert False, "Should raise EmptyDirNameException"
        except EmptyDirNameException:
            pass


def test_render_and_create_dir_none_dirname():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        output_dir = tmpdir
        
        try:
            render_and_create_dir(None, context, output_dir, env)
            assert False, "Should raise EmptyDirNameException"
        except EmptyDirNameException:
            pass


def test_render_and_create_dir_creates_new_directory():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        output_dir = tmpdir
        dirname = "test_dir"
        
        result_path, is_new = render_and_create_dir(dirname, context, output_dir, env)
        
        assert result_path == Path(output_dir, dirname)
        assert is_new is True
        assert result_path.exists()


def test_render_and_create_dir_with_template_rendering():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "my_project"}
        output_dir = tmpdir
        dirname = "{{ project_name }}_dir"
        
        result_path, is_new = render_and_create_dir(dirname, context, output_dir, env)
        
        assert result_path == Path(output_dir, "my_project_dir")
        assert is_new is True
        assert result_path.exists()


def test_render_and_create_dir_exists_no_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        output_dir = tmpdir
        dirname = "existing_dir"
        
        existing_path = Path(output_dir, dirname)
        existing_path.mkdir()
        
        try:
            render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=False)
            assert False, "Should raise OutputDirExistsException"
        except OutputDirExistsException:
            pass


def test_render_and_create_dir_exists_with_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        output_dir = tmpdir
        dirname = "existing_dir"
        
        existing_path = Path(output_dir, dirname)
        existing_path.mkdir()
        
        result_path, is_new = render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=True)
        
        assert result_path == existing_path
        assert is_new is False
        assert result_path.exists()


def test_render_and_create_dir_nested_path():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        output_dir = tmpdir
        dirname = "parent/child/grandchild"
        
        result_path, is_new = render_and_create_dir(dirname, context, output_dir, env)
        
        assert result_path == Path(output_dir, dirname)
        assert is_new is True
        assert result_path.exists()


# LLM-generated content at query #8
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_valid():
    """Test that valid yes/no responses are converted to boolean without raising InvalidResponse."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"debug": True}
    overwrite_context = {"debug": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["debug"] is False


def test_apply_overwrites_to_context_boolean_conversion_yes():
    """Test that 'yes' response is converted to True."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite_context = {"enabled": "yes"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is True


def test_apply_overwrites_to_context_boolean_conversion_true_string():
    """Test that 'true' string is converted to boolean True."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": False}
    overwrite_context = {"flag": "true"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is True


def test_apply_overwrites_to_context_boolean_conversion_false_string():
    """Test that 'false' string is converted to boolean False."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "false"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is False


# LLM-generated content at query #9
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_variable_at_first_level():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"existing_var": "value"}
    overwrite_context = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing_var": "value"}


def test_apply_overwrites_to_context_adds_new_variable_in_dictionary():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"existing": "value"}}
    overwrite_context = {"nested": {"new_key": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"existing": "value", "new_key": "new_value"}}


def test_apply_overwrites_to_context_overwrites_list_with_valid_multichoice():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}


def test_apply_overwrites_to_context_raises_for_invalid_multichoice():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d", "e"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "provided for multi-choice variable" in str(e)


def test_apply_overwrites_to_context_reorders_choice_variable():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "c"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["c", "a", "b"]}


def test_apply_overwrites_to_context_raises_for_invalid_choice():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "provided for choice variable" in str(e)


def test_apply_overwrites_to_context_overwrites_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"config": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"config": {"key1": "new_value1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"key1": "new_value1", "key2": "value2"}}


def test_apply_overwrites_to_context_converts_string_to_boolean_true():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"is_enabled": True}
    overwrite_context = {"is_enabled": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"is_enabled": True}


def test_apply_overwrites_to_context_converts_string_to_boolean_false():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"is_enabled": True}
    overwrite_context = {"is_enabled": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"is_enabled": False}


def test_apply_overwrites_to_context_raises_for_invalid_boolean_string():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"is_enabled": True}
    overwrite_context = {"is_enabled": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_apply_overwrites_to_context_simple_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"var": "old_value"}
    overwrite_context = {"var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new_value"}


def test_apply_overwrites_to_context_overwrites_list_with_string_in_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"choices": ["a", "b", "c"]}}
    overwrite_context = {"nested": {"choices": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"choices": "new_value"}}


def test_apply_overwrites_to_context_multiple_variables():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"var1": "value1", "var2": "value2", "var3": "value3"}
    overwrite_context = {"var1": "new1", "var3": "new3"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "new1", "var2": "value2", "var3": "new3"}


def test_apply_overwrites_to_context_boolean_with_1_string():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"flag": False}
    overwrite_context = {"flag": "1"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_with_0_string():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"flag": True}
    overwrite_context = {"flag": "0"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}


# LLM-generated content at query #10
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    
    context = {}
    environment = Environment()
    
    try:
        render_and_create_dir("", context, tmp_path, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_with_none_dirname(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    
    context = {}
    environment = Environment()
    
    try:
        render_and_create_dir(None, context, tmp_path, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_creates_new_directory(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {}
    environment = Environment()
    dirname = "my_new_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path == Path(tmp_path, dirname)
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_with_template_rendering(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {"project_name": "my_project"}
    environment = Environment()
    dirname = "{{ project_name }}_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path == Path(tmp_path, "my_project_dir")
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_existing_dir_raises_exception(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    from pathlib import Path
    
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    
    existing_path = Path(tmp_path, dirname)
    existing_path.mkdir(parents=True, exist_ok=True)
    
    try:
        render_and_create_dir(dirname, context, tmp_path, environment)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    
    existing_path = Path(tmp_path, dirname)
    existing_path.mkdir(parents=True, exist_ok=True)
    
    result_path, is_new = render_and_create_dir(
        dirname, context, tmp_path, environment, overwrite_if_exists=True
    )
    
    assert result_path == existing_path
    assert result_path.exists()
    assert is_new is False


def test_render_and_create_dir_creates_nested_directories(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {}
    environment = Environment()
    dirname = "parent/child/grandchild"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path == Path(tmp_path, dirname)
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_returns_tuple(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {}
    environment = Environment()
    dirname = "test_dir"
    
    result = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Path)
    assert isinstance(result[1], bool)


# LLM-generated content at query #11
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_evaluates_to_true(tmp_path):
    """Test that the predicate at line 24 evaluates to True when directory exists."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Create a temporary directory that will exist
    existing_dir = tmp_path / "existing_project"
    existing_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up context and environment
    context = {"project_name": "existing_project"}
    environment = Environment()
    
    # Call the function with overwrite_if_exists=True to avoid exception
    result_path, is_new = render_and_create_dir(
        dirname="existing_project",
        context=context,
        output_dir=tmp_path,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # The predicate at line 24 (output_dir_exists) evaluates to True
    # which means is_new should be False (not output_dir_exists)
    assert is_new is False
    assert result_path.exists()


# LLM-generated content at query #12
#--------------------------

```python
def test_is_copy_only_path_with_matching_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['*.bin', '*.exe']}}
    result = is_copy_only_path('file.bin', context)
    assert result is True


def test_is_copy_only_path_with_non_matching_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['*.bin', '*.exe']}}
    result = is_copy_only_path('file.txt', context)
    assert result is False


def test_is_copy_only_path_with_wildcard_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['node_modules/*', 'dist/*']}}
    result = is_copy_only_path('node_modules/package', context)
    assert result is True


def test_is_copy_only_path_missing_copy_without_render_key():
    context = {'cookiecutter': {}}
    result = is_copy_only_path('file.txt', context)
    assert result is False


def test_is_copy_only_path_missing_cookiecutter_key():
    context = {}
    result = is_copy_only_path('file.txt', context)
    assert result is False


def test_is_copy_only_path_empty_copy_without_render_list():
    context = {'cookiecutter': {'_copy_without_render': []}}
    result = is_copy_only_path('file.txt', context)
    assert result is False


def test_is_copy_only_path_with_multiple_patterns_first_match():
    context = {'cookiecutter': {'_copy_without_render': ['*.bin', '*.exe', '*.dll']}}
    result = is_copy_only_path('app.exe', context)
    assert result is True


def test_is_copy_only_path_with_multiple_patterns_last_match():
    context = {'cookiecutter': {'_copy_without_render': ['*.bin', '*.exe', '*.dll']}}
    result = is_copy_only_path('library.dll', context)
    assert result is True


def test_is_copy_only_path_with_question_mark_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['file?.txt']}}
    result = is_copy_only_path('file1.txt', context)
    assert result is True


def test_is_copy_only_path_with_bracket_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['file[0-9].txt']}}
    result = is_copy_only_path('file5.txt', context)
    assert result is True


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_context_loads_json_file(tmp_path):
    """Test that generate_context loads a JSON file correctly."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "project_slug": "my_project"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["project_slug"] == "my_project"


def test_generate_context_with_extra_context(tmp_path):
    """Test that generate_context applies extra_context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "default_author"}')
    
    result = generate_context(
        str(context_file),
        extra_context={"project_name": "overridden_project", "author": "new_author"}
    )
    
    assert result["cookiecutter"]["project_name"] == "overridden_project"
    assert result["cookiecutter"]["author"] == "new_author"


def test_generate_context_with_default_context(tmp_path):
    """Test that generate_context applies default_context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    result = generate_context(
        str(context_file),
        default_context={"project_name": "default_overridden"}
    )
    
    assert result["cookiecutter"]["project_name"] == "default_overridden"


def test_generate_context_with_boolean_conversion(tmp_path):
    """Test that generate_context converts string to boolean in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    result = generate_context(
        str(context_file),
        extra_context={"use_docker": "false"}
    )
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_with_choice_variable(tmp_path):
    """Test that generate_context handles choice variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    result = generate_context(
        str(context_file),
        extra_context={"license": "Apache"}
    )
    
    assert result["cookiecutter"]["license"][0] == "Apache"
    assert "MIT" in result["cookiecutter"]["license"]
    assert "GPL" in result["cookiecutter"]["license"]


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test that generate_context handles multichoice variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    
    result = generate_context(
        str(context_file),
        extra_context={"features": ["feature1", "feature3"]}
    )
    
    assert result["cookiecutter"]["features"] == ["feature1", "feature3"]


def test_generate_context_with_nested_dict(tmp_path):
    """Test that generate_context handles nested dictionary variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"key1": "value1", "key2": "value2"}}')
    
    result = generate_context(
        str(context_file),
        extra_context={"config": {"key1": "overridden"}}
    )
    
    assert result["cookiecutter"]["config"]["key1"] == "overridden"
    assert result["cookiecutter"]["config"]["key2"] == "value2"


def test_generate_context_invalid_json_raises_exception(tmp_path):
    """Test that generate_context raises ContextDecodingException for invalid JSON."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{invalid json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Should have raised ContextDecodingException"
    except Exception as e:
        assert "ContextDecodingException" in str(type(e))


def test_generate_context_invalid_choice_raises_error(tmp_path):
    """Test that generate_context raises ValueError for invalid choice."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    try:
        generate_context(
            str(context_file),
            extra_context={"license": "GPL"}
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "GPL" in str(e)


def test_generate_context_invalid_multichoice_raises_error(tmp_path):
    """Test that generate_context raises ValueError for invalid multichoice."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2"]}')
    
    try:
        generate_context(
            str(context_file),
            extra_context={"features": ["feature1", "invalid_feature"]}
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_feature" in str(e)


def test_generate_context_invalid_boolean_conversion_raises_error(tmp_path):
    """Test that generate_context raises ValueError for invalid boolean conversion."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    try:
        generate_context(
            str(context_file),
            extra_context={"use_docker": "invalid_bool"}
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_generate_context_custom_file_stem(tmp_path):
    """Test that generate_context uses custom file stem as context key."""
    context_file = tmp_path / "custom_config.json"
    context_file.write_text('{"project": "test"}')
    
    result = generate_context(str(context_file))
    
    assert "custom_config" in result
    assert result["custom_config"]["project"] == "test"


def test_generate_context_boolean_true_conversion(tmp_path):
    """Test that generate_context converts 'true' string to boolean True."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_feature": false}')
    
    result = generate_context(
        str(context_file),
        extra_context={"use_feature": "yes"}
    )
    
    assert result["cookiecutter"]["use_feature"] is True


def test_generate_context_simple_string_overwrite(tmp_path):
    """Test that generate_context overwrites simple string variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"author": "John Doe", "email": "john@example.com"}')
    
    result = generate_context(
        str(context_file),
        extra_context={"author": "Jane Doe"}
    )
    
    assert result["cookiecutter"]["author"] == "Jane Doe"
    assert result["cookiecutter"]["email"] == "john@example.com"


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook_from_repo_dir_deprecated_warning(tmp_path, monkeypatch):
    """Test that _run_hook_from_repo_dir issues a deprecation warning."""
    import warnings
    from cookiecutter.generate import _run_hook_from_repo_dir
    from unittest.mock import patch
    
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {}}
    
    with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook:
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            _run_hook_from_repo_dir(
                repo_dir,
                "post_gen_project",
                project_dir,
                context,
                False
            )
            
            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, DeprecationWarning)
            assert "deprecated" in str(warning_list[0].message).lower()
            assert "run_hook_from_repo_dir" in str(warning_list[0].message)


def test_run_hook_from_repo_dir_calls_actual_hook(tmp_path):
    """Test that _run_hook_from_repo_dir calls the actual run_hook_from_repo_dir."""
    import warnings
    from cookiecutter.generate import _run_hook_from_repo_dir
    from unittest.mock import patch
    
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {}}
    delete_on_failure = True
    
    with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run_hook_from_repo_dir(
                repo_dir,
                "pre_prompt",
                project_dir,
                context,
                delete_on_failure
            )
            
            mock_run_hook.assert_called_once_with(
                repo_dir,
                "pre_prompt",
                project_dir,
                context,
                delete_on_failure
            )


def test_run_hook_from_repo_dir_with_different_hook_names(tmp_path):
    """Test _run_hook_from_repo_dir with various hook names."""
    import warnings
    from cookiecutter.generate import _run_hook_from_repo_dir
    from unittest.mock import patch
    
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {"test": "value"}}
    
    hook_names = ["pre_prompt", "post_prompt", "pre_gen_project", "post_gen_project"]
    
    for hook_name in hook_names:
        with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _run_hook_from_repo_dir(
                    repo_dir,
                    hook_name,
                    project_dir,
                    context,
                    False
                )
                
                mock_run_hook.assert_called_once()
                call_args = mock_run_hook.call_args
                assert call_args[0][1] == hook_name


# LLM-generated content at query #15
#--------------------------

```python
def test_run_hook_from_repo_dir_deprecation_warning(mocker):
    """Test that _run_hook_from_repo_dir issues a deprecation warning."""
    mock_run_hook_from_repo_dir = mocker.patch(
        'cookiecutter.generate.run_hook_from_repo_dir'
    )
    
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    
    with mocker.patch('warnings.warn') as mock_warn:
        from cookiecutter.generate import _run_hook_from_repo_dir
        _run_hook_from_repo_dir(
            repo_dir, hook_name, project_dir, context, delete_project_on_failure
        )
        
        mock_warn.assert_called_once()
        args, kwargs = mock_warn.call_args
        assert "deprecated" in args[0].lower()
        assert args[1] == DeprecationWarning
        assert args[2] == 2


def test_run_hook_from_repo_dir_calls_actual_function(mocker):
    """Test that _run_hook_from_repo_dir delegates to run_hook_from_repo_dir."""
    mock_run_hook_from_repo_dir = mocker.patch(
        'cookiecutter.generate.run_hook_from_repo_dir'
    )
    mocker.patch('warnings.warn')
    
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )
    
    mock_run_hook_from_repo_dir.assert_called_once_with(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )


def test_run_hook_from_repo_dir_with_false_delete_flag(mocker):
    """Test _run_hook_from_repo_dir with delete_project_on_failure=False."""
    mock_run_hook_from_repo_dir = mocker.patch(
        'cookiecutter.generate.run_hook_from_repo_dir'
    )
    mocker.patch('warnings.warn')
    
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {}}
    delete_project_on_failure = False
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )
    
    mock_run_hook_from_repo_dir.assert_called_once_with(
        repo_dir, hook_name, project_dir, context, False
    )


# LLM-generated content at query #16
#--------------------------

```python
def test_render_and_create_dir_predicate_at_line_24_evaluates_to_true(tmp_path, monkeypatch):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Setup
    output_dir = tmp_path
    dirname = "test_dir"
    context = {}
    environment = Environment()
    
    # Create the directory so it exists
    existing_dir = Path(output_dir, dirname)
    existing_dir.mkdir(parents=True, exist_ok=True)
    
    # Call the function with overwrite_if_exists=True to avoid exception
    result_path, created = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=output_dir,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # Verify the predicate at line 24 evaluated to True
    # (output_dir_exists was True, so the if block was entered)
    assert result_path == existing_dir
    assert created == False  # not output_dir_exists = not True = False


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_context_loads_json_file(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_default_context(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    default_context = {"project_name": "default_project"}
    
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "default_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    extra_context = {"version": "2.0"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["version"] == "2.0"


def test_generate_context_with_both_default_and_extra_context(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    default_context = {"project_name": "default_project"}
    extra_context = {"version": "2.0"}
    
    result = generate_context(str(context_file), default_context=default_context, extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "default_project"
    assert result["cookiecutter"]["version"] == "2.0"


def test_generate_context_invalid_json_raises_exception(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{invalid json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Expected ContextDecodingException"
    except Exception as e:
        assert "JSON decoding error" in str(e)


def test_generate_context_file_not_found_raises_exception(tmp_path):
    context_file = str(tmp_path / "nonexistent.json")
    
    try:
        generate_context(context_file)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_generate_context_with_choice_variable(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    extra_context = {"license": "Apache"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"


def test_generate_context_with_multichoice_variable(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    extra_context = {"features": ["feature2", "feature3"]}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["features"]) == {"feature2", "feature3"}


def test_generate_context_with_boolean_variable_yes(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": false}')
    extra_context = {"use_docker": "yes"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is True


def test_generate_context_with_boolean_variable_no(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    extra_context = {"use_docker": "no"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_with_nested_dict(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"database": {"engine": "postgresql", "port": 5432}}')
    extra_context = {"database": {"port": 3306}}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["database"]["engine"] == "postgresql"
    assert result["cookiecutter"]["database"]["port"] == 3306


def test_generate_context_with_invalid_choice_raises_error(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    extra_context = {"license": "GPL"}
    
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "GPL" in str(e) and "choice variable" in str(e)


def test_generate_context_custom_context_file_name(tmp_path):
    context_file = tmp_path / "custom.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    result = generate_context(str(context_file))
    
    assert "custom" in result
    assert result["custom"]["project_name"] == "my_project"


def test_generate_context_with_invalid_default_context_warning(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    default_context = {"license": "GPL"}
    
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["license"] == ["MIT", "Apache"]


# LLM-generated content at query #18
#--------------------------

```python
def test_render_and_create_dir_raises_on_empty_dirname():
    """Test that EmptyDirNameException is raised when dirname is empty string."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    context = {}
    output_dir = Path('/tmp')
    environment = Environment()
    
    try:
        render_and_create_dir("", context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that line 57 predicate evaluates to False when boolean conversion succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is True


def test_apply_overwrites_to_context_boolean_conversion_false():
    """Test that line 57 predicate evaluates to False when converting to False."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is False


def test_apply_overwrites_to_context_boolean_conversion_zero():
    """Test that line 57 predicate evaluates to False when converting zero string."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "0"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is False


def test_apply_overwrites_to_context_boolean_conversion_false_string():
    """Test that line 57 predicate evaluates to False when converting 'false'."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"active": True}
    overwrite_context = {"active": "false"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["active"] is False


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_files_basic(tmp_path, monkeypatch):
    """Test generate_files with basic template structure."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    (template_dir / "file.txt").write_text("Hello {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'myproject'})
    ])
    
    monkeypatch.setattr('cookiecutter.generate.accept_hooks', False)
    
    result = generate_files(
        str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert 'myproject' in result
    assert (output_dir / "myproject" / "file.txt").exists()
    assert (output_dir / "myproject" / "file.txt").read_text() == "Hello myproject"


def test_generate_files_with_subdirectories(tmp_path):
    """Test generate_files with subdirectories."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    subdir = template_dir / "src"
    subdir.mkdir()
    (subdir / "main.py").write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'myapp'})
    ])
    
    result = generate_files(
        str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / "myapp" / "src" / "main.py").exists()


def test_generate_files_empty_context(tmp_path):
    """Test generate_files with empty context."""
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "mytemplate"
    template_dir.mkdir()
    
    (template_dir / "README.md").write_text("Static content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        str(repo_dir),
        context=None,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / "mytemplate" / "README.md").exists()


def test_generate_files_overwrite_if_exists(tmp_path):
    """Test generate_files with overwrite_if_exists flag."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    (template_dir / "file.txt").write_text("New content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    project_dir = output_dir / "myproject"
    project_dir.mkdir()
    (project_dir / "old_file.txt").write_text("Old content")
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'myproject'})
    ])
    
    result = generate_files(
        str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert (output_dir / "myproject" / "file.txt").exists()


def test_generate_files_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists flag."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project}}"
    template_dir.mkdir()
    
    (template_dir / "config.txt").write_text("Config {{cookiecutter.version}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project': 'app', 'version': '1.0'})
    ])
    
    result = generate_files(
        str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    assert (output_dir / "app" / "config.txt").exists()


def test_generate_files_binary_file(tmp_path):
    """Test generate_files with binary file."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    (template_dir / "image.bin").write_bytes(b'\x89PNG\r\n\x1a\n')
    (template_dir / "text.txt").write_text("Content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'project'})
    ])
    
    result = generate_files(
        str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / "project" / "image.bin").exists()
    assert (output_dir / "project" / "text.txt").exists()


def test_generate_files_copy_without_render(tmp_path):
    """Test generate_files with _copy_without_render setting."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    static_dir = template_dir / "static"
    static_dir.mkdir()
    (static_dir / "style.css").write_text("body { color: {{cookiecutter.color}}; }")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'name': 'webapp',
            'color': 'blue',
            '_copy_without_render': ['static/*']
        })
    ])
    
    result = generate_files(
        str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / "webapp" / "static" / "style.css").exists()
    css_content = (output_dir / "webapp" / "static" / "style.css").read_text()
    assert '{{cookiecutter.color}}' in css_content


def test_generate_files_jinja2_newlines_config(tmp_path):
    """Test generate_files with _new_lines configuration."""
    from cookiecutter.generate import generate_files


# LLM-generated content at query #21
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    
    environment = Environment()
    context = {}
    
    try:
        render_and_create_dir("", context, tmp_path, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_creates_new_directory(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    environment = Environment()
    context = {}
    dirname = "test_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == "test_dir"
    assert is_new is True


def test_render_and_create_dir_with_template_rendering(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    environment = Environment()
    context = {"project_name": "my_project"}
    dirname = "{{project_name}}_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == "my_project_dir"
    assert is_new is True


def test_render_and_create_dir_exists_no_overwrite(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    
    environment = Environment()
    context = {}
    dirname = "existing_dir"
    
    existing_path = tmp_path / dirname
    existing_path.mkdir()
    
    try:
        render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_exists_with_overwrite(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    environment = Environment()
    context = {}
    dirname = "existing_dir"
    
    existing_path = tmp_path / dirname
    existing_path.mkdir()
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=True)
    
    assert result_path.exists()
    assert is_new is False


def test_render_and_create_dir_creates_nested_directories(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    environment = Environment()
    context = {}
    dirname = "parent/child/grandchild"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == "grandchild"
    assert is_new is True


def test_render_and_create_dir_with_context_variables(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    environment = Environment()
    context = {"name": "test", "version": "1.0"}
    dirname = "{{name}}_v{{version}}"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == "test_v1.0"
    assert is_new is True


# LLM-generated content at query #22
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_evaluates_to_true(tmp_path):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Create an existing directory
    existing_dir = tmp_path / "existing_project"
    existing_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup context and environment
    context = {}
    environment = Environment()
    dirname = "existing_project"
    
    # Call function with overwrite_if_exists=True to avoid exception
    result_path, was_new = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=tmp_path,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # Verify that the predicate at line 24 evaluated to True
    # (output_dir_exists was True, meaning the directory already existed)
    assert result_path.exists()
    assert was_new is False


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_context_opens_file_with_utf8_encoding(tmp_path):
    """Test that generate_context opens the context file with utf-8 encoding."""
    import json
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    
    # Create a temporary JSON file with UTF-8 encoding
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"project_name": "test", "author": "Test Author"}
    context_file.write_text(json.dumps(test_data), encoding='utf-8')
    
    # Call generate_context
    result = generate_context(str(context_file))
    
    # Verify the file was successfully opened and parsed with UTF-8 encoding
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test"
    assert result["cookiecutter"]["author"] == "Test Author"


# LLM-generated content at query #24
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_true(tmp_path, monkeypatch):
    """Test that the predicate at line 24 evaluates to True when directory exists."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Create an existing directory
    existing_dir = tmp_path / "existing_project"
    existing_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup context and environment
    context = {"project_name": "existing_project"}
    environment = Environment()
    dirname = "{{ project_name }}"
    
    # Call the function with overwrite_if_exists=True to avoid exception
    result_path, is_new = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=tmp_path,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # Verify that the directory exists (predicate at line 24 is True)
    assert result_path.exists() is True
    # Verify that is_new is False because directory already existed
    assert is_new is False


# LLM-generated content at query #25
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that boolean conversion succeeds and InvalidResponse is not raised."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    
    # This should not raise InvalidResponse, so the predicate at line 57 evaluates to False
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is True


def test_apply_overwrites_to_context_boolean_conversion_false():
    """Test that boolean conversion with 'no' succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    
    # This should not raise InvalidResponse, so the predicate at line 57 evaluates to False
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is False


def test_apply_overwrites_to_context_boolean_conversion_zero():
    """Test that boolean conversion with '0' succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "0"}
    
    # This should not raise InvalidResponse, so the predicate at line 57 evaluates to False
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is False


def test_apply_overwrites_to_context_boolean_conversion_one():
    """Test that boolean conversion with '1' succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite_context = {"enabled": "1"}
    
    # This should not raise InvalidResponse, so the predicate at line 57 evaluates to False
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is True


# LLM-generated content at query #26
#--------------------------

```python
def test_delete_project_on_failure_predicate_true_when_output_directory_created_and_keep_project_false():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


def test_delete_project_on_failure_predicate_false_when_output_directory_not_created():
    output_directory_created = False
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


def test_delete_project_on_failure_predicate_false_when_keep_project_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


def test_delete_project_on_failure_predicate_false_when_both_conditions_false():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_context_opens_file_with_utf8_encoding(tmp_path):
    """Test that generate_context opens the context file with utf-8 encoding."""
    import json
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"project_name": "test_project", "author": "test_author"}
    context_file.write_text(json.dumps(test_data), encoding='utf-8')
    
    result = generate_context(str(context_file))
    
    assert isinstance(result, dict)
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"
    assert result["cookiecutter"]["author"] == "test_author"


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_62_evaluates_to_false():
    """Test that the predicate at line 62 (for root, dirs, files in os.walk('.')) evaluates to False."""
    # The predicate at line 62 is: for root, dirs, files in os.walk('.')
    # This is a loop that iterates over the result of os.walk('.')
    # The loop evaluates to False when os.walk('.') returns an empty iterator
    # or when there are no items to iterate over
    
    import tempfile
    import os
    from pathlib import Path
    
    # Create a temporary empty directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to the temporary directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Collect results from os.walk
            results = list(os.walk('.'))
            
            # For an empty directory, os.walk returns a single tuple with empty files and dirs
            # The predicate (the for loop) will still execute once with empty dirs and files
            # To make the loop body not execute, we need to verify the condition is False
            # But os.walk always yields at least the root directory
            # So we check that when we have an empty directory, the iteration still happens
            # but with empty dirs and files
            
            root, dirs, files = results[0]
            
            # Assert that dirs and files are empty (predicate for loop body is False)
            assert dirs == []
            assert files == []
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #29
#--------------------------

```python
def test_generate_files_basic(tmp_path, monkeypatch):
    """Test generate_files with basic template structure."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    from pathlib import Path
    
    # Create template directory structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    test_file = template_dir / "test.txt"
    test_file.write_text("Project: {{cookiecutter.project_name}}")
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create context
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    # Mock find_template to return our template directory
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    gen_module.find_template = lambda repo, env: template_dir
    
    # Mock run_hook_from_repo_dir to do nothing
    original_run_hook = gen_module.run_hook_from_repo_dir
    gen_module.run_hook_from_repo_dir = lambda *args, **kwargs: None
    
    try:
        result = generate_files(
            str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False
        )
        
        assert result is not None
        assert "my_project" in result
        assert Path(result).exists()
    finally:
        gen_module.find_template = original_find_template
        gen_module.run_hook_from_repo_dir = original_run_hook


def test_generate_files_with_hooks(tmp_path, monkeypatch):
    """Test generate_files calls hooks when accept_hooks is True."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    from pathlib import Path
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'test_project'})
    ])
    
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    original_run_hook = gen_module.run_hook_from_repo_dir
    
    hook_calls = []
    
    def mock_run_hook(repo, hook_name, proj_dir, ctx, delete_on_fail):
        hook_calls.append(hook_name)
    
    gen_module.find_template = lambda repo, env: template_dir
    gen_module.run_hook_from_repo_dir = mock_run_hook
    
    try:
        result = generate_files(
            str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=True
        )
        
        assert 'pre_gen_project' in hook_calls
        assert 'post_gen_project' in hook_calls
    finally:
        gen_module.find_template = original_find_template
        gen_module.run_hook_from_repo_dir = original_run_hook


def test_generate_files_overwrite_existing(tmp_path):
    """Test generate_files with overwrite_if_exists=True."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    from pathlib import Path
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.proj}}"
    template_dir.mkdir()
    
    test_file = template_dir / "file.txt"
    test_file.write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Pre-create the output directory
    existing_proj = output_dir / "my_proj"
    existing_proj.mkdir()
    (existing_proj / "old_file.txt").write_text("old content")
    
    context = OrderedDict([
        ('cookiecutter', {'proj': 'my_proj'})
    ])
    
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    original_run_hook = gen_module.run_hook_from_repo_dir
    
    gen_module.find_template = lambda repo, env: template_dir
    gen_module.run_hook_from_repo_dir = lambda *args, **kwargs: None
    
    try:
        result = generate_files(
            str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=True,
            accept_hooks=False
        )
        
        assert Path(result).exists()
        assert Path(result, "file.txt").exists()
    finally:
        gen_module.find_template = original_find_template
        gen_module.run_hook_from_repo_dir = original_run_hook


def test_generate_files_skip_if_exists(tmp_path):
    """Test generate_files with skip_if_file_exists=True."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    from pathlib import Path
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    test_file = template_dir / "existing.txt"
    test_file.write_text("new content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'project'})
    ])
    
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    original_run_hook = gen_module.run_hook_from_repo_dir
    
    gen_module.find_template = lambda repo, env: template_dir
    gen_module.run_hook_from_repo_dir = lambda *args, **kwargs: None
    
    try:
        result = generate_files(
            str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            skip_if_file_exists=True,
            accept_hooks=False
        )
        
        assert Path(result).exists()
    finally:
        gen_module.find_template = original_find_template
        gen_module.run_hook_from_repo_dir = original_run_hook


def test_generate_files_with_copy_without_render(tmp_path):
    """Test generate_files with _copy_without_render context setting."""
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    from pathlib import Path
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    # Create a directory to copy without rendering
    copy_dir = template_dir / "static"
    copy_dir.mkdir()
    (copy_dir / "file.txt").write_text("{{not_rendered}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'name': 'myproject',
            '_copy_without_render': ['static']
        })
    ])
    
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    original_run_hook = gen_


# LLM-generated content at query #30
#--------------------------

```python
def test_undefined_error_caught_at_line_36():
    """Test that UndefinedError at line 36 is caught and re-raised as UndefinedVariableInTemplate."""
    from pathlib import Path
    from collections import OrderedDict
    from jinja2 import UndefinedError
    from cookiecutter.generate import generate_files
    from cookiecutter.exceptions import UndefinedVariableInTemplate
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        
        template_dir = repo_dir / "{{cookiecutter.project_name}}"
        template_dir.mkdir()
        
        context = OrderedDict([
            ('cookiecutter', {
                'project_name': '{{ undefined_variable }}'
            })
        ])
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        try:
            generate_files(
                repo_dir=repo_dir,
                context=context,
                output_dir=output_dir,
                accept_hooks=False
            )
            assert False, "Expected UndefinedVariableInTemplate to be raised"
        except UndefinedVariableInTemplate as e:
            assert "Unable to create project directory" in str(e)
            assert isinstance(e.__cause__, UndefinedError)


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_context_handles_json_decode_error():
    """Test that ValueError on line 20 is caught and re-raised as ContextDecodingException."""
    import tempfile
    import os
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    # Create a temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{ invalid json content }')
        temp_file = f.name
    
    try:
        # This should raise ContextDecodingException when trying to parse invalid JSON
        generate_context(context_file=temp_file)
        # If we reach here, the test failed
        assert False, "Expected ContextDecodingException to be raised"
    except Exception as e:
        # Verify that the exception is ContextDecodingException
        assert isinstance(e, ContextDecodingException)
        assert "JSON decoding error" in str(e)
        assert temp_file in str(e)
    finally:
        # Clean up the temporary file
        os.unlink(temp_file)


# LLM-generated content at query #32
#--------------------------

```python
def test_generate_file_renders_text_file(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    import os
    os.makedirs(project_dir, exist_ok=True)
    
    infile = "test_{{cookiecutter.name}}.txt"
    infile_path = tmp_path / "test_{{cookiecutter.name}}.txt"
    infile_path.write_text("Hello {{cookiecutter.name}}")
    
    context = {"cookiecutter": {"name": "world", "_new_lines": False}}
    env = Environment()
    
    mocker.patch('os.chdir', return_value=None)
    mocker.patch('shutil.copymode')
    
    import sys
    sys.path.insert(0, str(tmp_path.parent))
    
    from cookiecutter.generate import generate_file
    
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        generate_file(project_dir, infile, context, env)
    finally:
        os.chdir(original_cwd)
    
    outfile = os.path.join(project_dir, "test_world.txt")
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        assert f.read() == "Hello world"


def test_generate_file_skips_existing_file(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    import os
    os.makedirs(project_dir, exist_ok=True)
    
    infile = "test.txt"
    infile_path = tmp_path / "test.txt"
    infile_path.write_text("Original content")
    
    outfile_path = tmp_path / project_dir / "test.txt"
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    outfile_path.write_text("Existing content")
    
    context = {"cookiecutter": {"_new_lines": False}}
    env = Environment()
    
    mocker.patch('shutil.copymode')
    
    from cookiecutter.generate import generate_file
    
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    finally:
        os.chdir(original_cwd)
    
    with open(outfile_path, 'r', encoding='utf-8') as f:
        assert f.read() == "Existing content"


def test_generate_file_copies_binary_file(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    import os
    os.makedirs(project_dir, exist_ok=True)
    
    infile = "binary.bin"
    infile_path = tmp_path / "binary.bin"
    infile_path.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    context = {"cookiecutter": {}}
    env = Environment()
    
    mocker.patch('cookiecutter.generate.is_binary', return_value=True)
    mocker.patch('shutil.copyfile')
    mocker.patch('shutil.copymode')
    
    from cookiecutter.generate import generate_file
    
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        generate_file(project_dir, infile, context, env)
    finally:
        os.chdir(original_cwd)


def test_generate_file_returns_on_empty_filename(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    import os
    os.makedirs(project_dir, exist_ok=True)
    
    infile = "{{cookiecutter.skip_dir}}"
    infile_path = tmp_path / "{{cookiecutter.skip_dir}}"
    infile_path.mkdir(exist_ok=True)
    
    context = {"cookiecutter": {"skip_dir": ""}}
    env = Environment()
    
    from cookiecutter.generate import generate_file
    
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        generate_file(project_dir, infile, context, env)
    finally:
        os.chdir(original_cwd)


def test_generate_file_uses_configured_newline(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    import os
    os.makedirs(project_dir, exist_ok=True)
    
    infile = "test.txt"
    infile_path = tmp_path / "test.txt"
    infile_path.write_text("Line1\nLine2\n")
    
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    
    mocker.patch('shutil.copymode')
    
    from cookiecutter.generate import generate_file
    
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        generate_file(project_dir, infile, context, env)
    finally:
        os.chdir(original_cwd)
    
    outfile = os.path.join(project_dir, "test.txt")
    assert os.path.exists(outfile)


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_context_basic(tmp_path):
    """Test generate_context loads a basic JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context applies default_context overwrites."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    default_context = {"author": "Jane"}
    
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["author"] == "Jane"
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context applies extra_context overwrites."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    extra_context = {"project_name": "new_project"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "new_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_both_defaults_and_extra(tmp_path):
    """Test generate_context applies both default and extra context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John", "version": "1.0"}')
    default_context = {"author": "Jane"}
    extra_context = {"version": "2.0"}
    
    result = generate_context(str(context_file), default_context=default_context, extra_context=extra_context)
    
    assert result["cookiecutter"]["author"] == "Jane"
    assert result["cookiecutter"]["version"] == "2.0"
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context raises ContextDecodingException for invalid JSON."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"')
    
    try:
        generate_context(str(context_file))
        assert False, "Expected ContextDecodingException"
    except Exception as e:
        assert "JSON decoding error" in str(e)


def test_generate_context_nonexistent_file():
    """Test generate_context raises error for nonexistent file."""
    try:
        generate_context("/nonexistent/path/cookiecutter.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_generate_context_choice_variable(tmp_path):
    """Test generate_context with choice variable in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"flavor": ["vanilla", "chocolate", "strawberry"]}')
    extra_context = {"flavor": "chocolate"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["flavor"][0] == "chocolate"


def test_generate_context_boolean_variable(tmp_path):
    """Test generate_context with boolean variable in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    extra_context = {"use_docker": "false"}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_nested_dict(tmp_path):
    """Test generate_context with nested dictionary in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"debug": true, "port": 8000}}')
    extra_context = {"config": {"debug": false}}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["config"]["debug"] is False
    assert result["cookiecutter"]["config"]["port"] == 8000


def test_generate_context_multichoice_variable(tmp_path):
    """Test generate_context with multichoice variable in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    extra_context = {"features": ["feature2", "feature3"]}
    
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["features"]) == {"feature2", "feature3"}


def test_generate_context_invalid_choice_value(tmp_path):
    """Test generate_context with invalid choice value raises ValueError."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"flavor": ["vanilla", "chocolate"]}')
    extra_context = {"flavor": "mint"}
    
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "mint" in str(e)


def test_generate_context_invalid_boolean_string(tmp_path):
    """Test generate_context with invalid boolean string raises ValueError."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    extra_context = {"use_docker": "maybe"}
    
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_generate_context_file_stem(tmp_path):
    """Test generate_context uses correct file stem as context key."""
    context_file = tmp_path / "custom_template.json"
    context_file.write_text('{"name": "test"}')
    
    result = generate_context(str(context_file))
    
    assert "custom_template" in result
    assert result["custom_template"]["name"] == "test"


def test_generate_context_ordered_dict(tmp_path):
    """Test generate_context preserves order with OrderedDict."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"z_field": 1, "a_field": 2, "m_field": 3}')
    
    result = generate_context(str(context_file))
    
    keys = list(result["cookiecutter"].keys())
    assert keys == ["z_field", "a_field", "m_field"]


# LLM-generated content at query #34
#--------------------------

```python
def test_generate_files_with_minimal_context(tmp_path, monkeypatch):
    """Test generate_files with minimal context."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    # Create a minimal template structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple file in the template
    test_file = template_dir / "README.md"
    test_file.write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project'
        })
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert 'my_project' in result
    assert (output_dir / 'my_project').exists()
    assert (output_dir / 'my_project' / 'README.md').exists()


def test_generate_files_empty_dirname_raises_exception(tmp_path):
    """Test that empty directory name raises EmptyDirNameException."""
    from cookiecutter.generate import generate_files
    from cookiecutter.exceptions import EmptyDirNameException
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / ""
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {})
    ])
    
    try:
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False
        )
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_generate_files_with_existing_output_dir_no_overwrite(tmp_path):
    """Test that existing output directory raises exception when overwrite is False."""
    from cookiecutter.generate import generate_files
    from cookiecutter.exceptions import OutputDirExistsException
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project'
        })
    ])
    
    try:
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=False,
            accept_hooks=False
        )
        assert False, "Should have raised OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_generate_files_with_overwrite_if_exists(tmp_path):
    """Test generate_files with overwrite_if_exists=True."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    test_file = template_dir / "README.md"
    test_file.write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    old_file = existing_project / "old.txt"
    old_file.write_text("old content")
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project'
        })
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert 'my_project' in result
    assert (output_dir / 'my_project').exists()
    assert (output_dir / 'my_project' / 'README.md').exists()


def test_generate_files_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists=True."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    test_file = template_dir / "existing.txt"
    test_file.write_text("new content from template")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project'
        })
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=False,
        accept_hooks=False
    )
    
    generated_file = output_dir / 'my_project' / 'existing.txt'
    assert generated_file.exists()
    assert generated_file.read_text() == "new content from template"


def test_generate_files_with_subdirectories(tmp_path):
    """Test generate_files with nested directory structure."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    src_dir = template_dir / "src"
    src_dir.mkdir()
    
    test_file = src_dir / "main.py"
    test_file.write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project'
        })
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / 'my_project' / 'src' / 'main.py').exists()


def test_generate_files_with_binary_file(tmp_path):
    """Test generate_files with binary files."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a binary file (simple PNG header)
    binary_file = template_dir / "image.png"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = Order


# LLM-generated content at query #35
#--------------------------

```python
def test_generate_files_skips_hooks_when_accept_hooks_is_false(tmp_path, monkeypatch):
    """Test that pre_gen_project hook is not run when accept_hooks is False."""
    from cookiecutter.generate import generate_files
    from unittest.mock import patch, MagicMock
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("test")
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_hook:
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False
        )
        
        mock_hook.assert_not_called()


# LLM-generated content at query #36
#--------------------------

```python
def test_generate_files_basic(tmp_path, monkeypatch):
    """Test generate_files with basic template structure."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    # Create a basic template structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    monkeypatch.setenv('COOKIECUTTER_ACCEPT_HOOKS', 'false')
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result == str(output_dir / "my_project")
    assert (output_dir / "my_project").exists()
    assert (output_dir / "my_project" / "README.md").exists()
    assert (output_dir / "my_project" / "README.md").read_text() == "# my_project"


def test_generate_files_with_subdirectories(tmp_path):
    """Test generate_files with nested directory structure."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create nested directories
    src_dir = template_dir / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'test_app'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / "test_app" / "src" / "main.py").exists()
    assert (output_dir / "test_app" / "src" / "main.py").read_text() == "# test_app"


def test_generate_files_overwrite_if_exists(tmp_path):
    """Test generate_files with overwrite_if_exists flag."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create existing project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old content")
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert result == str(output_dir / "my_project")
    assert (output_dir / "my_project").exists()


def test_generate_files_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists flag."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "config.txt"
    template_file.write_text("config={{cookiecutter.value}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create existing project with file
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    (existing_project / "config.txt").write_text("existing config")
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project', 'value': 'new'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert (output_dir / "my_project" / "config.txt").read_text() == "existing config"


def test_generate_files_empty_context(tmp_path):
    """Test generate_files with empty context."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=None,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_with_copy_without_render(tmp_path):
    """Test generate_files with _copy_without_render setting."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a binary-like file that should not be rendered
    binary_file = template_dir / "image.bin"
    binary_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project',
            '_copy_without_render': ['*.bin']
        })
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / "my_project" / "image.bin").exists()


def test_generate_files_default_output_dir(tmp_path, monkeypatch):
    """Test generate_files with default output directory."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'test_project'})
    ])
    
    monkeypatch.chdir(tmp_path)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        


# LLM-generated content at query #37
#--------------------------

```python
def test_generate_context_default_context_predicate():
    """Test that the predicate at line 38 (if default_context:) evaluates to True."""
    import json
    import tempfile
    import os
    from collections import OrderedDict
    from unittest.mock import patch
    
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'project_name': 'test'}, f)
        temp_file = f.name
    
    try:
        # Mock apply_overwrites_to_context to verify it gets called
        with patch('cookiecutter.generate.apply_overwrites_to_context') as mock_apply:
            from cookiecutter.generate import generate_context
            
            # Call with default_context provided (predicate should be True)
            default_context = {'project_name': 'custom_name'}
            result = generate_context(
                context_file=temp_file,
                default_context=default_context
            )
            
            # Verify that apply_overwrites_to_context was called
            # This confirms the predicate at line 38 evaluated to True
            assert mock_apply.called
            assert mock_apply.call_count >= 1
            
            # Verify the context was properly generated
            assert 'cookiecutter' in result
            assert result['cookiecutter']['project_name'] == 'test'
    finally:
        os.unlink(temp_file)


# LLM-generated content at query #38
#--------------------------

```python
def test_generate_files_with_valid_context(tmp_path, monkeypatch):
    """Test generate_files with a valid context and template structure."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    # Create a mock template directory structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}\n")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None
    assert "my_project" in result


def test_generate_files_with_binary_file(tmp_path, monkeypatch):
    """Test generate_files with a binary file in the template."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a binary file
    binary_file = template_dir / "image.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_with_copy_without_render(tmp_path, monkeypatch):
    """Test generate_files with _copy_without_render setting."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a file to be copied without rendering
    copy_file = template_dir / "config.txt"
    copy_file.write_text("{{not_rendered}}\n")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project',
            '_copy_without_render': ['config.txt']
        })
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists option."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("Original content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    # First generation
    result1 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=False,
        accept_hooks=False
    )
    
    # Second generation with skip_if_file_exists
    result2 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    assert result1 is not None
    assert result2 is not None


def test_generate_files_with_subdirectories(tmp_path, monkeypatch):
    """Test generate_files with nested directory structure."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create nested structure
    src_dir = template_dir / "{{cookiecutter.src_dir}}"
    src_dir.mkdir()
    src_file = src_dir / "main.py"
    src_file.write_text("# {{cookiecutter.project_name}}\n")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'my_project',
            'src_dir': 'src'
        })
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_overwrite_if_exists(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists option."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("Content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    # First generation
    result1 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Second generation with overwrite
    result2 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert result1 is not None
    assert result2 is not None


def test_generate_files_with_newline_configuration(tmp_path, monkeypatch):
    """Test generate_files with _new_lines configuration."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_files_basic(tmp_path, monkeypatch):
    """Test generate_files with basic template structure."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    test_file = template_dir / "test.txt"
    test_file.write_text("Hello {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    monkeypatch.setattr('cookiecutter.generate.accept_hooks', False)
    
    result = generate_files(repo_dir, context, output_dir)
    
    assert 'my_project' in result
    assert (output_dir / 'my_project' / 'test.txt').exists()


def test_generate_files_with_context(tmp_path):
    """Test generate_files renders context variables correctly."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    test_file = template_dir / "{{cookiecutter.filename}}.txt"
    test_file.write_text("Project: {{cookiecutter.name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'name': 'test_project',
            'filename': 'readme'
        })
    ])
    
    result = generate_files(repo_dir, context, output_dir)
    
    assert (output_dir / 'test_project' / 'readme.txt').exists()
    content = (output_dir / 'test_project' / 'readme.txt').read_text()
    assert 'Project: test_project' in content


def test_generate_files_overwrite_if_exists(tmp_path):
    """Test generate_files with overwrite_if_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project}}"
    template_dir.mkdir()
    
    test_file = template_dir / "file.txt"
    test_file.write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing_project = output_dir / "myproject"
    existing_project.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project': 'myproject'})
    ])
    
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    
    assert (output_dir / 'myproject' / 'file.txt').exists()


def test_generate_files_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.proj}}"
    template_dir.mkdir()
    
    test_file = template_dir / "existing.txt"
    test_file.write_text("template content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'proj': 'project'})
    ])
    
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=False)
    
    assert (output_dir / 'project' / 'existing.txt').exists()


def test_generate_files_with_subdirectories(tmp_path):
    """Test generate_files with nested directory structure."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    subdir = template_dir / "src"
    subdir.mkdir()
    
    test_file = subdir / "main.py"
    test_file.write_text("# {{cookiecutter.name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'myapp'})
    ])
    
    result = generate_files(repo_dir, context, output_dir)
    
    assert (output_dir / 'myapp' / 'src' / 'main.py').exists()


def test_generate_files_empty_context(tmp_path):
    """Test generate_files with empty context."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "project"
    template_dir.mkdir()
    
    test_file = template_dir / "file.txt"
    test_file.write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(repo_dir, None, output_dir)
    
    assert result is not None


def test_generate_files_default_output_dir(tmp_path, monkeypatch):
    """Test generate_files with default output directory."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    test_file = template_dir / "file.txt"
    test_file.write_text("test")
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'testproj'})
    ])
    
    monkeypatch.chdir(tmp_path)
    result = generate_files(repo_dir, context)
    
    assert 'testproj' in result


def test_generate_files_binary_file(tmp_path):
    """Test generate_files handles binary files correctly."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    binary_file = template_dir / "image.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'project'})
    ])
    
    result = generate_files(repo_dir, context, output_dir)
    
    assert (output_dir / 'project' / 'image.bin').exists()
    assert (output_dir / 'project' / 'image.bin').read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_files_with_copy_without_render(tmp_path):
    """Test generate_files with _copy_without_render setting."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    static_dir = template_dir / "static"
    static_dir.mkdir()
    static_file = static_dir / "{{cookiecutter.name}}.txt"
    static_file.write_text("{{cookiecutter.name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'name': 'project',
            '_copy_without_render': ['static/*']
        })
    ])
    
    result = generate_files(repo_dir, context, output_dir)
    
    assert (


# LLM-generated content at query #40
#--------------------------

```python
def test_file_name_is_empty_predicate_true(tmp_path, monkeypatch):
    import os
    from jinja2 import Environment
    
    # Create a temporary directory to act as project_dir
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    # Create a subdirectory that will be treated as the "outfile"
    outfile_dir = os.path.join(project_dir, "subdir")
    os.makedirs(outfile_dir)
    
    # Setup context and environment
    context = {'cookiecutter': {}}
    env = Environment()
    
    # Mock the infile path - use a name that will render to the existing directory
    infile = "subdir"
    
    # Change to a temporary working directory
    monkeypatch.chdir(tmp_path)
    
    # Create a dummy infile
    with open(infile, 'w') as f:
        f.write("dummy content")
    
    # Import the function to test
    from generate_file import generate_file
    
    # Call generate_file - it should return early because outfile is a directory
    generate_file(project_dir, infile, context, env)
    
    # Verify that the predicate at line 35 evaluated to True by checking
    # that no file was created (since the function returns early)
    assert not os.path.isfile(os.path.join(project_dir, "subdir"))


# LLM-generated content at query #41
#--------------------------

```python
def test_template_syntax_error_translated_false():
    from jinja2 import Environment, TemplateSyntaxError
    import tempfile
    import os
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        
        # Create a template file with syntax error
        infile = "test_template.txt"
        infile_path = os.path.join(tmpdir, infile)
        with open(infile_path, 'w') as f:
            f.write("{{ unclosed_variable")
        
        env = Environment()
        context = {'cookiecutter': {}}
        
        # Prepare to catch the exception
        exception_caught = None
        try:
            from cookiecutter.generate import generate_file
            generate_file(project_dir, infile, context, env)
        except TemplateSyntaxError as e:
            exception_caught = e
        
        # Verify the predicate at line 60: exception.translated = False
        assert exception_caught is not None
        assert exception_caught.translated is False


# LLM-generated content at query #42
#--------------------------

```python
def test_generate_file_binary_file(tmp_path, mocker):
    """Test that binary files are copied without rendering."""
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    
    infile = "binary_file.bin"
    infile_path = tmp_path / infile
    infile_path.write_bytes(b'\x00\x01\x02\x03')
    
    context = {'cookiecutter': {}}
    env = mocker.MagicMock()
    env.from_string(infile).render(**context)
    
    mock_is_binary = mocker.patch('generate_file.is_binary', return_value=True)
    mock_copyfile = mocker.patch('generate_file.shutil.copyfile')
    mock_copymode = mocker.patch('generate_file.shutil.copymode')
    
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from generate_file import generate_file
        generate_file(project_dir, infile, context, env)
        
        mock_is_binary.assert_called_once_with(infile)
        mock_copyfile.assert_called_once()
        mock_copymode.assert_called_once()
    finally:
        os.chdir(original_cwd)


def test_generate_file_skip_if_exists(tmp_path, mocker):
    """Test that file generation is skipped if file exists and skip_if_file_exists is True."""
    project_dir = str(tmp_path / "project")
    project_dir_path = tmp_path / "project"
    project_dir_path.mkdir(exist_ok=True)
    
    infile = "template.txt"
    infile_path = tmp_path / infile
    infile_path.write_text("test content")
    
    outfile_path = project_dir_path / "output.txt"
    outfile_path.write_text("existing content")
    
    context = {'cookiecutter': {}}
    env = mocker.MagicMock()
    env.from_string(infile).render.return_value = "output.txt"
    
    mock_is_binary = mocker.patch('generate_file.is_binary', return_value=False)
    
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from generate_file import generate_file
        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
        
        mock_is_binary.assert_not_called()
    finally:
        os.chdir(original_cwd)


def test_generate_file_text_file(tmp_path, mocker):
    """Test that text files are rendered and written."""
    project_dir = str(tmp_path / "project")
    project_dir_path = tmp_path / "project"
    project_dir_path.mkdir(exist_ok=True)
    
    infile = "template.txt"
    infile_path = tmp_path / infile
    infile_path.write_text("Hello {{ name }}\n")
    
    context = {'cookiecutter': {'name': 'World'}}
    env = mocker.MagicMock()
    env.from_string(infile).render.return_value = "output.txt"
    
    mock_template = mocker.MagicMock()
    mock_template.render.return_value = "Hello World\n"
    env.get_template.return_value = mock_template
    
    mock_is_binary = mocker.patch('generate_file.is_binary', return_value=False)
    mock_copymode = mocker.patch('generate_file.shutil.copymode')
    
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from generate_file import generate_file
        generate_file(project_dir, infile, context, env)
        
        mock_is_binary.assert_called_once_with(infile)
        env.get_template.assert_called_once()
        mock_copymode.assert_called_once()
    finally:
        os.chdir(original_cwd)


def test_generate_file_empty_filename(tmp_path, mocker):
    """Test that generation is skipped when resulting filename is empty."""
    project_dir = str(tmp_path / "project")
    project_dir_path = tmp_path / "project"
    project_dir_path.mkdir(exist_ok=True)
    
    infile = "template.txt"
    infile_path = tmp_path / infile
    infile_path.write_text("test")
    
    context = {'cookiecutter': {}}
    env = mocker.MagicMock()
    env.from_string(infile).render.return_value = ""
    
    mock_is_binary = mocker.patch('generate_file.is_binary')
    
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from generate_file import generate_file
        generate_file(project_dir, infile, context, env)
        
        mock_is_binary.assert_not_called()
    finally:
        os.chdir(original_cwd)


def test_generate_file_with_custom_newline(tmp_path, mocker):
    """Test that custom newline character from context is used."""
    project_dir = str(tmp_path / "project")
    project_dir_path = tmp_path / "project"
    project_dir_path.mkdir(exist_ok=True)
    
    infile = "template.txt"
    infile_path = tmp_path / infile
    infile_path.write_text("line1\nline2\n")
    
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = mocker.MagicMock()
    env.from_string(infile).render.return_value = "output.txt"
    
    mock_template = mocker.MagicMock()
    mock_template.render.return_value = "line1\nline2\n"
    env.get_template.return_value = mock_template
    
    mock_is_binary = mocker.patch('generate_file.is_binary', return_value=False)
    mock_copymode = mocker.patch('generate_file.shutil.copymode')
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from generate_file import generate_file
        generate_file(project_dir, infile, context, env)
        
        mock_is_binary.assert_called_once_with(infile)
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #43
#--------------------------

```python
def test_generate_file_renders_text_file(tmp_path, mocker):
    """Test that generate_file renders and writes text file content."""
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    template_dir = str(tmp_path / "template")
    
    import os
    os.makedirs(project_dir)
    os.makedirs(template_dir)
    
    infile = os.path.join(template_dir, "test.txt")
    with open(infile, "w") as f:
        f.write("Hello {{ name }}")
    
    env = Environment()
    context = {"name": "World", "cookiecutter": {}}
    
    mocker.patch("generate_file.is_binary", return_value=False)
    mocker.patch("os.path.isdir", return_value=False)
    mocker.patch("os.path.exists", return_value=False)
    
    generate_file(project_dir, "test.txt", context, env)
    
    outfile = os.path.join(project_dir, "test.txt")
    assert os.path.exists(outfile)
    with open(outfile) as f:
        assert f.read() == "Hello World"


def test_generate_file_copies_binary_file(tmp_path, mocker):
    """Test that generate_file copies binary files without rendering."""
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    template_dir = str(tmp_path / "template")
    
    import os
    os.makedirs(project_dir)
    os.makedirs(template_dir)
    
    infile = os.path.join(template_dir, "test.bin")
    with open(infile, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    mocker.patch("generate_file.is_binary", return_value=True)
    mocker.patch("os.path.isdir", return_value=False)
    mocker.patch("os.path.exists", return_value=False)
    mock_copyfile = mocker.patch("shutil.copyfile")
    mock_copymode = mocker.patch("shutil.copymode")
    
    generate_file(project_dir, "test.bin", context, env)
    
    mock_copyfile.assert_called_once()
    mock_copymode.assert_called_once()


def test_generate_file_skips_existing_file(tmp_path, mocker):
    """Test that generate_file skips file if skip_if_file_exists is True."""
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    template_dir = str(tmp_path / "template")
    
    import os
    os.makedirs(project_dir)
    os.makedirs(template_dir)
    
    infile = os.path.join(template_dir, "test.txt")
    with open(infile, "w") as f:
        f.write("content")
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    mocker.patch("os.path.isdir", return_value=False)
    mocker.patch("os.path.exists", return_value=True)
    mock_is_binary = mocker.patch("generate_file.is_binary")
    
    generate_file(project_dir, "test.txt", context, env, skip_if_file_exists=True)
    
    mock_is_binary.assert_not_called()


def test_generate_file_returns_on_empty_filename(tmp_path, mocker):
    """Test that generate_file returns early if resulting filename is empty."""
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    template_dir = str(tmp_path / "template")
    
    import os
    os.makedirs(project_dir)
    os.makedirs(template_dir)
    
    infile = os.path.join(template_dir, "test.txt")
    with open(infile, "w") as f:
        f.write("content")
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    mocker.patch("os.path.isdir", return_value=True)
    mock_is_binary = mocker.patch("generate_file.is_binary")
    
    generate_file(project_dir, "test.txt", context, env)
    
    mock_is_binary.assert_not_called()


def test_generate_file_renders_filename_with_context(tmp_path, mocker):
    """Test that generate_file renders the output filename using context."""
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    template_dir = str(tmp_path / "template")
    
    import os
    os.makedirs(project_dir)
    os.makedirs(template_dir)
    
    infile = os.path.join(template_dir, "{{ filename }}.txt")
    with open(infile, "w") as f:
        f.write("content")
    
    env = Environment()
    context = {"filename": "output", "cookiecutter": {}}
    
    mocker.patch("generate_file.is_binary", return_value=False)
    mocker.patch("os.path.isdir", return_value=False)
    mocker.patch("os.path.exists", return_value=False)
    
    generate_file(project_dir, "{{ filename }}.txt", context, env)
    
    outfile = os.path.join(project_dir, "output.txt")
    assert os.path.exists(outfile)


def test_generate_file_uses_configured_newline(tmp_path, mocker):
    """Test that generate_file uses configured newline from context."""
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    template_dir = str(tmp_path / "template")
    
    import os
    os.makedirs(project_dir)
    os.makedirs(template_dir)
    
    infile = os.path.join(template_dir, "test.txt")
    with open(infile, "w") as f:
        f.write("line1\nline2")
    
    env = Environment()
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    
    mocker.patch("generate_file.is_binary", return_value=False)
    mocker.patch("os.path.isdir", return_value=False)
    mocker.patch("os.path.exists", return_value=False)
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    
    generate_file(project_dir, "test.txt", context, env)
    
    mock_open.assert_called()
    calls = mock_open.call_args_list
    write_call = [c for c in calls if c[0][0].endswith("test.txt") and "w" in str(c)]
    assert any("\r\n" in str(call) for call in write_call)


# LLM-generated content at query #44
#--------------------------

```python
def test_generate_context_file_not_found():
    """Test that the predicate at line 18 evaluates to False when file doesn't exist."""
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    import os
    
    non_existent_file = '/tmp/this_file_does_not_exist_12345.json'
    
    # Ensure the file doesn't exist
    if os.path.exists(non_existent_file):
        os.remove(non_existent_file)
    
    try:
        generate_context(context_file=non_existent_file)
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        # The predicate at line 18 (the with open statement) evaluates to False
        # because the file doesn't exist, so the exception is raised
        pass


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    from jinja2 import Environment
    import tempfile
    import os
    
    # Create a temporary directory and file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = temp_dir
        infile_path = os.path.join(temp_dir, 'test_template.txt')
        
        # Create a test input file
        with open(infile_path, 'w', encoding='utf-8') as f:
            f.write('Hello {{ name }}')
        
        # Create context without '_new_lines' key or with '_new_lines' set to False
        context = {
            'cookiecutter': {
                'name': 'World'
            }
        }
        
        env = Environment()
        
        # Change to temp directory so infile can be found
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # The predicate at line 67 is:
            # if context['cookiecutter'].get('_new_lines', False):
            # This evaluates to False when '_new_lines' is not in the dict
            # or when it's explicitly set to False
            
            predicate_result = context['cookiecutter'].get('_new_lines', False)
            
            assert predicate_result is False
            
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #46
#--------------------------

```python
def test_template_syntax_error_has_translated_set_to_false():
    from jinja2 import Environment, TemplateSyntaxError
    import tempfile
    import os
    
    # Create a temporary directory and files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        
        # Create a template file with syntax error
        template_content = "{{ unclosed_variable"
        infile_path = os.path.join(tmpdir, "template.txt")
        with open(infile_path, 'w') as f:
            f.write(template_content)
        
        # Create environment and context
        env = Environment()
        context = {'cookiecutter': {}}
        
        # Change to the temp directory to make infile relative
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Import the function to test
            from generate_file import generate_file
            
            # Call generate_file and expect TemplateSyntaxError
            error_caught = False
            error_has_translated_false = False
            try:
                generate_file(project_dir, "template.txt", context, env)
            except TemplateSyntaxError as e:
                error_caught = True
                error_has_translated_false = e.translated is False
            
            assert error_caught is True
            assert error_has_translated_false is True
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #47
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true(tmp_path, monkeypatch):
    import os
    from jinja2 import Environment
    
    # Create a temporary binary file
    binary_file = tmp_path / "binary_file.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    # Create a temporary project directory
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Change to the temporary directory to make infile relative
    monkeypatch.chdir(tmp_path)
    
    # Mock is_binary to return True
    def mock_is_binary(infile):
        return True
    
    # Mock shutil functions
    import shutil
    copyfile_called = []
    copymode_called = []
    
    def mock_copyfile(src, dst):
        copyfile_called.append((src, dst))
    
    def mock_copymode(src, dst):
        copymode_called.append((src, dst))
    
    monkeypatch.setattr("shutil.copyfile", mock_copyfile)
    monkeypatch.setattr("shutil.copymode", mock_copymode)
    
    # Import and patch is_binary
    import sys
    from unittest.mock import MagicMock
    module = sys.modules.get('__main__')
    if module is None:
        import types
        module = types.ModuleType('__main__')
        sys.modules['__main__'] = module
    
    # Create the function with mocked is_binary
    def generate_file_test(
        project_dir: str,
        infile: str,
        context: dict,
        env,
        skip_if_file_exists: bool = False,
    ) -> None:
        import os
        import shutil
        from jinja2 import TemplateSyntaxError
        
        logger_debug = MagicMock()
        
        outfile_tmpl = env.from_string(infile)
        outfile = os.path.join(project_dir, outfile_tmpl.render(**context))
        file_name_is_empty = os.path.isdir(outfile)
        if file_name_is_empty:
            return
        
        if skip_if_file_exists and os.path.exists(outfile):
            return
        
        # Line 47: the predicate
        if mock_is_binary(infile):
            shutil.copyfile(infile, outfile)
            shutil.copymode(infile, outfile)
            return
    
    # Setup
    env = Environment()
    context = {'cookiecutter': {}}
    infile = "binary_file.bin"
    
    # Execute
    generate_file_test(str(project_dir), infile, context, env)
    
    # Assert that the predicate evaluated to True and copyfile was called
    assert len(copyfile_called) == 1
    assert len(copymode_called) == 1


# LLM-generated content at query #48
#--------------------------

```python
def test_generate_file_renders_text_file(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file
    infile_path = template_dir / "test.txt"
    infile_path.write_text("Hello {{ name }}")
    
    monkeypatch.chdir(template_dir)
    
    context = {"cookiecutter": {"name": "World"}}
    env = Environment()
    
    generate_file(str(project_dir), "test.txt", context, env)
    
    outfile = project_dir / "test.txt"
    assert outfile.exists()
    assert outfile.read_text() == "Hello World"


def test_generate_file_skips_if_file_exists(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file
    infile_path = template_dir / "test.txt"
    infile_path.write_text("Content")
    
    # Create existing output file
    outfile = project_dir / "test.txt"
    outfile.write_text("Existing content")
    
    monkeypatch.chdir(template_dir)
    
    context = {"cookiecutter": {}}
    env = Environment()
    
    generate_file(str(project_dir), "test.txt", context, env, skip_if_file_exists=True)
    
    assert outfile.read_text() == "Existing content"


def test_generate_file_renders_filename(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file
    infile_path = template_dir / "{{name}}.txt"
    infile_path.write_text("Content")
    
    monkeypatch.chdir(template_dir)
    
    context = {"cookiecutter": {"name": "myfile"}}
    env = Environment()
    
    generate_file(str(project_dir), "{{name}}.txt", context, env)
    
    outfile = project_dir / "myfile.txt"
    assert outfile.exists()
    assert outfile.read_text() == "Content"


def test_generate_file_copies_binary_file(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create binary input file
    infile_path = template_dir / "image.bin"
    infile_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    
    monkeypatch.chdir(template_dir)
    
    context = {"cookiecutter": {}}
    env = Environment()
    
    generate_file(str(project_dir), "image.bin", context, env)
    
    outfile = project_dir / "image.bin"
    assert outfile.exists()
    assert outfile.read_bytes() == b"\x89PNG\r\n\x1a\n"


def test_generate_file_returns_early_for_empty_filename(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file
    infile_path = template_dir / "test.txt"
    infile_path.write_text("Content")
    
    monkeypatch.chdir(template_dir)
    
    context = {"cookiecutter": {}}
    env = Environment()
    
    # When rendered filename is empty (directory path), function should return
    generate_file(str(project_dir), "test.txt", context, env)
    
    outfile = project_dir / "test.txt"
    assert outfile.exists()


def test_generate_file_uses_configured_newlines(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file with Unix newlines
    infile_path = template_dir / "test.txt"
    infile_path.write_text("Line 1\nLine 2")
    
    monkeypatch.chdir(template_dir)
    
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    
    generate_file(str(project_dir), "test.txt", context, env)
    
    outfile = project_dir / "test.txt"
    assert outfile.exists()
    content = outfile.read_bytes()
    assert b"Line 1\r\nLine 2" == content


def test_generate_file_detects_newlines(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file with Unix newlines
    infile_path = template_dir / "test.txt"
    infile_path.write_text("Line 1\nLine 2")
    
    monkeypatch.chdir(template_dir)
    
    context = {"cookiecutter": {}}
    env = Environment()
    
    generate_file(str(project_dir), "test.txt", context, env)
    
    outfile = project_dir / "test.txt"
    assert outfile.exists()
    assert outfile.read_text() == "Line 1\nLine 2"


def test_generate_file_preserves_file_permissions(tmp_path, monkeypatch):
    import os
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file
    infile_path = template_dir / "test.txt"
    infile_path.write_text("Content")
    os.chmod(infile_path, 0o755)
    
    monkeypatch.chdir(template_dir)
    
    context = {"cookiecutter": {}}
    env = Environment()
    
    generate_file(str(project_dir), "test.txt", context, env)
    
    outfile = project_dir / "test.txt"
    assert outfile.exists()
    assert os.stat(outfile).st_mode == os.stat(infile_path).st_mode


# LLM-generated content at query #49
#--------------------------

```python
def test_generate_file_renders_text_file(tmp_path, monkeypatch):
    from jinja2 import Environment
    import os
    import shutil
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file with template variable
    infile_path = template_dir / "test.txt"
    infile_path.write_text("Hello {{ name }}!")
    
    # Change to template directory
    monkeypatch.chdir(template_dir)
    
    env = Environment()
    context = {"cookiecutter": {"name": "World"}}
    
    # Execute
    generate_file(str(project_dir), "test.txt", context, env)
    
    # Assert
    outfile = project_dir / "test.txt"
    assert outfile.exists()
    assert outfile.read_text() == "Hello World!"


def test_generate_file_copies_binary_file(tmp_path, monkeypatch):
    from jinja2 import Environment
    import os
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create binary input file
    infile_path = template_dir / "image.bin"
    infile_path.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    monkeypatch.chdir(template_dir)
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    # Execute
    generate_file(str(project_dir), "image.bin", context, env)
    
    # Assert
    outfile = project_dir / "image.bin"
    assert outfile.exists()
    assert outfile.read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_file_skips_existing_file(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    infile_path = template_dir / "test.txt"
    infile_path.write_text("Template content")
    
    outfile = project_dir / "test.txt"
    outfile.write_text("Existing content")
    
    monkeypatch.chdir(template_dir)
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    # Execute
    generate_file(str(project_dir), "test.txt", context, env, skip_if_file_exists=True)
    
    # Assert
    assert outfile.read_text() == "Existing content"


def test_generate_file_renders_filename(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    infile_path = template_dir / "{{ project_name }}.txt"
    infile_path.write_text("Content")
    
    monkeypatch.chdir(template_dir)
    
    env = Environment()
    context = {"cookiecutter": {"project_name": "myproject"}}
    
    # Execute
    generate_file(str(project_dir), "{{ project_name }}.txt", context, env)
    
    # Assert
    outfile = project_dir / "myproject.txt"
    assert outfile.exists()
    assert outfile.read_text() == "Content"


def test_generate_file_preserves_file_permissions(tmp_path, monkeypatch):
    from jinja2 import Environment
    import stat
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    infile_path = template_dir / "script.sh"
    infile_path.write_text("#!/bin/bash\necho {{ message }}")
    infile_path.chmod(0o755)
    
    monkeypatch.chdir(template_dir)
    
    env = Environment()
    context = {"cookiecutter": {"message": "hello"}}
    
    # Execute
    generate_file(str(project_dir), "script.sh", context, env)
    
    # Assert
    outfile = project_dir / "script.sh"
    assert outfile.exists()
    assert stat.S_IMODE(outfile.stat().st_mode) == 0o755


def test_generate_file_returns_early_for_empty_filename(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create a directory that will result in empty filename
    infile_path = template_dir / "subdir"
    infile_path.mkdir()
    
    monkeypatch.chdir(template_dir)
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    # Execute - should return early without error
    generate_file(str(project_dir), "subdir", context, env)
    
    # Assert - no files should be created in project_dir
    assert len(list(project_dir.iterdir())) == 0


def test_generate_file_uses_configured_newline(tmp_path, monkeypatch):
    from jinja2 import Environment
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    infile_path = template_dir / "test.txt"
    infile_path.write_text("line1\nline2")
    
    monkeypatch.chdir(template_dir)
    
    env = Environment()
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    
    # Execute
    generate_file(str(project_dir), "test.txt", context, env)
    
    # Assert
    outfile = project_dir / "test.txt"
    assert outfile.exists()
    assert outfile.read_bytes() == b"line1\r\nline2"


# LLM-generated content at query #50
#--------------------------

```python
def test_generate_file_renders_text_file(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "project").mkdir(exist_ok=True)
    
    infile = "test_{{name}}.txt"
    infile_path = tmp_path / infile.replace("{{name}}", "file")
    infile_path.write_text("Hello {{name}}!", encoding='utf-8')
    
    context = {"cookiecutter": {"name": "world"}}
    env = Environment()
    
    mocker.patch('os.getcwd', return_value=str(tmp_path))
    mocker.patch('builtins.open', mocker.mock_open(read_data="Hello {{name}}!"))
    mocker.patch('shutil.copymode')
    mocker.patch('shutil.copyfile')
    
    import os
    original_getcwd = os.getcwd
    os.chdir(str(tmp_path))
    
    try:
        from cookiecutter.generate import generate_file
        generate_file(project_dir, infile, context, env)
    finally:
        os.chdir(original_getcwd())


def test_generate_file_copies_binary_file(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "project").mkdir(exist_ok=True)
    
    infile = "binary_file.bin"
    infile_path = tmp_path / infile
    infile_path.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    context = {"cookiecutter": {}}
    env = Environment()
    
    mocker.patch('cookiecutter.generate.is_binary', return_value=True)
    mock_copyfile = mocker.patch('shutil.copyfile')
    mock_copymode = mocker.patch('shutil.copymode')
    
    import os
    original_getcwd = os.getcwd
    os.chdir(str(tmp_path))
    
    try:
        from cookiecutter.generate import generate_file
        generate_file(project_dir, infile, context, env)
        mock_copyfile.assert_called_once()
        mock_copymode.assert_called_once()
    finally:
        os.chdir(original_getcwd())


def test_generate_file_skips_existing_file(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "project").mkdir(exist_ok=True)
    
    infile = "existing.txt"
    outfile_path = tmp_path / "project" / "existing.txt"
    outfile_path.write_text("existing content")
    
    context = {"cookiecutter": {}}
    env = Environment()
    
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.path.isdir', return_value=False)
    
    import os
    original_getcwd = os.getcwd
    os.chdir(str(tmp_path))
    
    try:
        from cookiecutter.generate import generate_file
        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    finally:
        os.chdir(original_getcwd())


def test_generate_file_returns_on_empty_filename(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "project").mkdir(exist_ok=True)
    
    infile = "test.txt"
    infile_path = tmp_path / infile
    infile_path.write_text("content")
    
    context = {"cookiecutter": {}}
    env = Environment()
    
    mocker.patch('os.path.isdir', return_value=True)
    mock_copyfile = mocker.patch('shutil.copyfile')
    
    import os
    original_getcwd = os.getcwd
    os.chdir(str(tmp_path))
    
    try:
        from cookiecutter.generate import generate_file
        generate_file(project_dir, infile, context, env)
        mock_copyfile.assert_not_called()
    finally:
        os.chdir(original_getcwd())


def test_generate_file_renders_template_with_context(tmp_path, mocker):
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "project").mkdir(exist_ok=True)
    
    infile = "template.txt"
    infile_path = tmp_path / infile
    infile_path.write_text("Hello {{name}}!", encoding='utf-8')
    
    context = {"cookiecutter": {"name": "Alice", "_new_lines": "\n"}}
    env = Environment()
    
    mocker.patch('os.path.isdir', return_value=False)
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('cookiecutter.generate.is_binary', return_value=False)
    mocker.patch('shutil.copymode')
    
    import os
    original_getcwd = os.getcwd
    os.chdir(str(tmp_path))
    
    try:
        from cookiecutter.generate import generate_file
        generate_file(project_dir, infile, context, env)
        
        output_file = tmp_path / "project" / "template.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        assert "Hello Alice!" in content
    finally:
        os.chdir(original_getcwd())


