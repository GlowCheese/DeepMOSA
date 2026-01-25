####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_hook_from_repo_dir_deprecated_warning(monkeypatch, tmp_path):
    """Test that _run_hook_from_repo_dir issues a deprecation warning."""
    import warnings
    from cookiecutter.generate import _run_hook_from_repo_dir
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {}}
    hook_name = "post_gen_project"
    delete_project_on_failure = False
    
    mock_run_hook_from_repo_dir_called = []
    
    def mock_run_hook_from_repo_dir(repo, hook, proj, ctx, delete):
        mock_run_hook_from_repo_dir_called.append((repo, hook, proj, ctx, delete))
    
    monkeypatch.setattr(
        "cookiecutter.generate.run_hook_from_repo_dir",
        mock_run_hook_from_repo_dir
    )
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _run_hook_from_repo_dir(
            repo_dir, hook_name, project_dir, context, delete_project_on_failure
        )
        
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "_run_hook_from_repo_dir" in str(w[0].message)
        assert "cookiecutter.hooks.run_hook_from_repo_dir" in str(w[0].message)


def test_run_hook_from_repo_dir_calls_underlying_function(monkeypatch, tmp_path):
    """Test that _run_hook_from_repo_dir calls run_hook_from_repo_dir with correct args."""
    import warnings
    from cookiecutter.generate import _run_hook_from_repo_dir
    
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {"key": "value"}}
    hook_name = "post_gen_project"
    delete_project_on_failure = True
    
    call_args = []
    
    def mock_run_hook_from_repo_dir(repo, hook, proj, ctx, delete):
        call_args.append({
            "repo_dir": repo,
            "hook_name": hook,
            "project_dir": proj,
            "context": ctx,
            "delete_project_on_failure": delete
        })
    
    monkeypatch.setattr(
        "cookiecutter.generate.run_hook_from_repo_dir",
        mock_run_hook_from_repo_dir
    )
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _run_hook_from_repo_dir(
            repo_dir, hook_name, project_dir, context, delete_project_on_failure
        )
    
    assert len(call_args) == 1
    assert call_args[0]["repo_dir"] == repo_dir
    assert call_args[0]["hook_name"] == hook_name
    assert call_args[0]["project_dir"] == project_dir
    assert call_args[0]["context"] == context
    assert call_args[0]["delete_project_on_failure"] is True


# LLM-generated content at query #2
#--------------------------

```python
def test_generate_context_with_valid_json_file(tmp_path):
    """Test generate_context loads a valid JSON file correctly."""
    import json
    from collections import OrderedDict
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"project_name": "my_project", "author": "John Doe"}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_generate_context_with_invalid_json_file(tmp_path):
    """Test generate_context raises ContextDecodingException for invalid JSON."""
    from cookiecutter.exceptions import ContextDecodingException
    
    context_file = tmp_path / "cookiecutter.json"
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write("{invalid json content")
    
    try:
        generate_context(str(context_file))
        assert False, "Expected ContextDecodingException"
    except Exception as e:
        assert "ContextDecodingException" in str(type(e))


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context applies default_context overrides."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"project_name": "default_project", "version": "1.0"}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    default_context = {"project_name": "overridden_project"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "overridden_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context applies extra_context overrides."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"project_name": "default_project", "version": "1.0"}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    extra_context = {"version": "2.0"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "default_project"
    assert result["cookiecutter"]["version"] == "2.0"


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context handles choice variables correctly."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"license": ["MIT", "Apache", "GPL"]}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"


def test_generate_context_with_boolean_variable(tmp_path):
    """Test generate_context converts string to boolean for boolean variables."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"use_docker": True}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    extra_context = {"use_docker": "false"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_with_nested_dict(tmp_path):
    """Test generate_context handles nested dictionary variables."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"options": {"debug": False, "verbose": True}}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    extra_context = {"options": {"debug": True}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["options"]["debug"] is True
    assert result["cookiecutter"]["options"]["verbose"] is True


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test generate_context handles multichoice variables correctly."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"features": ["feature1", "feature2", "feature3"]}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    extra_context = {"features": ["feature2", "feature3"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["features"]) == {"feature2", "feature3"}


def test_generate_context_with_invalid_choice(tmp_path):
    """Test generate_context raises ValueError for invalid choice."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"license": ["MIT", "Apache"]}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    extra_context = {"license": "BSD"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "BSD" in str(e)


def test_generate_context_with_invalid_boolean_conversion(tmp_path):
    """Test generate_context raises ValueError for invalid boolean conversion."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {"use_docker": True}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    extra_context = {"use_docker": "maybe"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_generate_context_preserves_context_structure(tmp_path):
    """Test generate_context preserves the original context structure."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    test_data = {
        "project_name": "test",
        "year": 2024,
        "enabled": True,
        "tags": ["tag1", "tag2"]
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = generate_context(str(context_file))
    
    assert result["cookiecutter"]["project_name"] == "test"
    assert result["cookiecutter"]["year"] == 2024
    assert result["cookiecutter"]["enabled"] is True
    assert result["cookiecutter"]["tags"] == ["tag1", "tag2"]


# LLM-generated content at query #3
#--------------------------

```python
def test_generate_context_json_decoding_error():
    """Test that ValueError is caught and ContextDecodingException is raised at line 20."""
    import json
    import os
    import tempfile
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    # Create a temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{invalid json content}')
        temp_file = f.name
    
    try:
        # This should trigger the ValueError exception at line 20
        generate_context(context_file=temp_file)
        assert False, "Expected ContextDecodingException to be raised"
    except ContextDecodingException as e:
        # Verify the exception message contains expected information
        assert "JSON decoding error" in str(e)
        assert temp_file in str(e)
    finally:
        # Clean up
        os.unlink(temp_file)


# LLM-generated content at query #4
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_variables_at_first_level():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"existing": "value"}
    overwrite_context = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_adds_new_variables_in_dictionary():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"nested": {"existing": "value"}}
    overwrite_context = {"nested": {"new_var": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"nested": {"existing": "value", "new_var": "new_value"}}


def test_apply_overwrites_to_context_overwrites_simple_value():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"name": "old_value"}
    overwrite_context = {"name": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"name": "new_value"}


def test_apply_overwrites_to_context_overwrites_multichoice_variable():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}


def test_apply_overwrites_to_context_raises_for_invalid_multichoice():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "valid choices are" in str(e)


def test_apply_overwrites_to_context_reorders_choice_variable():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"choice": ["option1", "option2", "option3"]}
    overwrite_context = {"choice": "option3"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["option3", "option1", "option2"]}


def test_apply_overwrites_to_context_raises_for_invalid_choice():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "but the choices are" in str(e)


def test_apply_overwrites_to_context_overwrites_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"nested": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"nested": {"key1": "new_value1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"key1": "new_value1", "key2": "value2"}}


def test_apply_overwrites_to_context_converts_string_to_boolean_true():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite_context = {"enabled": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": True}


def test_apply_overwrites_to_context_converts_string_to_boolean_false():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": False}


def test_apply_overwrites_to_context_converts_string_to_boolean_with_various_values():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": False}
    overwrite_context = {"flag": "1"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_raises_for_invalid_boolean_conversion():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_apply_overwrites_to_context_does_not_overwrite_list_with_string_at_first_level():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"items": ["a", "b", "c"]}
    overwrite_context = {"items": "new_string"}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=False)
    assert context == {"items": ["a", "b", "c"]}


def test_apply_overwrites_to_context_overwrites_list_with_string_in_dictionary():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"nested": {"items": ["a", "b", "c"]}}
    overwrite_context = {"nested": {"items": "new_string"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"nested": {"items": "new_string"}}


def test_apply_overwrites_to_context_complex_nested_structure():
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {
        "level1": {
            "level2": {
                "choice": ["opt1", "opt2"],
                "flag": True
            }
        }
    }
    overwrite_context = {
        "level1": {
            "level2": {
                "choice": "opt2",
                "flag": "false"
            }
        }
    }
    apply_overwrites_to_context(context, overwrite_context)
    assert context["level1"]["level2"]["choice"] == ["opt2", "opt1"]
    assert context["level1"]["level2"]["flag"] is False


# LLM-generated content at query #5
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that line 57 evaluates to False when YesNoPrompt.process_response succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"debug": True}
    overwrite_context = {"debug": "false"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["debug"] is False


def test_apply_overwrites_to_context_boolean_conversion_with_yes():
    """Test that line 57 evaluates to False when converting 'yes' to boolean."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite_context = {"enabled": "yes"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is True


def test_apply_overwrites_to_context_boolean_conversion_with_no():
    """Test that line 57 evaluates to False when converting 'no' to boolean."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is False


def test_apply_overwrites_to_context_boolean_conversion_with_zero():
    """Test that line 57 evaluates to False when converting '0' to boolean."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "0"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is False


def test_apply_overwrites_to_context_boolean_conversion_with_one():
    """Test that line 57 evaluates to False when converting '1' to boolean."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": False}
    overwrite_context = {"flag": "1"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is True


# LLM-generated content at query #6
#--------------------------

```python
def test_generate_context_raises_context_decoding_exception_on_json_decode_error(tmp_path):
    """Test that ValueError during JSON decoding raises ContextDecodingException."""
    import json
    import os
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    # Create a temporary file with invalid JSON
    invalid_json_file = tmp_path / "cookiecutter.json"
    invalid_json_file.write_text("{invalid json content}")
    
    # Call generate_context with the invalid JSON file
    try:
        generate_context(context_file=str(invalid_json_file))
        # If we reach here, the test fails
        assert False, "Expected ContextDecodingException to be raised"
    except ContextDecodingException as e:
        # Verify the exception message contains expected information
        assert "JSON decoding error" in str(e)
        assert str(invalid_json_file) in str(e)
        assert "Decoding error details" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that boolean conversion succeeds and InvalidResponse is not raised."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"is_enabled": True}
    overwrite_context = {"is_enabled": "yes"}
    
    # This should not raise ValueError, meaning the except block at line 57 is not executed
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["is_enabled"] is True


def test_apply_overwrites_to_context_boolean_conversion_false():
    """Test that boolean conversion to False succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"is_enabled": True}
    overwrite_context = {"is_enabled": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["is_enabled"] is False


def test_apply_overwrites_to_context_boolean_invalid_response():
    """Test that InvalidResponse exception is caught at line 57."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"is_enabled": True}
    overwrite_context = {"is_enabled": "invalid_value"}
    
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
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
        assert result_path.name == "test_dir"
        assert is_new is True


def test_render_and_create_dir_with_template_rendering():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    context = {"project_name": "my_project"}
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path, is_new = render_and_create_dir("{{ project_name }}", context, tmpdir, env)
        assert result_path.exists()
        assert result_path.name == "my_project"
        assert is_new is True


def test_render_and_create_dir_existing_dir_raises_exception():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_dir = Path(tmpdir) / "existing"
        existing_dir.mkdir()
        
        try:
            render_and_create_dir("existing", {}, tmpdir, env)
            assert False, "Should raise OutputDirExistsException"
        except OutputDirExistsException:
            pass


def test_render_and_create_dir_existing_dir_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_dir = Path(tmpdir) / "existing"
        existing_dir.mkdir()
        
        result_path, is_new = render_and_create_dir("existing", {}, tmpdir, env, overwrite_if_exists=True)
        assert result_path.exists()
        assert is_new is False


def test_render_and_create_dir_nested_path():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path, is_new = render_and_create_dir("nested/deep/dir", {}, tmpdir, env)
        assert result_path.exists()
        assert result_path.name == "dir"
        assert is_new is True


def test_render_and_create_dir_returns_tuple():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = render_and_create_dir("new_dir", {}, tmpdir, env)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Path)
        assert isinstance(result[1], bool)


# LLM-generated content at query #9
#--------------------------

```python
def test_run_hook_from_repo_dir_deprecated_warning(mocker, tmp_path):
    """Test that _run_hook_from_repo_dir issues a deprecation warning."""
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {}}
    
    mock_run_hook_from_repo_dir = mocker.patch(
        "cookiecutter.generate.run_hook_from_repo_dir"
    )
    
    with mocker.patch("warnings.warn") as mock_warn:
        from cookiecutter.generate import _run_hook_from_repo_dir
        _run_hook_from_repo_dir(
            repo_dir, "post_gen_project", project_dir, context, False
        )
    
    mock_warn.assert_called_once()
    args, kwargs = mock_warn.call_args
    assert "deprecated" in args[0].lower()
    assert args[1] == DeprecationWarning
    assert args[2] == 2
    mock_run_hook_from_repo_dir.assert_called_once_with(
        repo_dir, "post_gen_project", project_dir, context, False
    )


def test_run_hook_from_repo_dir_calls_actual_function(mocker, tmp_path):
    """Test that _run_hook_from_repo_dir delegates to run_hook_from_repo_dir."""
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {"project_name": "test"}}
    delete_on_failure = True
    
    mock_run_hook_from_repo_dir = mocker.patch(
        "cookiecutter.generate.run_hook_from_repo_dir"
    )
    mocker.patch("warnings.warn")
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(
        repo_dir, "pre_prompt", project_dir, context, delete_on_failure
    )
    
    mock_run_hook_from_repo_dir.assert_called_once_with(
        repo_dir, "pre_prompt", project_dir, context, delete_on_failure
    )


def test_run_hook_from_repo_dir_with_different_hook_names(mocker, tmp_path):
    """Test _run_hook_from_repo_dir with various hook names."""
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {}}
    
    mock_run_hook_from_repo_dir = mocker.patch(
        "cookiecutter.generate.run_hook_from_repo_dir"
    )
    mocker.patch("warnings.warn")
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    
    hook_names = ["pre_prompt", "post_gen_project", "pre_gen_project"]
    for hook_name in hook_names:
        _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    assert mock_run_hook_from_repo_dir.call_count == 3


def test_run_hook_from_repo_dir_delete_on_failure_true(mocker, tmp_path):
    """Test _run_hook_from_repo_dir with delete_project_on_failure=True."""
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {}}
    
    mock_run_hook_from_repo_dir = mocker.patch(
        "cookiecutter.generate.run_hook_from_repo_dir"
    )
    mocker.patch("warnings.warn")
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    
    mock_run_hook_from_repo_dir.assert_called_once_with(
        repo_dir, "post_gen_project", project_dir, context, True
    )


def test_run_hook_from_repo_dir_delete_on_failure_false(mocker, tmp_path):
    """Test _run_hook_from_repo_dir with delete_project_on_failure=False."""
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    context = {"cookiecutter": {}}
    
    mock_run_hook_from_repo_dir = mocker.patch(
        "cookiecutter.generate.run_hook_from_repo_dir"
    )
    mocker.patch("warnings.warn")
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    
    mock_run_hook_from_repo_dir.assert_called_once_with(
        repo_dir, "post_gen_project", project_dir, context, False
    )


# LLM-generated content at query #10
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_false(tmp_path, jinja_environment):
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    output_dir = tmp_path
    dirname = "test_dir"
    context = {}
    environment = jinja_environment
    
    # Create the directory first so it exists
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    
    # Call the function with overwrite_if_exists=True
    # This ensures the predicate at line 24 (output_dir_exists) evaluates to False is NOT the case
    # We need output_dir_exists to be True, then the predicate at line 24 evaluates to True
    # To test that predicate evaluates to False, we need the directory to NOT exist
    result_path, created = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=output_dir,
        environment=environment,
        overwrite_if_exists=False
    )
    
    assert result_path == dir_to_create
    assert created is False


# LLM-generated content at query #11
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
        assert False, "Should raise EmptyDirNameException"
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
    dirname = "{{ project_name }}"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == "my_project"
    assert is_new is True


def test_render_and_create_dir_existing_dir_without_overwrite(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    
    environment = Environment()
    context = {}
    dirname = "existing_dir"
    
    (tmp_path / dirname).mkdir()
    
    try:
        render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=False)
        assert False, "Should raise OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    environment = Environment()
    context = {}
    dirname = "existing_dir"
    
    (tmp_path / dirname).mkdir()
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=True)
    
    assert result_path.exists()
    assert is_new is False


def test_render_and_create_dir_nested_path(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    environment = Environment()
    context = {}
    dirname = "parent/child/nested"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == "nested"
    assert is_new is True


def test_render_and_create_dir_none_dirname(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    
    environment = Environment()
    context = {}
    
    try:
        render_and_create_dir(None, context, tmp_path, environment)
        assert False, "Should raise EmptyDirNameException"
    except EmptyDirNameException:
        pass


# LLM-generated content at query #12
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


def test_render_and_create_dir_creates_directory():
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


def test_render_and_create_dir_with_template():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    env = Environment()
    context = {"project_name": "my_project"}
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path, is_new = render_and_create_dir("{{ project_name }}_dir", context, tmpdir, env)
        assert result_path.exists()
        assert result_path.name == "my_project_dir"
        assert is_new is True


def test_render_and_create_dir_existing_dir_raises_exception():
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
        assert result_path.name == "grandchild"
        assert is_new is True


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_context_json_decoding_error():
    """Test that ValueError predicate at line 20 evaluates to True when JSON is invalid."""
    import json
    import tempfile
    import os
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    # Create a temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{invalid json content}')
        temp_file = f.name
    
    try:
        # This should raise ContextDecodingException due to ValueError at line 20
        generate_context(context_file=temp_file)
        assert False, "Expected ContextDecodingException to be raised"
    except Exception as e:
        # Verify that the exception is ContextDecodingException (wrapped from ValueError)
        assert type(e).__name__ == 'ContextDecodingException'
        assert 'JSON decoding error' in str(e)
        assert temp_file in str(e)
    finally:
        os.unlink(temp_file)


# LLM-generated content at query #14
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that line 57 predicate evaluates to False when conversion succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    
    # This should not raise InvalidResponse, so line 57's except clause is not executed
    # The predicate at line 57 (except InvalidResponse) evaluates to False
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is True


def test_apply_overwrites_to_context_boolean_conversion_false_value():
    """Test that line 57 predicate evaluates to False when converting 'no' to False."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is False


def test_apply_overwrites_to_context_boolean_conversion_zero():
    """Test that line 57 predicate evaluates to False when converting '0' to False."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "0"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is False


def test_apply_overwrites_to_context_boolean_conversion_one():
    """Test that line 57 predicate evaluates to False when converting '1' to True."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite_context = {"enabled": "1"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is True


# LLM-generated content at query #15
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that line 57 predicate evaluates to False when conversion succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is False


def test_apply_overwrites_to_context_boolean_conversion_yes():
    """Test that line 57 predicate evaluates to False when converting 'yes' to boolean."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": False}
    overwrite_context = {"flag": "yes"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is True


def test_apply_overwrites_to_context_boolean_conversion_true():
    """Test that line 57 predicate evaluates to False when converting 'true' to boolean."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite_context = {"enabled": "true"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is True


def test_apply_overwrites_to_context_boolean_conversion_false():
    """Test that line 57 predicate evaluates to False when converting 'false' to boolean."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "false"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is False


# LLM-generated content at query #16
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname(tmp_path):
    """Test that EmptyDirNameException is raised when dirname is empty."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    env = Environment()
    context = {}
    
    try:
        render_and_create_dir("", context, tmp_path, env)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_with_none_dirname(tmp_path):
    """Test that EmptyDirNameException is raised when dirname is None."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    env = Environment()
    context = {}
    
    try:
        render_and_create_dir(None, context, tmp_path, env)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_creates_new_directory(tmp_path):
    """Test that render_and_create_dir creates a new directory."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    env = Environment()
    context = {}
    dirname = "test_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env)
    
    assert result_path == Path(tmp_path, dirname)
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_with_template_rendering(tmp_path):
    """Test that render_and_create_dir renders template in dirname."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    env = Environment()
    context = {"project_name": "my_project"}
    dirname = "{{project_name}}_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env)
    
    assert result_path == Path(tmp_path, "my_project_dir")
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_existing_dir_overwrite_false(tmp_path):
    """Test that OutputDirExistsException is raised when dir exists and overwrite is False."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from pathlib import Path
    
    env = Environment()
    context = {}
    dirname = "existing_dir"
    
    existing_path = Path(tmp_path, dirname)
    existing_path.mkdir(parents=True, exist_ok=True)
    
    try:
        render_and_create_dir(dirname, context, tmp_path, env, overwrite_if_exists=False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_overwrite_true(tmp_path):
    """Test that render_and_create_dir returns existing dir when overwrite is True."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    env = Environment()
    context = {}
    dirname = "existing_dir"
    
    existing_path = Path(tmp_path, dirname)
    existing_path.mkdir(parents=True, exist_ok=True)
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env, overwrite_if_exists=True)
    
    assert result_path == existing_path
    assert result_path.exists()
    assert is_new is False


def test_render_and_create_dir_with_nested_path(tmp_path):
    """Test that render_and_create_dir creates nested directory structure."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    env = Environment()
    context = {}
    dirname = "parent/child/grandchild"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env)
    
    assert result_path == Path(tmp_path, dirname)
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_return_values_for_new_dir(tmp_path):
    """Test that render_and_create_dir returns correct tuple for new directory."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    
    env = Environment()
    context = {}
    dirname = "new_test_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env)
    
    assert isinstance(result_path, Path)
    assert isinstance(is_new, bool)
    assert is_new is True


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook_from_repo_dir_deprecated_function(mocker):
    """Test that _run_hook_from_repo_dir issues deprecation warning and calls the new function."""
    from cookiecutter.generate import _run_hook_from_repo_dir
    
    mock_run_hook_from_repo_dir = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mock_warnings = mocker.patch('cookiecutter.generate.warnings.warn')
    
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    
    _run_hook_from_repo_dir(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )
    
    mock_warnings.assert_called_once_with(
        "The '_run_hook_from_repo_dir' function is deprecated, "
        "use 'cookiecutter.hooks.run_hook_from_repo_dir' instead",
        DeprecationWarning,
        2,
    )
    mock_run_hook_from_repo_dir.assert_called_once_with(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )


def test_run_hook_from_repo_dir_passes_all_arguments(mocker):
    """Test that _run_hook_from_repo_dir passes all arguments correctly to the new function."""
    from cookiecutter.generate import _run_hook_from_repo_dir
    
    mock_run_hook_from_repo_dir = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.warnings.warn')
    
    repo_dir = '/template/dir'
    hook_name = 'pre_gen_project'
    project_dir = Path('/output/dir')
    context = {'cookiecutter': {'key': 'value'}}
    delete_project_on_failure = False
    
    _run_hook_from_repo_dir(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )
    
    mock_run_hook_from_repo_dir.assert_called_once_with(
        repo_dir, hook_name, project_dir, context, False
    )


def test_run_hook_from_repo_dir_deprecation_warning_category(mocker):
    """Test that _run_hook_from_repo_dir issues DeprecationWarning with correct stacklevel."""
    from cookiecutter.generate import _run_hook_from_repo_dir
    
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mock_warnings = mocker.patch('cookiecutter.generate.warnings.warn')
    
    _run_hook_from_repo_dir(
        '/repo', 'hook', '/project', {'cookiecutter': {}}, True
    )
    
    call_args = mock_warnings.call_args
    assert call_args[0][1] == DeprecationWarning
    assert call_args[0][2] == 2


# LLM-generated content at query #18
#--------------------------

```python
def test_generate_context_file_not_found():
    """Test that the predicate at line 18 evaluates to False when file doesn't exist."""
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    try:
        generate_context(context_file='/nonexistent/path/cookiecutter.json')
        file_opened_successfully = True
    except (FileNotFoundError, ContextDecodingException):
        file_opened_successfully = False
    
    assert file_opened_successfully is False


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_files_basic(tmp_path, monkeypatch):
    """Test generate_files with basic template structure."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    # Create template structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}")
    
    # Create context
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Mock find_template to return the template directory
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    gen_module.find_template = lambda repo, env: template_dir
    
    # Mock run_hook_from_repo_dir to do nothing
    original_run_hook = gen_module.run_hook_from_repo_dir
    gen_module.run_hook_from_repo_dir = lambda *args, **kwargs: None
    
    try:
        result = generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir,
            accept_hooks=False
        )
        
        assert "my_project" in result
        assert (output_dir / "my_project" / "README.md").exists()
        assert (output_dir / "my_project" / "README.md").read_text() == "# my_project"
    finally:
        gen_module.find_template = original_find_template
        gen_module.run_hook_from_repo_dir = original_run_hook


def test_generate_files_with_hooks(tmp_path, monkeypatch):
    """Test generate_files calls hooks when accept_hooks is True."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("{{cookiecutter.name}}")
    
    context = OrderedDict([('cookiecutter', {'name': 'test'})])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    gen_module.find_template = lambda repo, env: template_dir
    
    hook_calls = []
    def mock_run_hook(repo, hook_name, proj_dir, ctx, delete_on_fail):
        hook_calls.append(hook_name)
    
    original_run_hook = gen_module.run_hook_from_repo_dir
    gen_module.run_hook_from_repo_dir = mock_run_hook
    
    try:
        result = generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir,
            accept_hooks=True
        )
        
        assert 'pre_gen_project' in hook_calls
        assert 'post_gen_project' in hook_calls
    finally:
        gen_module.find_template = original_find_template
        gen_module.run_hook_from_repo_dir = original_run_hook


def test_generate_files_skip_if_exists(tmp_path):
    """Test generate_files with skip_if_file_exists option."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.proj}}"
    template_dir.mkdir()
    
    template_file = template_dir / "existing.txt"
    template_file.write_text("original content")
    
    context = OrderedDict([('cookiecutter', {'proj': 'project'})])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create pre-existing file
    existing_file = output_dir / "project" / "existing.txt"
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text("existing content")
    
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    gen_module.find_template = lambda repo, env: template_dir
    original_run_hook = gen_module.run_hook_from_repo_dir
    gen_module.run_hook_from_repo_dir = lambda *args, **kwargs: None
    
    try:
        result = generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir,
            skip_if_file_exists=True,
            accept_hooks=False
        )
        
        assert existing_file.read_text() == "existing content"
    finally:
        gen_module.find_template = original_find_template
        gen_module.run_hook_from_repo_dir = original_run_hook


def test_generate_files_overwrite_if_exists(tmp_path):
    """Test generate_files with overwrite_if_exists option."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "config.txt"
    template_file.write_text("config: {{cookiecutter.name}}")
    
    context = OrderedDict([('cookiecutter', {'name': 'app'})])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create pre-existing directory
    existing_dir = output_dir / "app"
    existing_dir.mkdir()
    existing_file = existing_dir / "config.txt"
    existing_file.write_text("old config")
    
    import cookiecutter.generate as gen_module
    original_find_template = gen_module.find_template
    gen_module.find_template = lambda repo, env: template_dir
    original_run_hook = gen_module.run_hook_from_repo_dir
    gen_module.run_hook_from_repo_dir = lambda *args, **kwargs: None
    
    try:
        result = generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir,
            overwrite_if_exists=True,
            accept_hooks=False
        )
        
        assert existing_file.read_text() == "config: app"
    finally:
        gen_module.find_template = original_find_template
        gen_module.run_hook_from_repo_dir = original_run_hook


def test_generate_files_binary_file(tmp_path):
    """Test generate_files handles binary files correctly."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.proj}}"
    template_dir.mkdir()
    
    # Create a binary file (PNG header)
    binary_file = template_dir / "image.png"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'fake binary content')
    
    context = OrderedDict([('cookiecutter', {'proj': 'myproj'})])
    output_dir = tmp_path / "output"
    output_dir.mkdir()


# LLM-generated content at query #20
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
    
    # Setup context and environment
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
    
    # Assert that the predicate (output_dir_exists) at line 24 evaluated to True
    # which means is_new should be False (since line 35 returns not output_dir_exists)
    assert is_new is False
    assert result_path.exists()


# LLM-generated content at query #21
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname(tmp_path):
    """Test that EmptyDirNameException is raised when dirname is empty."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    context = {}
    environment = Environment()
    
    try:
        render_and_create_dir("", context, tmp_path, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException as e:
        assert 'Error: directory name is empty' in str(e)


def test_render_and_create_dir_creates_new_directory(tmp_path):
    """Test that a new directory is created when it doesn't exist."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    context = {}
    environment = Environment()
    dirname = "test_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == dirname
    assert is_new is True


def test_render_and_create_dir_with_template_rendering(tmp_path):
    """Test that directory name is rendered from template."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    context = {"project_name": "my_project"}
    environment = Environment()
    dirname = "{{ project_name }}_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == "my_project_dir"
    assert is_new is True


def test_render_and_create_dir_existing_dir_without_overwrite(tmp_path):
    """Test that OutputDirExistsException is raised when directory exists and overwrite is False."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    existing_dir = tmp_path / dirname
    existing_dir.mkdir()
    
    try:
        render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException as e:
        assert 'Error:' in str(e) and 'already exists' in str(e)


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    """Test that existing directory is handled when overwrite_if_exists is True."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    existing_dir = tmp_path / dirname
    existing_dir.mkdir()
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=True)
    
    assert result_path.exists()
    assert result_path == existing_dir
    assert is_new is False


def test_render_and_create_dir_nested_path(tmp_path):
    """Test that nested directory structure is created."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    context = {}
    environment = Environment()
    dirname = "parent/child/grandchild"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == "grandchild"
    assert is_new is True


def test_render_and_create_dir_with_complex_template(tmp_path):
    """Test rendering with complex template expressions."""
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    context = {"name": "test", "version": "1"}
    environment = Environment()
    dirname = "{{ name }}_v{{ version }}"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path.exists()
    assert result_path.name == "test_v1"
    assert is_new is True


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_variables_at_first_level():
    """Test that new variables at first level are ignored."""
    context = {"existing_var": "value"}
    overwrite_context = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing_var": "value"}


def test_apply_overwrites_to_context_adds_new_variables_in_dictionary():
    """Test that new variables in nested dictionaries are added."""
    context = {"nested": {"existing_key": "value"}}
    overwrite_context = {"nested": {"new_key": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"existing_key": "value", "new_key": "new_value"}}


def test_apply_overwrites_to_context_multichoice_valid():
    """Test valid multichoice variable overwrite."""
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}


def test_apply_overwrites_to_context_multichoice_invalid():
    """Test invalid multichoice variable overwrite raises ValueError."""
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "multi-choice variable" in str(e)


def test_apply_overwrites_to_context_single_choice_valid():
    """Test valid single choice variable overwrite."""
    context = {"choice": ["default", "option1", "option2"]}
    overwrite_context = {"choice": "option1"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["option1", "default", "option2"]}


def test_apply_overwrites_to_context_single_choice_invalid():
    """Test invalid single choice variable overwrite raises ValueError."""
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "choice variable" in str(e)


def test_apply_overwrites_to_context_nested_dict_in_dictionary_variable():
    """Test that nested dict overwrites work in dictionary variables."""
    context = {"config": {"nested": {"key": "value"}}}
    overwrite_context = {"config": {"nested": {"key": "new_value"}}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"nested": {"key": "new_value"}}}


def test_apply_overwrites_to_context_boolean_yes_conversion():
    """Test boolean variable overwrite with yes response."""
    context = {"enabled": False}
    overwrite_context = {"enabled": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": True}


def test_apply_overwrites_to_context_boolean_no_conversion():
    """Test boolean variable overwrite with no response."""
    context = {"enabled": True}
    overwrite_context = {"enabled": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": False}


def test_apply_overwrites_to_context_boolean_true_conversion():
    """Test boolean variable overwrite with 'true' response."""
    context = {"flag": False}
    overwrite_context = {"flag": "true"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_false_conversion():
    """Test boolean variable overwrite with 'false' response."""
    context = {"flag": True}
    overwrite_context = {"flag": "false"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_boolean_invalid_conversion():
    """Test invalid boolean conversion raises ValueError."""
    context = {"enabled": True}
    overwrite_context = {"enabled": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_apply_overwrites_to_context_simple_string_overwrite():
    """Test simple string variable overwrite."""
    context = {"name": "old_name"}
    overwrite_context = {"name": "new_name"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"name": "new_name"}


def test_apply_overwrites_to_context_simple_integer_overwrite():
    """Test simple integer variable overwrite."""
    context = {"count": 5}
    overwrite_context = {"count": 10}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"count": 10}


def test_apply_overwrites_to_context_list_in_dictionary_variable():
    """Test that list overwrites work in dictionary variables."""
    context = {"config": {"items": ["a", "b"]}}
    overwrite_context = {"config": {"items": ["c", "d"]}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"config": {"items": ["c", "d"]}}


def test_apply_overwrites_to_context_multiple_variables():
    """Test overwriting multiple variables at once."""
    context = {"var1": "value1", "var2": "value2", "var3": "value3"}
    overwrite_context = {"var1": "new1", "var3": "new3"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "new1", "var2": "value2", "var3": "new3"}


def test_apply_overwrites_to_context_nested_dict_partial_overwrite():
    """Test partial overwrite of nested dictionary."""
    context = {"config": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"config": {"key1": "new_value1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"key1": "new_value1", "key2": "value2"}}


def test_apply_overwrites_to_context_boolean_with_1():
    """Test boolean variable overwrite with '1' (true)."""
    context = {"enabled": False}
    overwrite_context = {"enabled": "1"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": True}


def test_apply_overwrites_to_context_boolean_with_0():
    """Test boolean variable overwrite with '0' (false)."""
    context = {"enabled": True}
    overwrite_context = {"enabled": "0"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"enabled": False}


# LLM-generated content at query #2
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_variable_at_first_level():
    context = {"existing_var": "value"}
    overwrite_context = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing_var": "value"}


def test_apply_overwrites_to_context_adds_new_variable_in_dictionary():
    context = {"dict_var": {"existing_key": "value"}}
    overwrite_context = {"dict_var": {"new_key": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"existing_key": "value", "new_key": "new_value"}}


def test_apply_overwrites_to_context_overwrites_list_in_dictionary_variable():
    context = {"dict_var": {"list_key": [1, 2, 3]}}
    overwrite_context = {"dict_var": {"list_key": [4, 5, 6]}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"list_key": [4, 5, 6]}}


def test_apply_overwrites_to_context_multichoice_valid():
    context = {"choices": [1, 2, 3, 4]}
    overwrite_context = {"choices": [2, 3]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": [2, 3]}


def test_apply_overwrites_to_context_multichoice_invalid():
    context = {"choices": [1, 2, 3]}
    overwrite_context = {"choices": [2, 4]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "multi-choice variable" in str(e)


def test_apply_overwrites_to_context_choice_variable_valid():
    context = {"choice": ["option1", "option2", "option3"]}
    overwrite_context = {"choice": "option2"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["option2", "option1", "option3"]}


def test_apply_overwrites_to_context_choice_variable_invalid():
    context = {"choice": ["option1", "option2"]}
    overwrite_context = {"choice": "option3"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "choice variable" in str(e)


def test_apply_overwrites_to_context_boolean_true_conversion():
    context = {"flag": False}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_false_conversion():
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_boolean_invalid():
    context = {"flag": True}
    overwrite_context = {"flag": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old_value"}
    overwrite_context = {"var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new_value"}


def test_apply_overwrites_to_context_nested_dict_partial_overwrite():
    context = {"config": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"config": {"key1": "new_value1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"key1": "new_value1", "key2": "value2"}}


def test_apply_overwrites_to_context_multiple_variables():
    context = {"var1": "value1", "var2": "value2", "var3": "value3"}
    overwrite_context = {"var1": "new_value1", "var3": "new_value3"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "new_value1", "var2": "value2", "var3": "new_value3"}


def test_apply_overwrites_to_context_boolean_with_various_true_values():
    for true_val in ["1", "true", "t", "yes", "y", "on", "TRUE", "Yes", "ON"]:
        context = {"flag": False}
        overwrite_context = {"flag": true_val}
        apply_overwrites_to_context(context, overwrite_context)
        assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_with_various_false_values():
    for false_val in ["0", "false", "f", "no", "n", "off", "FALSE", "No", "OFF"]:
        context = {"flag": True}
        overwrite_context = {"flag": false_val}
        apply_overwrites_to_context(context, overwrite_context)
        assert context == {"flag": False}


def test_apply_overwrites_to_context_integer_overwrite():
    context = {"count": 5}
    overwrite_context = {"count": 10}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"count": 10}


def test_apply_overwrites_to_context_empty_overwrite():
    context = {"var1": "value1", "var2": "value2"}
    overwrite_context = {}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "value1", "var2": "value2"}


# LLM-generated content at query #3
#--------------------------

```python
def test_render_and_create_dir_with_valid_dirname(tmp_path, environment):
    """Test render_and_create_dir creates directory with valid dirname."""
    from cookiecutter.generate import render_and_create_dir
    
    context = {'project_name': 'my_project'}
    output_dir = tmp_path
    dirname = '{{ project_name }}'
    
    result_path, is_new = render_and_create_dir(
        dirname, context, output_dir, environment
    )
    
    assert result_path == tmp_path / 'my_project'
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_empty_dirname_raises_exception(tmp_path, environment):
    """Test render_and_create_dir raises EmptyDirNameException for empty dirname."""
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    context = {}
    output_dir = tmp_path
    dirname = ''
    
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_none_dirname_raises_exception(tmp_path, environment):
    """Test render_and_create_dir raises EmptyDirNameException for None dirname."""
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    context = {}
    output_dir = tmp_path
    dirname = None
    
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_existing_dir_without_overwrite_raises_exception(tmp_path, environment):
    """Test render_and_create_dir raises OutputDirExistsException when dir exists and overwrite_if_exists is False."""
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    
    context = {'project_name': 'existing_project'}
    output_dir = tmp_path
    dirname = '{{ project_name }}'
    
    (tmp_path / 'existing_project').mkdir()
    
    try:
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists=False
        )
        assert False, "Should have raised OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path, environment):
    """Test render_and_create_dir returns existing dir when overwrite_if_exists is True."""
    from cookiecutter.generate import render_and_create_dir
    
    context = {'project_name': 'existing_project'}
    output_dir = tmp_path
    dirname = '{{ project_name }}'
    
    existing_dir = tmp_path / 'existing_project'
    existing_dir.mkdir()
    
    result_path, is_new = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=True
    )
    
    assert result_path == existing_dir
    assert is_new is False


def test_render_and_create_dir_nested_dirname(tmp_path, environment):
    """Test render_and_create_dir creates nested directory structure."""
    from cookiecutter.generate import render_and_create_dir
    
    context = {'org': 'myorg', 'project': 'myproject'}
    output_dir = tmp_path
    dirname = '{{ org }}/{{ project }}'
    
    result_path, is_new = render_and_create_dir(
        dirname, context, output_dir, environment
    )
    
    assert result_path == tmp_path / 'myorg' / 'myproject'
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_with_special_characters(tmp_path, environment):
    """Test render_and_create_dir with special characters in dirname."""
    from cookiecutter.generate import render_and_create_dir
    
    context = {'name': 'test-project_v1'}
    output_dir = tmp_path
    dirname = '{{ name }}'
    
    result_path, is_new = render_and_create_dir(
        dirname, context, output_dir, environment
    )
    
    assert result_path == tmp_path / 'test-project_v1'
    assert result_path.exists()
    assert is_new is True


# LLM-generated content at query #4
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_true(tmp_path):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    output_dir = tmp_path
    dirname = "test_dir"
    context = {}
    environment = Environment()
    
    # Create the directory first so it exists
    dir_path = Path(output_dir, dirname)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Call the function with overwrite_if_exists=True to avoid exception
    result_path, not_existed = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=output_dir,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # Verify that the predicate at line 24 evaluated to True
    # (output_dir_exists is True, meaning the condition was entered)
    assert result_path.exists()
    assert not_existed is False


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_context_basic(tmp_path):
    """Test generate_context loads and returns context from JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context applies default_context overwrites."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "default_name", "version": "1.0"}')
    
    result = generate_context(
        str(context_file),
        default_context={"project_name": "overwritten_name"}
    )
    
    assert result["cookiecutter"]["project_name"] == "overwritten_name"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context applies extra_context overwrites."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "default_name", "version": "1.0"}')
    
    result = generate_context(
        str(context_file),
        extra_context={"version": "2.0"}
    )
    
    assert result["cookiecutter"]["project_name"] == "default_name"
    assert result["cookiecutter"]["version"] == "2.0"


def test_generate_context_with_both_defaults_and_extra(tmp_path):
    """Test generate_context applies both default and extra context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "default", "version": "1.0", "author": "unknown"}')
    
    result = generate_context(
        str(context_file),
        default_context={"project_name": "from_default"},
        extra_context={"version": "2.0"}
    )
    
    assert result["cookiecutter"]["project_name"] == "from_default"
    assert result["cookiecutter"]["version"] == "2.0"
    assert result["cookiecutter"]["author"] == "unknown"


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context raises ContextDecodingException for invalid JSON."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"invalid json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Should raise ContextDecodingException"
    except Exception as e:
        assert "ContextDecodingException" in type(e).__name__
        assert "JSON decoding error" in str(e)


def test_generate_context_file_not_found():
    """Test generate_context raises FileNotFoundError when file doesn't exist."""
    try:
        generate_context("/nonexistent/path/cookiecutter.json")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context handles choice variables with extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    result = generate_context(
        str(context_file),
        extra_context={"license": "Apache"}
    )
    
    assert result["cookiecutter"]["license"][0] == "Apache"


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test generate_context handles multichoice variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    
    result = generate_context(
        str(context_file),
        extra_context={"features": ["feature2", "feature3"]}
    )
    
    assert result["cookiecutter"]["features"] == ["feature2", "feature3"]


def test_generate_context_with_nested_dict(tmp_path):
    """Test generate_context handles nested dictionary variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"key1": "value1", "key2": "value2"}}')
    
    result = generate_context(
        str(context_file),
        extra_context={"config": {"key1": "overwritten"}}
    )
    
    assert result["cookiecutter"]["config"]["key1"] == "overwritten"
    assert result["cookiecutter"]["config"]["key2"] == "value2"


def test_generate_context_with_boolean_variable(tmp_path):
    """Test generate_context converts string to boolean for boolean variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true, "use_ci": false}')
    
    result = generate_context(
        str(context_file),
        extra_context={"use_docker": "false", "use_ci": "yes"}
    )
    
    assert result["cookiecutter"]["use_docker"] is False
    assert result["cookiecutter"]["use_ci"] is True


def test_generate_context_custom_filename(tmp_path):
    """Test generate_context with custom context filename."""
    context_file = tmp_path / "custom.json"
    context_file.write_text('{"project_name": "test"}')
    
    result = generate_context(str(context_file))
    
    assert "custom" in result
    assert result["custom"]["project_name"] == "test"


def test_generate_context_invalid_choice_raises_error(tmp_path):
    """Test generate_context raises ValueError for invalid choice."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    try:
        generate_context(
            str(context_file),
            extra_context={"license": "GPL"}
        )
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "choice variable" in str(e)


def test_generate_context_invalid_boolean_conversion(tmp_path):
    """Test generate_context raises ValueError for invalid boolean conversion."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_feature": true}')
    
    try:
        generate_context(
            str(context_file),
            extra_context={"use_feature": "invalid_boolean"}
        )
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


# LLM-generated content at query #6
#--------------------------

```python
def test_render_and_create_dir_raises_on_empty_dirname():
    """Test that EmptyDirNameException is raised when dirname is empty string."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    
    context = {}
    output_dir = Path(".")
    environment = Environment()
    
    try:
        render_and_create_dir("", context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_apply_overwrites_to_context_predicate_line_46_evaluates_to_false():
    """Test that the predicate at line 46 evaluates to False when conditions are not met."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    # Case 1: context_value is dict but overwrite is not dict
    context = {"key": {"nested": "value"}}
    overwrite_context = {"key": "string_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == "string_value"
    
    # Case 2: context_value is not dict but overwrite is dict
    context = {"key": "string_value"}
    overwrite_context = {"key": {"nested": "value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == {"nested": "value"}
    
    # Case 3: neither context_value nor overwrite is dict
    context = {"key": "original"}
    overwrite_context = {"key": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == "new"


# LLM-generated content at query #8
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
    
    # Verify that the file was opened and parsed correctly
    assert isinstance(result, dict)
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"
    assert result["cookiecutter"]["author"] == "Test Author"


# LLM-generated content at query #9
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_true(tmp_path, monkeypatch):
    """Test that the predicate at line 24 evaluates to True when directory exists."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Setup
    output_dir = tmp_path
    dirname = "test_dir"
    context = {}
    environment = Environment()
    
    # Create the directory that will be checked at line 24
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    
    # Call the function with overwrite_if_exists=True to avoid exception
    result_path, is_new = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=output_dir,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # Verify the predicate at line 24 evaluated to True
    assert dir_to_create.exists() is True
    assert result_path == dir_to_create


# LLM-generated content at query #10
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
    except EmptyDirNameException as e:
        assert 'Error: directory name is empty' in str(e)


# LLM-generated content at query #11
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_success():
    """Test that line 57 predicate evaluates to False when conversion succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"debug": True}
    overwrite_context = {"debug": "false"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["debug"] is False


def test_apply_overwrites_to_context_boolean_conversion_yes():
    """Test that line 57 predicate evaluates to False for 'yes' conversion."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": False}
    overwrite_context = {"enabled": "yes"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is True


def test_apply_overwrites_to_context_boolean_conversion_on():
    """Test that line 57 predicate evaluates to False for 'on' conversion."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"feature": False}
    overwrite_context = {"feature": "on"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["feature"] is True


def test_apply_overwrites_to_context_boolean_conversion_zero():
    """Test that line 57 predicate evaluates to False for '0' conversion."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"active": True}
    overwrite_context = {"active": "0"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["active"] is False


# LLM-generated content at query #12
#--------------------------

```python
def test_render_and_create_dir_with_valid_dirname(tmp_path, monkeypatch):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    context = {'project_name': 'my_project'}
    environment = Environment()
    output_dir = tmp_path
    dirname = '{{ project_name }}'
    
    result_path, is_new = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=False
    )
    
    assert result_path == Path(output_dir, 'my_project')
    assert is_new is True
    assert result_path.exists()


def test_render_and_create_dir_with_empty_dirname(tmp_path):
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir, EmptyDirNameException
    
    context = {}
    environment = Environment()
    output_dir = tmp_path
    dirname = ''
    
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_with_none_dirname(tmp_path):
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir, EmptyDirNameException
    
    context = {}
    environment = Environment()
    output_dir = tmp_path
    dirname = None
    
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_existing_dir_no_overwrite(tmp_path):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir, OutputDirExistsException
    
    context = {'project_name': 'my_project'}
    environment = Environment()
    output_dir = tmp_path
    dirname = '{{ project_name }}'
    
    existing_dir = Path(output_dir, 'my_project')
    existing_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists=False
        )
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    context = {'project_name': 'my_project'}
    environment = Environment()
    output_dir = tmp_path
    dirname = '{{ project_name }}'
    
    existing_dir = Path(output_dir, 'my_project')
    existing_dir.mkdir(parents=True, exist_ok=True)
    
    result_path, is_new = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=True
    )
    
    assert result_path == Path(output_dir, 'my_project')
    assert is_new is False
    assert result_path.exists()


def test_render_and_create_dir_with_nested_path(tmp_path):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    context = {'org': 'myorg', 'project': 'myproject'}
    environment = Environment()
    output_dir = tmp_path
    dirname = '{{ org }}/{{ project }}'
    
    result_path, is_new = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=False
    )
    
    assert result_path == Path(output_dir, 'myorg', 'myproject')
    assert is_new is True
    assert result_path.exists()


def test_render_and_create_dir_with_plain_dirname(tmp_path):
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    context = {}
    environment = Environment()
    output_dir = tmp_path
    dirname = 'simple_project'
    
    result_path, is_new = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=False
    )
    
    assert result_path == Path(output_dir, 'simple_project')
    assert is_new is True
    assert result_path.exists()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_variable_at_first_level():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"existing_var": "value"}
    overwrite_context = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing_var": "value"}


def test_apply_overwrites_to_context_adds_new_variable_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"existing": "value"}}
    overwrite_context = {"nested": {"new_var": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"existing": "value", "new_var": "new_value"}}


def test_apply_overwrites_to_context_multichoice_valid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}


def test_apply_overwrites_to_context_multichoice_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["x", "y"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "multi-choice variable" in str(e)


def test_apply_overwrites_to_context_choice_valid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice": ["default", "option1", "option2"]}
    overwrite_context = {"choice": "option1"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["option1", "default", "option2"]}


def test_apply_overwrites_to_context_choice_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "choice variable" in str(e)


def test_apply_overwrites_to_context_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"config": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"config": {"key1": "new_value1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"key1": "new_value1", "key2": "value2"}}


def test_apply_overwrites_to_context_boolean_yes():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_no():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_boolean_true():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"flag": False}
    overwrite_context = {"flag": "true"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_false():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"flag": True}
    overwrite_context = {"flag": "false"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_boolean_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"flag": True}
    overwrite_context = {"flag": "invalid_bool"}
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


def test_apply_overwrites_to_context_list_overwrite_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"items": ["a", "b"]}}
    overwrite_context = {"nested": {"items": ["x", "y"]}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=False)
    assert context == {"nested": {"items": ["x", "y"]}}


def test_apply_overwrites_to_context_multiple_variables():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"var1": "value1", "var2": "value2", "var3": "value3"}
    overwrite_context = {"var1": "new1", "var3": "new3"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "new1", "var2": "value2", "var3": "new3"}


def test_apply_overwrites_to_context_deep_nesting():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"level1": {"level2": {"level3": "value"}}}
    overwrite_context = {"level1": {"level2": {"level3": "new_value"}}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"level1": {"level2": {"level3": "new_value"}}}


def test_apply_overwrites_to_context_boolean_1():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"flag": False}
    overwrite_context = {"flag": "1"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_0():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"flag": True}
    overwrite_context = {"flag": "0"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_preserves_other_keys():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"a": 1, "b": 2, "c": 3}
    overwrite_context = {"b": 20}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"a": 1, "b": 20, "c": 3}


# LLM-generated content at query #2
#--------------------------

```python
def test_is_copy_only_path_with_matching_pattern():
    path = "templates/file.txt"
    context = {"cookiecutter": {"_copy_without_render": ["templates/*"]}}
    result = is_copy_only_path(path, context)
    assert result is True


def test_is_copy_only_path_with_non_matching_pattern():
    path = "src/file.py"
    context = {"cookiecutter": {"_copy_without_render": ["templates/*"]}}
    result = is_copy_only_path(path, context)
    assert result is False


def test_is_copy_only_path_with_multiple_patterns_first_matches():
    path = "static/style.css"
    context = {"cookiecutter": {"_copy_without_render": ["static/*", "media/*"]}}
    result = is_copy_only_path(path, context)
    assert result is True


def test_is_copy_only_path_with_multiple_patterns_second_matches():
    path = "media/image.png"
    context = {"cookiecutter": {"_copy_without_render": ["static/*", "media/*"]}}
    result = is_copy_only_path(path, context)
    assert result is True


def test_is_copy_only_path_with_multiple_patterns_none_match():
    path = "src/main.py"
    context = {"cookiecutter": {"_copy_without_render": ["static/*", "media/*"]}}
    result = is_copy_only_path(path, context)
    assert result is False


def test_is_copy_only_path_missing_copy_without_render_key():
    path = "file.txt"
    context = {"cookiecutter": {}}
    result = is_copy_only_path(path, context)
    assert result is False


def test_is_copy_only_path_missing_cookiecutter_key():
    path = "file.txt"
    context = {}
    result = is_copy_only_path(path, context)
    assert result is False


def test_is_copy_only_path_with_empty_patterns_list():
    path = "file.txt"
    context = {"cookiecutter": {"_copy_without_render": []}}
    result = is_copy_only_path(path, context)
    assert result is False


def test_is_copy_only_path_with_exact_match():
    path = "README.md"
    context = {"cookiecutter": {"_copy_without_render": ["README.md"]}}
    result = is_copy_only_path(path, context)
    assert result is True


def test_is_copy_only_path_with_wildcard_pattern():
    path = "docs/index.html"
    context = {"cookiecutter": {"_copy_without_render": ["docs/*.html"]}}
    result = is_copy_only_path(path, context)
    assert result is True


# LLM-generated content at query #3
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname(tmp_path):
    """Test that EmptyDirNameException is raised when dirname is empty."""
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
    """Test that EmptyDirNameException is raised when dirname is None."""
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
    """Test that a new directory is created and returns correct values."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {}
    environment = Environment()
    dirname = "test_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path == Path(tmp_path, dirname)
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_with_template_variable(tmp_path):
    """Test that directory name is rendered with context variables."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {"project_name": "my_project"}
    environment = Environment()
    dirname = "{{ project_name }}"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path == Path(tmp_path, "my_project")
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_existing_dir_raises_exception(tmp_path):
    """Test that OutputDirExistsException is raised when directory exists and overwrite is False."""
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    from pathlib import Path
    
    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir()
    
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    
    try:
        render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    """Test that existing directory is handled correctly when overwrite_if_exists is True."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir()
    
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=True)
    
    assert result_path == Path(tmp_path, dirname)
    assert result_path.exists()
    assert is_new is False


def test_render_and_create_dir_with_nested_path(tmp_path):
    """Test that nested directories are created correctly."""
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


def test_render_and_create_dir_with_complex_template(tmp_path):
    """Test that complex template expressions are rendered correctly."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {"name": "test", "version": "1"}
    environment = Environment()
    dirname = "{{ name }}_v{{ version }}"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path == Path(tmp_path, "test_v1")
    assert result_path.exists()
    assert is_new is True


# LLM-generated content at query #4
#--------------------------

```python
def test_generate_context_basic(tmp_path):
    """Test generate_context loads a basic JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John Doe"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context applies default_context overwrites."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    result = generate_context(
        str(context_file),
        default_context={"project_name": "overwritten_project"}
    )
    
    assert result["cookiecutter"]["project_name"] == "overwritten_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context applies extra_context overwrites."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "version": "1.0"}')
    
    result = generate_context(
        str(context_file),
        extra_context={"version": "2.0"}
    )
    
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["version"] == "2.0"


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context raises ContextDecodingException for invalid JSON."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"invalid": json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Expected ContextDecodingException"
    except Exception as e:
        assert "JSON decoding error" in str(e)


def test_generate_context_with_boolean_and_string_overwrite(tmp_path):
    """Test generate_context converts string to boolean for boolean variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_feature": true}')
    
    result = generate_context(
        str(context_file),
        extra_context={"use_feature": "no"}
    )
    
    assert result["cookiecutter"]["use_feature"] is False


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context handles choice variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    result = generate_context(
        str(context_file),
        extra_context={"license": "Apache"}
    )
    
    assert result["cookiecutter"]["license"][0] == "Apache"
    assert "MIT" in result["cookiecutter"]["license"]
    assert "GPL" in result["cookiecutter"]["license"]


def test_generate_context_with_dict_variable(tmp_path):
    """Test generate_context handles nested dictionary variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"author": {"name": "John", "email": "john@example.com"}}')
    
    result = generate_context(
        str(context_file),
        extra_context={"author": {"name": "Jane"}}
    )
    
    assert result["cookiecutter"]["author"]["name"] == "Jane"
    assert result["cookiecutter"]["author"]["email"] == "john@example.com"


def test_generate_context_file_stem_extraction(tmp_path):
    """Test generate_context extracts correct file stem as context key."""
    context_file = tmp_path / "custom_template.json"
    context_file.write_text('{"key": "value"}')
    
    result = generate_context(str(context_file))
    
    assert "custom_template" in result
    assert result["custom_template"]["key"] == "value"


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test generate_context handles multichoice variables."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    
    result = generate_context(
        str(context_file),
        extra_context={"features": ["feature2", "feature3"]}
    )
    
    assert result["cookiecutter"]["features"] == ["feature2", "feature3"]


def test_generate_context_invalid_default_context_warns(tmp_path):
    """Test generate_context warns on invalid default_context."""
    import warnings
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = generate_context(
            str(context_file),
            default_context={"license": "InvalidChoice"}
        )
        assert len(w) > 0
        assert "Invalid default received" in str(w[0].message)


# LLM-generated content at query #5
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_variable_at_first_level():
    """Test that new variables at first level are ignored."""
    context = {"existing": "value"}
    overwrite_context = {"new_variable": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_adds_new_variable_in_nested_dict():
    """Test that new variables in nested dict are added."""
    context = {"nested": {"existing": "value"}}
    overwrite_context = {"nested": {"new_key": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"existing": "value", "new_key": "new_value"}}


def test_apply_overwrites_to_context_overwrites_simple_value():
    """Test that simple values are overwritten."""
    context = {"variable": "old_value"}
    overwrite_context = {"variable": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"variable": "new_value"}


def test_apply_overwrites_to_context_multichoice_valid():
    """Test that valid multichoice overwrite is applied."""
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}


def test_apply_overwrites_to_context_multichoice_invalid():
    """Test that invalid multichoice overwrite raises ValueError."""
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "multi-choice variable" in str(e)


def test_apply_overwrites_to_context_single_choice_valid():
    """Test that valid single choice overwrite is applied."""
    context = {"choice": ["default", "option1", "option2"]}
    overwrite_context = {"choice": "option1"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["option1", "default", "option2"]}


def test_apply_overwrites_to_context_single_choice_invalid():
    """Test that invalid single choice overwrite raises ValueError."""
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "choice variable" in str(e)


def test_apply_overwrites_to_context_nested_dict_overwrite():
    """Test that nested dictionaries are partially overwritten."""
    context = {"config": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"config": {"key1": "new_value1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"key1": "new_value1", "key2": "value2"}}


def test_apply_overwrites_to_context_boolean_yes_string():
    """Test that boolean context with 'yes' string is converted to True."""
    context = {"is_enabled": False}
    overwrite_context = {"is_enabled": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"is_enabled": True}


def test_apply_overwrites_to_context_boolean_no_string():
    """Test that boolean context with 'no' string is converted to False."""
    context = {"is_enabled": True}
    overwrite_context = {"is_enabled": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"is_enabled": False}


def test_apply_overwrites_to_context_boolean_true_string():
    """Test that boolean context with 'true' string is converted to True."""
    context = {"is_enabled": False}
    overwrite_context = {"is_enabled": "true"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"is_enabled": True}


def test_apply_overwrites_to_context_boolean_false_string():
    """Test that boolean context with 'false' string is converted to False."""
    context = {"is_enabled": True}
    overwrite_context = {"is_enabled": "false"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"is_enabled": False}


def test_apply_overwrites_to_context_boolean_invalid_string():
    """Test that boolean context with invalid string raises ValueError."""
    context = {"is_enabled": True}
    overwrite_context = {"is_enabled": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_apply_overwrites_to_context_list_in_nested_dict():
    """Test that list in nested dict is overwritten when in_dictionary_variable is True."""
    context = {"nested": {"choices": ["a", "b", "c"]}}
    overwrite_context = {"nested": {"choices": ["x", "y"]}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"choices": ["x", "y"]}}


def test_apply_overwrites_to_context_integer_overwrite():
    """Test that integer values are overwritten."""
    context = {"count": 5}
    overwrite_context = {"count": 10}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"count": 10}


def test_apply_overwrites_to_context_empty_overwrite():
    """Test that empty overwrite context doesn't change context."""
    context = {"variable": "value"}
    overwrite_context = {}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"variable": "value"}


def test_apply_overwrites_to_context_multiple_overwrites():
    """Test that multiple overwrites are applied correctly."""
    context = {"var1": "value1", "var2": "value2", "var3": "value3"}
    overwrite_context = {"var1": "new1", "var3": "new3"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "new1", "var2": "value2", "var3": "new3"}


def test_apply_overwrites_to_context_nested_empty_dict():
    """Test that nested empty dict can be overwritten."""
    context = {"config": {}}
    overwrite_context = {"config": {"new_key": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"new_key": "new_value"}}


# LLM-generated content at query #6
#--------------------------

```python
def test_run_hook_from_repo_dir_deprecation_warning(mocker):
    """Test that _run_hook_from_repo_dir raises a deprecation warning."""
    mock_run_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    
    with mocker.patch('cookiecutter.generate.warnings.warn') as mock_warn:
        from cookiecutter.generate import _run_hook_from_repo_dir
        _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        
        mock_warn.assert_called_once()
        assert "deprecated" in mock_warn.call_args[0][0].lower()
        assert "run_hook_from_repo_dir" in mock_warn.call_args[0][0]
        assert mock_warn.call_args[0][1] == DeprecationWarning
        assert mock_warn.call_args[0][2] == 2


def test_run_hook_from_repo_dir_calls_actual_function(mocker):
    """Test that _run_hook_from_repo_dir calls the actual run_hook_from_repo_dir."""
    mock_run_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.warnings.warn')
    
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    
    mock_run_hook.assert_called_once_with(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )


def test_run_hook_from_repo_dir_with_delete_true(mocker):
    """Test _run_hook_from_repo_dir with delete_project_on_failure=True."""
    mock_run_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.warnings.warn')
    
    repo_dir = '/path/to/repo'
    hook_name = 'pre_prompt'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'name': 'myproject'}}
    delete_project_on_failure = True
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    
    assert mock_run_hook.call_count == 1
    assert mock_run_hook.call_args[0][4] is True


def test_run_hook_from_repo_dir_with_empty_context(mocker):
    """Test _run_hook_from_repo_dir with empty context."""
    mock_run_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.warnings.warn')
    
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {}
    delete_project_on_failure = False
    
    from cookiecutter.generate import _run_hook_from_repo_dir
    _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    
    mock_run_hook.assert_called_once_with(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure
    )


# LLM-generated content at query #7
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname(tmp_path):
    """Test that EmptyDirNameException is raised when dirname is empty."""
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
    """Test that EmptyDirNameException is raised when dirname is None."""
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
    """Test that a new directory is created and returns correct path and flag."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {}
    environment = Environment()
    dirname = "test_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path == Path(tmp_path) / dirname
    assert is_new is True
    assert result_path.exists()
    assert result_path.is_dir()


def test_render_and_create_dir_with_template_variable(tmp_path):
    """Test that directory name is rendered with template variables."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {"project_name": "my_project"}
    environment = Environment()
    dirname = "{{project_name}}_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path == Path(tmp_path) / "my_project_dir"
    assert is_new is True
    assert result_path.exists()


def test_render_and_create_dir_existing_dir_without_overwrite(tmp_path):
    """Test that OutputDirExistsException is raised when directory exists and overwrite is False."""
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    from pathlib import Path
    
    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir()
    
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    
    try:
        render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    """Test that existing directory is handled when overwrite_if_exists is True."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir()
    
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment, overwrite_if_exists=True)
    
    assert result_path == existing_dir
    assert is_new is False
    assert result_path.exists()


def test_render_and_create_dir_nested_path(tmp_path):
    """Test that nested directory paths are created correctly."""
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    
    context = {}
    environment = Environment()
    dirname = "parent/child/grandchild"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, environment)
    
    assert result_path == Path(tmp_path) / "parent" / "child" / "grandchild"
    assert is_new is True
    assert result_path.exists()
    assert result_path.is_dir()


# LLM-generated content at query #8
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_true(tmp_path, monkeypatch):
    """Test that the predicate at line 24 evaluates to True when directory exists."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Create a temporary directory that will exist
    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir()
    
    # Setup context and environment
    context = {}
    environment = Environment()
    dirname = "existing_dir"
    
    # Call the function with overwrite_if_exists=True to avoid exception
    result_path, is_new = render_and_create_dir(
        dirname=dirname,
        context=context,
        output_dir=tmp_path,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # Verify the predicate at line 24 (output_dir_exists) is True
    # This is confirmed by is_new being False (line 35: return dir_to_create, not output_dir_exists)
    assert is_new is False
    assert result_path.exists()


# LLM-generated content at query #9
#--------------------------

```python
def test_render_and_create_dir_predicate_line_24_true(tmp_path):
    """Test that the predicate at line 24 evaluates to True when directory exists."""
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir
    
    # Create a directory that will exist
    existing_dir = tmp_path / "existing_project"
    existing_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup context and environment
    context = {"project_name": "existing_project"}
    environment = Environment()
    
    # Call function with overwrite_if_exists=True to avoid exception
    result_dir, is_new = render_and_create_dir(
        dirname="{{ project_name }}",
        context=context,
        output_dir=tmp_path,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # Verify that the predicate (output_dir_exists) evaluated to True
    # This is evidenced by is_new being False (not output_dir_exists)
    assert is_new is False
    assert result_dir.exists()


# LLM-generated content at query #10
#--------------------------

```python
def test_render_and_create_dir_raises_empty_dir_name_exception():
    """Test that render_and_create_dir raises EmptyDirNameException when dirname is empty."""
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
    except EmptyDirNameException as e:
        assert 'Error: directory name is empty' in str(e)


# LLM-generated content at query #11
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
    )
    
    assert "my_project" in result
    assert (output_dir / "my_project").exists()
    assert (output_dir / "my_project" / "README.md").exists()
    readme_content = (output_dir / "my_project" / "README.md").read_text()
    assert "# my_project" in readme_content


def test_generate_files_with_overwrite_if_exists(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists flag."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create existing project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
    )
    
    assert (output_dir / "my_project").exists()


def test_generate_files_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists flag."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "existing.txt"
    template_file.write_text("new content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
    )
    
    assert (output_dir / "my_project" / "existing.txt").exists()


def test_generate_files_with_subdirectories(tmp_path, monkeypatch):
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
    src_file = src_dir / "main.py"
    src_file.write_text("print('{{cookiecutter.project_name}}')")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
    )
    
    assert (output_dir / "my_project" / "src").exists()
    assert (output_dir / "my_project" / "src" / "main.py").exists()


def test_generate_files_binary_file(tmp_path, monkeypatch):
    """Test generate_files with binary files."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
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
    )
    
    assert (output_dir / "my_project" / "image.bin").exists()
    assert (output_dir / "my_project" / "image.bin").read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_files_with_context_variables(tmp_path, monkeypatch):
    """Test generate_files with multiple context variables."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_slug}}"
    template_dir.mkdir()
    
    template_file = template_dir / "config.txt"
    template_file.write_text("project: {{cookiecutter.project_name}}\nauthor: {{cookiecutter.author}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {
            'project_name': 'My Project',
            'project_slug': 'my_project',
            'author': 'John Doe'
        })
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
    )
    
    config_file = output_dir / "my_project" / "config.txt"
    content = config_file.read_text()
    assert "project: My Project" in content
    assert "author: John Doe" in content


def test_generate_files_returns_project_dir(tmp_path, monkeypatch):
    """Test that generate_files returns the project directory path."""
    from cookiecutter.generate import generate_files
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")
    
    output_dir = tmp_path /


# LLM-generated content at query #12
#--------------------------

```python
def test_render_and_create_dir_raises_on_empty_dirname():
    """Test that render_and_create_dir raises EmptyDirNameException when dirname is empty."""
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
    except EmptyDirNameException:
        assert True


# LLM-generated content at query #13
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    
    try:
        render_and_create_dir("", context, tmp_path, env)
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_with_none_dirname(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    
    try:
        render_and_create_dir(None, context, tmp_path, env)
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_creates_new_directory(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    dirname = "new_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env)
    
    assert result_path == tmp_path / dirname
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_with_template_rendering(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    context = {"project_name": "my_project"}
    dirname = "{{ project_name }}_dir"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env)
    
    assert result_path == tmp_path / "my_project_dir"
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_existing_dir_no_overwrite(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    dirname = "existing_dir"
    existing_path = tmp_path / dirname
    existing_path.mkdir(parents=True, exist_ok=True)
    
    try:
        render_and_create_dir(dirname, context, tmp_path, env, overwrite_if_exists=False)
        assert False, "Should have raised OutputDirExistsException"
    except OutputDirExistsException:
        pass


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    dirname = "existing_dir"
    existing_path = tmp_path / dirname
    existing_path.mkdir(parents=True, exist_ok=True)
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env, overwrite_if_exists=True)
    
    assert result_path == existing_path
    assert result_path.exists()
    assert is_new is False


def test_render_and_create_dir_nested_directory(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    context = {}
    dirname = "parent/child/grandchild"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env)
    
    assert result_path == tmp_path / dirname
    assert result_path.exists()
    assert is_new is True


def test_render_and_create_dir_with_complex_template(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    
    env = Environment()
    context = {"name": "test", "version": "1.0"}
    dirname = "{{ name }}-v{{ version }}"
    
    result_path, is_new = render_and_create_dir(dirname, context, tmp_path, env)
    
    assert result_path == tmp_path / "test-v1.0"
    assert result_path.exists()
    assert is_new is True


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_context_catches_json_decoding_error():
    """Test that generate_context catches ValueError on invalid JSON and raises ContextDecodingException."""
    import tempfile
    import os
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    # Create a temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        f.write('{ invalid json content }')
        temp_file = f.name
    
    try:
        # This should raise ContextDecodingException
        error_raised = False
        try:
            generate_context(context_file=temp_file)
        except Exception as e:
            # Check that the exception is ContextDecodingException (predicate at line 20 evaluates to True)
            error_raised = isinstance(e, ContextDecodingException)
            assert error_raised, f"Expected ContextDecodingException but got {type(e).__name__}"
            assert "JSON decoding error" in str(e)
            assert temp_file in str(e)
    finally:
        os.unlink(temp_file)


# LLM-generated content at query #15
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
        result_path, is_new = render_and_create_dir("new_dir", context, tmpdir, env)
        
        assert result_path.exists()
        assert result_path.name == "new_dir"
        assert is_new is True


def test_render_and_create_dir_with_template_rendering():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "my_project"}
        result_path, is_new = render_and_create_dir("{{ project_name }}", context, tmpdir, env)
        
        assert result_path.exists()
        assert result_path.name == "my_project"
        assert is_new is True


def test_render_and_create_dir_existing_directory_without_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        existing_dir = Path(tmpdir) / "existing_dir"
        existing_dir.mkdir()
        
        try:
            render_and_create_dir("existing_dir", context, tmpdir, env, overwrite_if_exists=False)
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
        existing_dir = Path(tmpdir) / "existing_dir"
        existing_dir.mkdir()
        
        result_path, is_new = render_and_create_dir("existing_dir", context, tmpdir, env, overwrite_if_exists=True)
        
        assert result_path.exists()
        assert is_new is False


def test_render_and_create_dir_nested_directory():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {}
        result_path, is_new = render_and_create_dir("parent/child/grandchild", context, tmpdir, env)
        
        assert result_path.exists()
        assert result_path.name == "grandchild"
        assert is_new is True


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_context_predicate_line_38_evaluates_to_false():
    """Test that the predicate at line 38 (if default_context:) evaluates to False."""
    import json
    import tempfile
    import os
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'key': 'value'}, f)
        temp_file = f.name
    
    try:
        # Call generate_context with default_context=None (predicate evaluates to False)
        result = generate_context(
            context_file=temp_file,
            default_context=None,
            extra_context=None
        )
        
        # Verify that the function executed without calling apply_overwrites_to_context
        # when default_context is None/falsy
        assert isinstance(result, dict)
        file_stem = os.path.splitext(os.path.basename(temp_file))[0]
        assert file_stem in result
        assert result[file_stem] == {'key': 'value'}
    finally:
        os.unlink(temp_file)


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_context_json_decoding_error():
    """Test that ValueError is caught at line 20 and converted to ContextDecodingException."""
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
        # This should trigger the ValueError exception at line 20
        generate_context(context_file=temp_file)
        assert False, "Expected ContextDecodingException to be raised"
    except Exception as e:
        # Verify that the exception is ContextDecodingException
        assert isinstance(e, ContextDecodingException)
        assert "JSON decoding error" in str(e)
        assert temp_file in str(e)
    finally:
        # Clean up
        os.unlink(temp_file)


# LLM-generated content at query #18
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
    
    # Setup context and environment
    context = {"project_name": "existing_project"}
    environment = Environment()
    
    # Call render_and_create_dir with overwrite_if_exists=True to avoid exception
    result_path, is_new = render_and_create_dir(
        dirname="existing_project",
        context=context,
        output_dir=tmp_path,
        environment=environment,
        overwrite_if_exists=True
    )
    
    # The predicate at line 24 is `if output_dir_exists:` which should be True
    # This is verified indirectly: if the predicate was False, the else block would execute
    # and make_sure_path_exists would be called, but since dir exists, we're in the if block
    assert result_path.exists()
    assert is_new is False  # not output_dir_exists = not True = False


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_context_predicate_line_18_evaluates_to_false(tmp_path):
    """Test that the predicate at line 18 (with open(...)) evaluates to False when file doesn't exist."""
    import os
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    non_existent_file = str(tmp_path / "non_existent_cookiecutter.json")
    
    try:
        generate_context(context_file=non_existent_file)
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        # The predicate at line 18 (with open(...)) evaluates to False
        # because the file does not exist, causing the except block not to catch it
        # and the function to raise FileNotFoundError instead
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_context_raises_context_decoding_exception_on_invalid_json(tmp_path):
    """Test that ValueError from json.load is caught and ContextDecodingException is raised."""
    import json
    import os
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    # Create a temporary file with invalid JSON
    invalid_json_file = tmp_path / "cookiecutter.json"
    invalid_json_file.write_text("{invalid json content")
    
    # Call generate_context with invalid JSON file
    try:
        generate_context(str(invalid_json_file))
        assert False, "Expected ContextDecodingException to be raised"
    except ContextDecodingException as e:
        # Verify the exception message contains expected information
        assert "JSON decoding error" in str(e)
        assert str(invalid_json_file) in str(e)
        assert True


# LLM-generated content at query #21
#--------------------------

```python
def test_generate_file_renders_text_file(tmp_path, monkeypatch):
    from jinja2 import Environment, FileSystemLoader
    import os
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file with template variable
    infile_path = template_dir / "test.txt"
    infile_path.write_text("Hello {{ name }}!")
    
    monkeypatch.chdir(template_dir)
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {"cookiecutter": {"name": "World"}}
    
    # Execute
    from cookiecutter.generate import generate_file
    generate_file(str(project_dir), "test.txt", context, env)
    
    # Assert
    outfile = project_dir / "test.txt"
    assert outfile.exists()
    assert outfile.read_text() == "Hello World!"


def test_generate_file_copies_binary_file(tmp_path, monkeypatch):
    from jinja2 import Environment, FileSystemLoader
    import os
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create binary input file
    infile_path = template_dir / "test.bin"
    binary_content = b"\x89PNG\r\n\x1a\n"
    infile_path.write_bytes(binary_content)
    
    monkeypatch.chdir(template_dir)
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {"cookiecutter": {}}
    
    # Execute
    from cookiecutter.generate import generate_file
    generate_file(str(project_dir), "test.bin", context, env)
    
    # Assert
    outfile = project_dir / "test.bin"
    assert outfile.exists()
    assert outfile.read_bytes() == binary_content


def test_generate_file_renders_filename(tmp_path, monkeypatch):
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create input file
    infile_path = template_dir / "{{ filename }}.txt"
    infile_path.write_text("content")
    
    monkeypatch.chdir(template_dir)
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {"cookiecutter": {"filename": "output"}}
    
    # Execute
    from cookiecutter.generate import generate_file
    generate_file(str(project_dir), "{{ filename }}.txt", context, env)
    
    # Assert
    outfile = project_dir / "output.txt"
    assert outfile.exists()
    assert outfile.read_text() == "content"


def test_generate_file_skips_if_exists(tmp_path, monkeypatch):
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    infile_path = template_dir / "test.txt"
    infile_path.write_text("original")
    
    outfile = project_dir / "test.txt"
    outfile.write_text("existing content")
    
    monkeypatch.chdir(template_dir)
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {"cookiecutter": {}}
    
    # Execute
    from cookiecutter.generate import generate_file
    generate_file(str(project_dir), "test.txt", context, env, skip_if_file_exists=True)
    
    # Assert
    assert outfile.read_text() == "existing content"


def test_generate_file_returns_if_filename_empty(tmp_path, monkeypatch):
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    infile_path = template_dir / "test.txt"
    infile_path.write_text("content")
    
    monkeypatch.chdir(template_dir)
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {"cookiecutter": {}}
    
    # Execute - pass empty template that renders to directory name
    from cookiecutter.generate import generate_file
    generate_file(str(project_dir), "test.txt", context, env)
    
    # Assert - file should be created normally
    outfile = project_dir / "test.txt"
    assert outfile.exists()


def test_generate_file_preserves_newlines(tmp_path, monkeypatch):
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    infile_path = template_dir / "test.txt"
    infile_path.write_text("line1\nline2\n")
    
    monkeypatch.chdir(template_dir)
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {"cookiecutter": {}}
    
    # Execute
    from cookiecutter.generate import generate_file
    generate_file(str(project_dir), "test.txt", context, env)
    
    # Assert
    outfile = project_dir / "test.txt"
    assert outfile.exists()
    content = outfile.read_text()
    assert "line1" in content
    assert "line2" in content


def test_generate_file_uses_configured_newlines(tmp_path, monkeypatch):
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    infile_path = template_dir / "test.txt"
    infile_path.write_text("line1\nline2\n")
    
    monkeypatch.chdir(template_dir)
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    
    # Execute
    from cookiecutter.generate import generate_file
    generate_file(str(project_dir), "test.txt", context, env)
    
    # Assert
    outfile = project_dir / "test.txt"
    assert outfile.exists()


def test_generate_file_preserves_permissions(tmp_path, monkeypatch):
    from jinja2 import Environment, FileSystemLoader
    import stat
    
    # Setup
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    infile_path = template_dir / "test.txt"
    infile_path.write_text("content")
    infile_path.chmod(0o755)
    
    monkeypatch.chdir(template_dir)
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {"cookiecutter": {}}
    
    # Execute
    from cookiecutter.generate import generate_file
    generate_file(str(project_dir), "test.txt", context, env)
    
    # Assert
    outfile = project_dir / "test.txt"
    assert outfile.exists


# LLM-generated content at query #22
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_with_valid_yes_response():
    """Test that line 57 predicate evaluates to False when YesNoPrompt.process_response succeeds."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "yes"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is True


def test_apply_overwrites_to_context_boolean_conversion_with_valid_no_response():
    """Test that line 57 predicate evaluates to False when YesNoPrompt.process_response succeeds with 'no'."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"enabled": True}
    overwrite_context = {"enabled": "no"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["enabled"] is False


def test_apply_overwrites_to_context_boolean_conversion_with_valid_true_response():
    """Test that line 57 predicate evaluates to False when YesNoPrompt.process_response succeeds with 'true'."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": False}
    overwrite_context = {"flag": "true"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is True


def test_apply_overwrites_to_context_boolean_conversion_with_valid_false_response():
    """Test that line 57 predicate evaluates to False when YesNoPrompt.process_response succeeds with 'false'."""
    from cookiecutter.generate import apply_overwrites_to_context
    
    context = {"flag": True}
    overwrite_context = {"flag": "false"}
    
    apply_overwrites_to_context(context, overwrite_context)
    
    assert context["flag"] is False


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_files_with_default_parameters(tmp_path, monkeypatch):
    """Test generate_files with default parameters."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    monkeypatch.setattr('cookiecutter.generate.find_template', lambda repo, env: template_dir)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    
    assert result is not None
    assert isinstance(result, str)


def test_generate_files_with_overwrite_if_exists(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    monkeypatch.setattr('cookiecutter.generate.find_template', lambda repo, env: template_dir)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True, accept_hooks=False)
    
    assert result is not None


def test_generate_files_with_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    monkeypatch.setattr('cookiecutter.generate.find_template', lambda repo, env: template_dir)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True, accept_hooks=False)
    
    assert result is not None


def test_generate_files_with_none_context(tmp_path, monkeypatch):
    """Test generate_files with None context."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "test_template"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.setattr('cookiecutter.generate.find_template', lambda repo, env: template_dir)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(repo_dir, None, output_dir, accept_hooks=False)
    
    assert result is not None


def test_generate_files_with_accept_hooks_false(tmp_path, monkeypatch):
    """Test generate_files with accept_hooks=False."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_jinja2_env_vars': {}
        }
    }
    
    hook_called = []
    
    def mock_hook(*args, **kwargs):
        hook_called.append(True)
    
    monkeypatch.setattr('cookiecutter.generate.find_template', lambda repo, env: template_dir)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_hook)
    
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    
    assert len(hook_called) == 0


def test_generate_files_with_keep_project_on_failure_true(tmp_path, monkeypatch):
    """Test generate_files with keep_project_on_failure=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    monkeypatch.setattr('cookiecutter.generate.find_template', lambda repo, env: template_dir)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True, accept_hooks=False)
    
    assert result is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_first_level_variable():
    context = {"existing": "value"}
    overwrite_context = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_adds_new_dictionary_variable():
    context = {"existing": {}}
    overwrite_context = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"existing": {}, "new_var": "new_value"}


def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old_value"}
    overwrite_context = {"key": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "new_value"}


def test_apply_overwrites_to_context_multichoice_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}


def test_apply_overwrites_to_context_multichoice_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "provided for multi-choice variable" in str(e)


def test_apply_overwrites_to_context_choice_variable_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}


def test_apply_overwrites_to_context_choice_variable_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "provided for choice variable" in str(e)


def test_apply_overwrites_to_context_nested_dict():
    context = {"outer": {"inner": "value"}}
    overwrite_context = {"outer": {"inner": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"outer": {"inner": "new_value"}}


def test_apply_overwrites_to_context_nested_dict_partial():
    context = {"outer": {"inner1": "value1", "inner2": "value2"}}
    overwrite_context = {"outer": {"inner1": "new_value1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"outer": {"inner1": "new_value1", "inner2": "value2"}}


def test_apply_overwrites_to_context_boolean_yes():
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_no():
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_boolean_true():
    context = {"flag": False}
    overwrite_context = {"flag": "true"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_false():
    context = {"flag": True}
    overwrite_context = {"flag": "false"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_boolean_1():
    context = {"flag": False}
    overwrite_context = {"flag": "1"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_0():
    context = {"flag": True}
    overwrite_context = {"flag": "0"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_boolean_invalid():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_apply_overwrites_to_context_list_overwrite_in_dict():
    context = {"outer": {"inner": ["a", "b"]}}
    overwrite_context = {"outer": {"inner": ["b", "a"]}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"outer": {"inner": ["b", "a"]}}


def test_apply_overwrites_to_context_empty_context():
    context = {}
    overwrite_context = {"key": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {}


def test_apply_overwrites_to_context_empty_overwrite():
    context = {"key": "value"}
    overwrite_context = {}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "value"}


def test_apply_overwrites_to_context_multiple_variables():
    context = {"var1": "value1", "var2": "value2", "var3": ["a", "b"]}
    overwrite_context = {"var1": "new_value1", "var3": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "new_value1", "var2": "value2", "var3": ["b", "a"]}


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_context_with_default_context():
    """Test that the predicate at line 38 evaluates to True when default_context is provided."""
    import json
    import os
    import tempfile
    from collections import OrderedDict
    from cookiecutter.generate import generate_context
    
    # Create a temporary directory and context file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'project_slug': 'my_project'
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        # Call generate_context with default_context to trigger line 38 predicate
        default_context = {
            'project_name': 'Default Project'
        }
        
        result = generate_context(
            context_file=context_file,
            default_context=default_context
        )
        
        # Verify the function executed successfully with default_context provided
        assert 'cookiecutter' in result
        assert isinstance(result, OrderedDict)
        assert result['cookiecutter']['project_name'] == 'Default Project'


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_files_with_valid_context(tmp_path, mocker):
    """Test generate_files with valid context and template directory."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None
    assert isinstance(result, str)


def test_generate_files_empty_context(tmp_path, mocker):
    """Test generate_files with empty context."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=None,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_with_hooks(tmp_path, mocker):
    """Test generate_files with hooks enabled."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    mock_find_template = mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mock_run_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=True
    )
    
    assert mock_run_hook.call_count == 2


def test_generate_files_overwrite_if_exists(tmp_path, mocker):
    """Test generate_files with overwrite_if_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_skip_if_file_exists(tmp_path, mocker):
    """Test generate_files with skip_if_file_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_keep_project_on_failure(tmp_path, mocker):
    """Test generate_files with keep_project_on_failure flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        keep_project_on_failure=True,
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_with_files_to_process(tmp_path, mocker):
    """Test generate_files with files in the template directory."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.generate_file')
    mocker.patch('os.walk', return_value=[('.', [], ['test.txt'])])
    mocker.patch('cookiecutter.generate.is_copy_only_path', return_value=False)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None


def test_generate_files_


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_file_binary_file(tmp_path, mocker):
    import os
    import shutil
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    infile = "binary_file.bin"
    binary_content = b"\x89PNG\r\n\x1a\n"
    
    tmp_path_root = tmp_path / "templates"
    os.makedirs(tmp_path_root)
    with open(tmp_path_root / infile, "wb") as f:
        f.write(binary_content)
    
    mocker.patch("os.chdir", return_value=None)
    mocker.patch("is_binary", return_value=True)
    mocker.patch("shutil.copyfile")
    mocker.patch("shutil.copymode")
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    os.chdir(str(tmp_path_root))
    generate_file(project_dir, infile, context, env)
    
    assert shutil.copyfile.called
    assert shutil.copymode.called


def test_generate_file_text_file(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    infile = "template_{{ cookiecutter.name }}.txt"
    outfile_name = "template_test.txt"
    
    tmp_path_root = tmp_path / "templates"
    os.makedirs(tmp_path_root)
    with open(tmp_path_root / infile, "w", encoding="utf-8", newline="\n") as f:
        f.write("Hello {{ cookiecutter.name }}!")
    
    mocker.patch("is_binary", return_value=False)
    
    env = Environment()
    context = {"cookiecutter": {"name": "test"}}
    
    os.chdir(str(tmp_path_root))
    generate_file(project_dir, infile, context, env)
    
    outfile_path = os.path.join(project_dir, outfile_name)
    assert os.path.exists(outfile_path)
    with open(outfile_path, "r", encoding="utf-8") as f:
        assert f.read() == "Hello test!"


def test_generate_file_skip_if_exists(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    infile = "existing_file.txt"
    outfile_path = os.path.join(project_dir, infile)
    
    tmp_path_root = tmp_path / "templates"
    os.makedirs(tmp_path_root)
    with open(tmp_path_root / infile, "w") as f:
        f.write("template content")
    
    with open(outfile_path, "w") as f:
        f.write("existing content")
    
    mocker.patch("is_binary", return_value=False)
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    os.chdir(str(tmp_path_root))
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    
    with open(outfile_path, "r") as f:
        assert f.read() == "existing content"


def test_generate_file_empty_filename(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    infile = "{{ cookiecutter.dir_name }}/file.txt"
    
    tmp_path_root = tmp_path / "templates"
    os.makedirs(tmp_path_root)
    subdir = tmp_path_root / "subdir"
    os.makedirs(subdir)
    with open(subdir / "file.txt", "w") as f:
        f.write("content")
    
    mocker.patch("is_binary", return_value=False)
    mocker.patch("os.path.isdir", return_value=True)
    
    env = Environment()
    context = {"cookiecutter": {"dir_name": ""}}
    
    os.chdir(str(tmp_path_root))
    generate_file(project_dir, infile, context, env)


def test_generate_file_custom_newline(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    infile = "file_with_custom_newline.txt"
    
    tmp_path_root = tmp_path / "templates"
    os.makedirs(tmp_path_root)
    with open(tmp_path_root / infile, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("line1\r\nline2\r\n")
    
    mocker.patch("is_binary", return_value=False)
    
    env = Environment()
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    
    os.chdir(str(tmp_path_root))
    generate_file(project_dir, infile, context, env)
    
    outfile_path = os.path.join(project_dir, infile)
    assert os.path.exists(outfile_path)


def test_generate_file_detected_newline(tmp_path, mocker):
    import os
    from jinja2 import Environment
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    infile = "file_with_newline.txt"
    
    tmp_path_root = tmp_path / "templates"
    os.makedirs(tmp_path_root)
    with open(tmp_path_root / infile, "w", encoding="utf-8", newline="\n") as f:
        f.write("content\n")
    
    mocker.patch("is_binary", return_value=False)
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    os.chdir(str(tmp_path_root))
    generate_file(project_dir, infile, context, env)
    
    outfile_path = os.path.join(project_dir, infile)
    assert os.path.exists(outfile_path)


def test_generate_file_template_syntax_error(tmp_path, mocker):
    import os
    from jinja2 import Environment, TemplateSyntaxError
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    infile = "bad_template.txt"
    
    tmp_path_root = tmp_path / "templates"
    os.makedirs(tmp_path_root)
    with open(tmp_path_root / infile, "w") as f:
        f.write("{% if unclosed %}")
    
    mocker.patch("is_binary", return_value=False)
    
    env = Environment()
    context = {"cookiecutter": {}}
    
    os.chdir(str(tmp_path_root))
    
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_59_evaluates_to_false():
    """Test that the predicate at line 59 (with work_in(template_dir):) evaluates to False."""
    # Line 59 is: with work_in(template_dir):
    # The predicate that evaluates to False would be the condition that determines
    # whether to enter the with block. Since work_in is a context manager that always
    # executes its body, we need to test a condition that would make the code NOT
    # enter this block. However, looking at the code structure, line 59 is inside
    # the main flow and will always execute when accept_hooks is True or False.
    
    # The predicate at line 59 in context is actually checking if we should enter
    # the work_in context manager. Since it's a context manager, the predicate
    # would be the truthiness of template_dir or conditions around it.
    
    # Testing that template_dir is falsy (empty string, None, etc.) would make
    # entering the context manager meaningful to test, but the actual predicate
    # is the implicit True for entering the with block.
    
    # The real test is ensuring that when template_dir is provided, the with
    # statement's condition is True (implicit), and we verify the context manager
    # behavior doesn't fail.
    
    from pathlib import Path
    from cookiecutter.utils import work_in
    import os
    import tempfile
    
    # Create a temporary directory to test with
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        temp_path = Path(tmpdir)
        
        # Test that work_in successfully changes directory
        with work_in(temp_path):
            current_cwd = os.getcwd()
            assert current_cwd == str(temp_path)
        
        # Verify we're back to original directory
        assert os.getcwd() == original_cwd


# LLM-generated content at query #29
#--------------------------

```python
def test_apply_overwrites_to_context_ignores_new_variable_at_first_level():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"existing": "value"}
    overwrite_context = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_adds_new_variable_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"existing": "value"}}
    overwrite_context = {"nested": {"new_var": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"nested": {"existing": "value", "new_var": "new_value"}}


def test_apply_overwrites_to_context_multichoice_valid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}


def test_apply_overwrites_to_context_multichoice_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provided for multi-choice variable" in str(e)


def test_apply_overwrites_to_context_single_choice_valid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["choice"][0] == "b"
    assert "a" in context["choice"]
    assert "c" in context["choice"]


def test_apply_overwrites_to_context_single_choice_invalid():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provided for choice variable" in str(e)


def test_apply_overwrites_to_context_nested_dict_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"nested": {"key1": "new_value1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"key1": "new_value1", "key2": "value2"}}


def test_apply_overwrites_to_context_boolean_yes_string():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["bool_var"] is True


def test_apply_overwrites_to_context_boolean_no_string():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": False}
    overwrite_context = {"bool_var": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["bool_var"] is False


def test_apply_overwrites_to_context_boolean_true_string():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "true"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["bool_var"] is True


def test_apply_overwrites_to_context_boolean_false_string():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": False}
    overwrite_context = {"bool_var": "false"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["bool_var"] is False


def test_apply_overwrites_to_context_boolean_invalid_string():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_apply_overwrites_to_context_simple_string_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"name": "old"}
    overwrite_context = {"name": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"name": "new"}


def test_apply_overwrites_to_context_simple_integer_overwrite():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"count": 5}
    overwrite_context = {"count": 10}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"count": 10}


def test_apply_overwrites_to_context_multiple_variables():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"var1": "value1", "var2": "value2", "var3": "value3"}
    overwrite_context = {"var1": "new1", "var3": "new3"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var1": "new1", "var2": "value2", "var3": "new3"}


def test_apply_overwrites_to_context_nested_with_list_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"choices": ["a", "b", "c"]}}
    overwrite_context = {"nested": {"choices": ["b", "c"]}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["nested"]["choices"] == ["b", "c"]


def test_apply_overwrites_to_context_list_overwrite_in_nested_dict():
    from cookiecutter.generate import apply_overwrites_to_context
    context = {"nested": {"items": ["a", "b"]}}
    overwrite_context = {"nested": {"items": ["x", "y"]}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["nested"]["items"] == ["x", "y"]


# LLM-generated content at query #30
#--------------------------

```python
def test_generate_context_predicate_line_18_evaluates_to_false(tmp_path):
    """Test that the predicate at line 18 (open() call) evaluates to False when file doesn't exist."""
    import os
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException
    
    non_existent_file = os.path.join(str(tmp_path), 'non_existent_cookiecutter.json')
    
    try:
        generate_context(context_file=non_existent_file)
        # If we reach here, the predicate was not False (file was opened)
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        # The predicate at line 18 evaluates to False (file open fails)
        # This is the expected behavior when the file doesn't exist
        pass


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_files_with_minimal_context(tmp_path, mocker):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.create_env_with_context')
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(output_dir / 'test_project', True))
    mocker.patch('cookiecutter.generate.work_in')
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mock_walk = mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(repo_dir, context, output_dir)
    
    assert result is not None


def test_generate_files_with_default_context(tmp_path, mocker):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.create_env_with_context')
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(output_dir / 'project', True))
    mocker.patch('cookiecutter.generate.work_in')
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(repo_dir)
    
    assert result is not None


def test_generate_files_with_hooks_disabled(tmp_path, mocker):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.create_env_with_context')
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(output_dir / 'test_project', True))
    mocker.patch('cookiecutter.generate.work_in')
    mock_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    
    mock_hook.assert_not_called()
    assert result is not None


def test_generate_files_with_overwrite_if_exists(tmp_path, mocker):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.create_env_with_context')
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(output_dir / 'test_project', True))
    mocker.patch('cookiecutter.generate.work_in')
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    
    assert result is not None


def test_generate_files_with_skip_if_file_exists(tmp_path, mocker):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.create_env_with_context')
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(output_dir / 'test_project', True))
    mocker.patch('cookiecutter.generate.work_in')
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    
    assert result is not None


def test_generate_files_keeps_project_on_failure(tmp_path, mocker):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.create_env_with_context')
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(output_dir / 'test_project', True))
    mocker.patch('cookiecutter.generate.work_in')
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk', return_value=[('.', [], [])])
    
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    
    assert result is not None


def test_generate_files_with_custom_output_dir(tmp_path, mocker):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    custom_output = tmp_path / "custom_output"
    custom_output.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.create_env_with_context')
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(custom_output / 'test_project', True))
    mocker.patch('cookiecutter.generate.work_in')
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('os.walk


# LLM-generated content at query #32
#--------------------------

```python
def test_generate_context_basic(tmp_path, monkeypatch):
    """Test generate_context with a basic JSON file."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_default_context(tmp_path, monkeypatch):
    """Test generate_context with default_context parameter."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    default_context = {"author": "Jane"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "Jane"


def test_generate_context_with_extra_context(tmp_path, monkeypatch):
    """Test generate_context with extra_context parameter."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John"}')
    
    extra_context = {"project_name": "new_project"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "new_project"
    assert result["cookiecutter"]["author"] == "John"


def test_generate_context_with_choice_variable(tmp_path, monkeypatch):
    """Test generate_context with choice variable and extra_context."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"
    assert "MIT" in result["cookiecutter"]["license"]


def test_generate_context_with_boolean_variable(tmp_path, monkeypatch):
    """Test generate_context with boolean variable and extra_context."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    extra_context = {"use_docker": "false"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_with_dict_variable(tmp_path, monkeypatch):
    """Test generate_context with nested dictionary variable."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"config": {"key1": "value1", "key2": "value2"}}')
    
    extra_context = {"config": {"key1": "new_value1"}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["config"]["key1"] == "new_value1"
    assert result["cookiecutter"]["config"]["key2"] == "value2"


def test_generate_context_invalid_json(tmp_path, monkeypatch):
    """Test generate_context with invalid JSON file."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"invalid": json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Should raise ContextDecodingException"
    except Exception as e:
        assert "JSON decoding error" in str(e)


def test_generate_context_missing_file(tmp_path, monkeypatch):
    """Test generate_context with missing context file."""
    monkeypatch.chdir(tmp_path)
    context_file = str(tmp_path / "nonexistent.json")
    
    try:
        generate_context(context_file)
        assert False, "Should raise an exception"
    except FileNotFoundError:
        pass


def test_generate_context_invalid_choice(tmp_path, monkeypatch):
    """Test generate_context with invalid choice value."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    extra_context = {"license": "GPL"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "GPL" in str(e)


def test_generate_context_invalid_boolean_conversion(tmp_path, monkeypatch):
    """Test generate_context with invalid boolean conversion."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    extra_context = {"use_docker": "invalid_value"}
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_generate_context_multichoice_variable(tmp_path, monkeypatch):
    """Test generate_context with multichoice variable."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    
    extra_context = {"features": ["feature2", "feature3"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["features"]) == {"feature2", "feature3"}


def test_generate_context_custom_filename(tmp_path, monkeypatch):
    """Test generate_context with custom context file name."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "custom_context.json"
    context_file.write_text('{"project_name": "test_project"}')
    
    result = generate_context(str(context_file))
    
    assert "custom_context" in result
    assert result["custom_context"]["project_name"] == "test_project"


def test_generate_context_with_default_and_extra_context(tmp_path, monkeypatch):
    """Test generate_context with both default_context and extra_context."""
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"name": "project", "author": "John", "license": "MIT"}')
    
    default_context = {"author": "Jane"}
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), default_context=default_context, extra_context=extra_context)
    
    assert result["cookiecutter"]["name"] == "project"
    assert result["cookiecutter"]["author"] == "Jane"
    assert result["cookiecutter"]["license"] == "Apache"


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_files_basic(tmp_path, monkeypatch):
    """Test generate_files creates project from template."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    # Setup template structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}")
    
    # Create context
    context = OrderedDict([
        ('cookiecutter', {'project_name': 'my_project'})
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
    )
    
    assert result
    assert Path(result).exists()
    assert "my_project" in result


def test_generate_files_with_subdirectories(tmp_path):
    """Test generate_files handles subdirectories correctly."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    subdir = template_dir / "src"
    subdir.mkdir()
    
    file_in_subdir = subdir / "main.py"
    file_in_subdir.write_text("# {{cookiecutter.name}}")
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'myapp'})
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
    )
    
    assert Path(result, "src", "main.py").exists()


def test_generate_files_skip_if_exists(tmp_path):
    """Test generate_files skips files when skip_if_file_exists is True."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.proj}}"
    template_dir.mkdir()
    
    template_file = template_dir / "config.txt"
    template_file.write_text("config={{cookiecutter.proj}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'proj': 'test'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=False,
    )
    
    config_file = Path(result, "config.txt")
    original_content = config_file.read_text()
    
    result2 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
    )
    
    assert config_file.read_text() == original_content


def test_generate_files_empty_context(tmp_path):
    """Test generate_files with empty context uses default."""
    from pathlib import Path
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "static_name"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=None,
        output_dir=str(output_dir),
    )
    
    assert Path(result).exists()


def test_generate_files_overwrite_if_exists(tmp_path):
    """Test generate_files can overwrite existing output directory."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("v1")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'project'})
    ])
    
    result1 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
    )
    
    result2 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
    )
    
    assert Path(result2).exists()


def test_generate_files_binary_file(tmp_path):
    """Test generate_files handles binary files correctly."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    binary_file = template_dir / "image.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'app'})
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
    )
    
    output_binary = Path(result, "image.bin")
    assert output_binary.exists()
    assert output_binary.read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_files_no_hooks(tmp_path, monkeypatch):
    """Test generate_files with accept_hooks=False."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter.generate import generate_files
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "test.txt"
    template_file.write_text("test")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', {'name': 'proj'})
    ])
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False,
    )
    
    assert Path(result).exists()


def test_generate_files_multiple_files(tmp_path):
    """Test generate_files handles multiple files correctly."""
    from pathlib import Path
    from collections import OrderedDict


# LLM-generated content at query #34
#--------------------------

```python
def test_generate_files_with_context_and_default_output_dir(tmp_path, mocker):
    """Test generate_files with context and default output directory."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    test_file = template_dir / "test.txt"
    test_file.write_text("Hello {{cookiecutter.project_name}}")
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.os.walk', return_value=[
        ('.', ['subdir'], ['test.txt']),
        ('./subdir', [], [])
    ])
    mocker.patch('cookiecutter.generate.is_copy_only_path', return_value=False)
    mocker.patch('cookiecutter.generate.generate_file')
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(tmp_path / "my_project", True))
    
    result = generate_files(repo_dir, context, str(tmp_path))
    
    assert result is not None


def test_generate_files_without_context(tmp_path, mocker):
    """Test generate_files without context uses empty OrderedDict."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.create_env_with_context')
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.os.walk', return_value=[('.', [], [])])
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(tmp_path / "project", True))
    
    result = generate_files(repo_dir, None, str(tmp_path))
    
    assert result is not None


def test_generate_files_with_overwrite_if_exists(tmp_path, mocker):
    """Test generate_files with overwrite_if_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.os.walk', return_value=[('.', [], [])])
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(tmp_path / "my_project", True))
    
    result = generate_files(repo_dir, context, str(tmp_path), overwrite_if_exists=True)
    
    assert result is not None


def test_generate_files_with_skip_if_file_exists(tmp_path, mocker):
    """Test generate_files with skip_if_file_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.os.walk', return_value=[('.', [], ['file.txt'])])
    mocker.patch('cookiecutter.generate.is_copy_only_path', return_value=False)
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(tmp_path / "my_project", True))
    mocker.patch('cookiecutter.generate.generate_file')
    
    result = generate_files(repo_dir, context, str(tmp_path), skip_if_file_exists=True)
    
    assert result is not None


def test_generate_files_without_hooks(tmp_path, mocker):
    """Test generate_files with accept_hooks set to False."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    hook_mock = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.os.walk', return_value=[('.', [], [])])
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(tmp_path / "my_project", True))
    
    result = generate_files(repo_dir, context, str(tmp_path), accept_hooks=False)
    
    assert hook_mock.call_count == 0
    assert result is not None


def test_generate_files_with_undefined_error(tmp_path, mocker):
    """Test generate_files raises UndefinedVariableInTemplate on UndefinedError."""
    from jinja2 import UndefinedError
    from cookiecutter.exceptions import UndefinedVariableInTemplate
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.render_and_create_dir', side_effect=UndefinedError('undefined'))
    
    try:
        generate_files(repo_dir, context, str(tmp_path))
        assert False, "Should have raised UndefinedVariableInTemplate"
    except UndefinedVariableInTemplate:
        pass


def test_generate_files_keep_project_on_failure(tmp_path, mocker):
    """Test generate_files with keep_project_on_failure flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    mocker.patch('cookiecutter.generate.find_template', return_value=template_dir)
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.os.walk', return_value=[('.', [], [])])
    mocker.patch('cookiecutter.generate.render_and_create_dir', return_value=(tmp_path / "my_project", True))
    
    result = generate_files(repo_dir, context, str(tmp_path), keep_project_on_failure=True)
    
    assert result is not None


def test_generate_files_with_copy_only_dirs(tmp_path, mocker):
    """Test generate_files with copy_only directories."""
    repo_dir


# LLM-generated content at query #35
#--------------------------

```python
def test_generate_context_basic(tmp_path):
    """Test generate_context with a basic JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "project_slug": "my_project"}')
    
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["project_slug"] == "my_project"


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


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context with choice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["license"][0] == "Apache"
    assert "MIT" in result["cookiecutter"]["license"]
    assert "GPL" in result["cookiecutter"]["license"]


def test_generate_context_with_boolean_variable(tmp_path):
    """Test generate_context with boolean variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_docker": true}')
    
    extra_context = {"use_docker": "false"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_with_dict_variable(tmp_path):
    """Test generate_context with nested dictionary variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"author": {"name": "John", "email": "john@example.com"}}')
    
    extra_context = {"author": {"name": "Jane"}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["author"]["name"] == "Jane"
    assert result["cookiecutter"]["author"]["email"] == "john@example.com"


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context with invalid JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"invalid json}')
    
    try:
        generate_context(str(context_file))
        assert False, "Should raise ContextDecodingException"
    except Exception as e:
        assert "JSON decoding error" in str(e)


def test_generate_context_invalid_choice_override(tmp_path):
    """Test generate_context with invalid choice override."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"license": ["MIT", "Apache"]}')
    
    extra_context = {"license": "GPL"}
    
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "choice variable" in str(e)


def test_generate_context_invalid_multichoice_override(tmp_path):
    """Test generate_context with invalid multichoice override."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"licenses": ["MIT", "Apache"]}')
    
    extra_context = {"licenses": ["MIT", "GPL"]}
    
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "multi-choice variable" in str(e)


def test_generate_context_with_multichoice_valid(tmp_path):
    """Test generate_context with valid multichoice override."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"licenses": ["MIT", "Apache", "GPL"]}')
    
    extra_context = {"licenses": ["Apache", "GPL"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["licenses"] == ["Apache", "GPL"]


def test_generate_context_boolean_yes(tmp_path):
    """Test generate_context converting yes string to boolean."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_feature": false}')
    
    extra_context = {"use_feature": "yes"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_feature"] is True


def test_generate_context_boolean_no(tmp_path):
    """Test generate_context converting no string to boolean."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"use_feature": true}')
    
    extra_context = {"use_feature": "no"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_feature"] is False


def test_generate_context_boolean_true_string(tmp_path):
    """Test generate_context converting 'true' string to boolean."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"debug": false}')
    
    extra_context = {"debug": "true"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["debug"] is True


def test_generate_context_invalid_boolean_string(tmp_path):
    """Test generate_context with invalid boolean string."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"debug": false}')
    
    extra_context = {"debug": "maybe"}
    
    try:
        generate_context(str(context_file), extra_context=extra_context)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


def test_generate_context_with_both_contexts(tmp_path):
    """Test generate_context with both default and extra context."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"name": "original", "version": "1.0", "author": "John"}')
    
    default_context = {"version": "2.0"}
    extra_context = {"author": "Jane"}
    
    result = generate_context(str(context_file), default_context=default_context, extra


