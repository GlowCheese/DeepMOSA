####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_prompt_and_delete_no_input_dir_exists(monkeypatch, tmp_path):
    import os
    from pathlib import Path
    from cookiecutter.prompt import prompt_and_delete

    directory = tmp_path / "test_dir"
    directory.mkdir()
    
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    monkeypatch.setattr("cookiecutter.prompt.rmtree", lambda p: None)
    
    result = prompt_and_delete(str(directory), no_input=True)
    assert result is True

def test_prompt_and_delete_no_input_file_exists(monkeypatch, tmp_path):
    import os
    from pathlib import Path
    from cookiecutter.prompt import prompt_and_delete

    file = tmp_path / "test_file.txt"
    file.write_text("content")
    
    monkeypatch.setattr("os.path.isdir", lambda p: False)
    monkeypatch.setattr("os.remove", lambda p: None)
    
    result = prompt_and_delete(str(file), no_input=True)
    assert result is True

def test_prompt_and_delete_with_input_yes_to_delete_dir(monkeypatch, tmp_path):
    import os
    from pathlib import Path
    from cookiecutter.prompt import prompt_and_delete

    directory = tmp_path / "test_dir"
    directory.mkdir()
    
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    monkeypatch.setattr("cookiecutter.prompt.read_user_yes_no", lambda q, d: True)
    monkeypatch.setattr("cookiecutter.prompt.rmtree", lambda p: None)
    
    result = prompt_and_delete(str(directory), no_input=False)
    assert result is True

def test_prompt_and_delete_with_input_no_to_delete_reuse_true(monkeypatch, tmp_path):
    import os
    from pathlib import Path
    from cookiecutter.prompt import prompt_and_delete

    directory = tmp_path / "test_dir"
    directory.mkdir()
    
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    monkeypatch.setattr("cookiecutter.prompt.read_user_yes_no", lambda q, d: False if "delete" in q else True)
    
    result = prompt_and_delete(str(directory), no_input=False)
    assert result is False

def test_prompt_and_delete_with_input_no_to_delete_reuse_false_exits(monkeypatch, tmp_path):
    import os
    from pathlib import Path
    from cookiecutter.prompt import prompt_and_delete

    directory = tmp_path / "test_dir"
    directory.mkdir()
    
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    monkeypatch.setattr("cookiecutter.prompt.read_user_yes_no", lambda q, d: False)
    mock_exit = monkeypatch.setattr("sys.exit", lambda x: None)
    
    result = prompt_and_delete(str(directory), no_input=False)
    assert mock_exit.called
```


# LLM-generated content at query #2
#--------------------------

```python
def test_prompt_and_delete_true_when_no_input_is_true():
    import os
    from pathlib import Path
    from unittest.mock import patch

    test_path = Path("test_dir_to_delete")
    test_path.mkdir(exist_ok=True)
    
    with patch("cookiecutter.prompt.os.path.isdir", return_value=True), \
         patch("cookiecutter.prompt.rmtree") as mock_rmtree:
        result = prompt_and_delete(test_path, no_input=True)
        assert result is True
        mock_rmtree.assert_called_once_with(test_path)

    test_path.rmdir()
```


# LLM-generated content at query #3
#--------------------------

```python
import json
from collections import OrderedDict

def test_process_json_valid_dict():
    user_value = '{"key": "value", "number": 123}'
    expected = OrderedDict([('key', 'value'), ('number', 123)])
    assert process_json(user_value) == expected

def test_process_json_invalid_json_syntax():
    user_value = '{"key": "value", missing_quote}'
    try:
        process_json(user_value)
    except InvalidResponse as e:
        assert str(e) == 'Unable to decode to JSON.'

def test_process_json_not_a_dict_list():
    user_value = '[1, 2, 3]'
    try:
        process_json(user_value)
    except InvalidResponse as e:
        assert str(e) == 'Requires JSON dict.'

def test_process_json_not_a_dict_string():
    user_value = '"just a string"'
    try:
        process_json(user_value)
    except InvalidResponse as e:
        assert str(e) == 'Requires JSON dict.'

def test_process_json_empty_dict():
    user_value = '{}'
    expected = OrderedDict()
    assert process_json(user_value) == expected
```


# LLM-generated content at query #4
#--------------------------

```python
def test_read_user_variable_returns_default_when_prompt_is_none():
    from unittest.mock import patch
    with patch('__main__.Prompt.ask', return_value="default_val") as mock_ask:
        result = read_user_variable("test_var", "default_val")
        assert result == "default_val"
        mock_ask.assert_called_once_with("test_var", default="default_val")

def test_read_user_variable_uses_custom_prompt_from_dict():
    from unittest.mock import patch
    prompts = {"test_var": "Custom Question"}
    with patch('__main__.Prompt.ask', return_value="user_input") as mock_ask:
        result = read_user_variable("test_var", "default_val", prompts=prompts)
        assert result == "user_input"
        mock_ask.assert_called_once_with("Custom Question", default="default_val")

def test_read_user_variable_applies_prefix():
    from unittest.mock import patch
    with patch('__main__.Prompt.ask', return_value="input") as mock_ask:
        result = read_user_variable("var", "def", prefix="PRE_: ")
        assert result == "input"
        mock_ask.assert_called_once_with("PRE_: var", default="def")

def test_read_user_variable_handles_none_input_retry():
    from unittest.mock import patch
    with patch('__main__.Prompt.ask', side_effect=[None, "valid_input"]) as mock_ask:
        result = read_user_variable("var", "def")
        assert result == "valid_input"
        assert mock_ask.call_count == 2

def test_read_user_variable_ignores_empty_prompt_string_in_dict():
    from unittest.mock import patch
    prompts = {"test_var": ""}
    with patch('__main__.Prompt.ask', return_value="val") as mock_ask:
        result = read_user_variable("test_var", "def", prompts=prompts)
        assert result == "val"
        mock_ask.assert_called_once_with("test_var", default="def")
```


# LLM-generated content at query #5
#--------------------------

```python
import json
from collections import OrderedDict

def test_process_json_valid_dict():
    input_data = '{"key": "value", "number": 123}'
    expected_output = OrderedDict([('key', 'value'), ('number', 123)])
    assert process_json(input_data) == expected_output

def test_process_json_invalid_syntax():
    input_data = '{"key": "value",}'
    try:
        process_json(input_data)
    except InvalidResponse as error:
        assert str(error.args[0]) == 'Unable to decode to JSON.'

def test_process_json_not_a_dict():
    input_data = '"just a string"'
    try:
        process_json(input_data)
    except InvalidResponse as error:
        assert str(error.args[0]) == 'Requires JSON dict.'

def test_process_json_list_input():
    input_data = '[1, 2, 3]'
    try:
        process_json(input_data)
    except InvalidResponse as error:
        assert str(error.args[0]) == 'Requires JSON dict.'

def test_process_json_empty_dict():
    input_data = '{}'
    expected_output = OrderedDict()
    assert process_json(input_data) == expected_output
```


# LLM-generated content at query #6
#--------------------------

```python
import json
from collections import OrderedDict

def test_process_json_valid_dict():
    input_value = '{"key": "value", "number": 123}'
    expected_output = OrderedDict([("key", "value"), ("number", 123)])
    assert process_json(input_value) == expected_output

def test_process_json_invalid_json_syntax():
    input_value = '{"key": "value",}'  # Trailing comma is invalid in standard JSON
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(input_value)
    assert str(excinfo.value) == 'Unable to decode to JSON.'

def test_process_json_not_a_dict_list():
    input_value = '[1, 2, 3]'
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(input_value)
    assert str(excinfo.value) == 'Requires JSON dict.'

def test_process_json_not_a_dict_string():
    input_value = '"just a string"'
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(input_value)
    assert str(excinfo.value) == 'Requires JSON dict.'

def test_process_json_empty_dict():
    input_value = '{}'
    expected_output = OrderedDict()
    assert process_json(input_value) == expected_output

def test_process_json_preserves_order():
    input_value = '{"a": 1, "b": 2, "c": 3}'
    result = process_json(input_value)
    assert list(result.keys()) == ["a", "b", "c"]
```


# LLM-generated content at query #7
#--------------------------

```python
def test_render_variable_none():
    from jinja2 import Environment
    env = Environment()
    assert render_variable(env, None, {}) is None

def test_render_variable_bool():
    from jinja2 import Environment
    env = Environment()
    assert render_variable(env, True, {}) is True
    assert render_variable(env, False, {}) is False

def test_render_variable_simple_string():
    from jinja2 import Environment
    env = Environment()
    assert render_variable(env, "hello", {}) == "hello"

def test_render_variable_template_rendering():
    from jinja2 import Environment
    env = Environment()
    context = {"project_name": "my_project"}
    assert render_variable(env, "{{ cookiecutter.project_name }}", context) == "my_project"

def test_render_variable_template_with_logic():
    from jinja2 import Environment
    env = Environment()
    context = {"project_name": "Peanut Butter Cookie"}
    assert render_variable(env, "{{ cookiecutter.project_name.replace(' ', '_') }}", context) == "Peanut_Butter_Cookie"

def test_render_variable_list():
    from jinja2 import Environment
    env = Environment()
    context = {"val": "foo"}
    raw_list = ["{{ cookiecutter.val }}", "bar"]
    assert render_variable(env, raw_list, context) == ["foo", "bar"]

def test_render_variable_dict():
    from jinja2 import Environment
    env = Environment()
    context = {"name": "test"}
    raw_dict = {"key_{{ cookiecutter.name }}": "{{ cookiecutter.name }}"}
    assert render_variable(env, raw_dict, context) == {"key_test": "test"}

def test_render_variable_non_string_type_conversion():
    from jinja2 import Environment
    env = Environment()
    assert render_variable(env, 123, {}) == "123"
```


# LLM-generated content at query #8
#--------------------------

```python
from unittest.mock import patch, MagicMock

def test_read_user_yes_no_uses_var_name_when_no_prompts():
    with patch("builtins.print"):
        with patch("your_module.YesNoPrompt.ask") as mock_ask:
            mock_ask.return_value = True
            result = read_user_yes_no("confirm", False, prompts=None, prefix="[?] ")
            assert result is True
            mock_ask.assert_called_once_with("[?] confirm", default=False)

def test_read_user_yes_no_uses_prompt_from_dict():
    with patch("builtins.print"):
        with patch("your_module.YesNoPrompt.ask") as mock_ask:
            mock_ask.return_value = False
            prompts = {"save": "Do you want to save changes?"}
            result = read_user_yes_no("save", True, prompts=prompts, prefix="")
            assert result is False
            mock_ask.assert_called_once_with("Do you want to save changes?", default=True)

def test_read_user_yes_no_uses_var_name_when_prompt_key_missing():
    with patch("builtins.print"):
        with patch("your_module.YesNoPrompt.ask") as mock_ask:
            mock_ask.return_value = True
            prompts = {"other_key": "Different prompt"}
            result = read_user_yes_no("missing_key", False, prompts=prompts, prefix="Prefix: ")
            assert result is True
            mock_ask.assert_called_once_with("Prefix: missing_key", default=False)

def test_read_user_yes_no_handles_empty_prompt_string():
    with patch("builtins.print"):
        with patch("your_module.YesNoPrompt.ask") as mock_ask:
            mock_ask.return_value = True
            prompts = {"empty": ""}
            result = read_user_yes_no("empty", False, prompts=prompts, prefix="")
            assert result is True
            mock_ask.assert_called_once_with("empty", default=False)

def test_read_user_yes_no_with_prefix_and_prompt():
    with patch("builtins.print"):
        with patch("your_module.YesNoPrompt.ask") as mock_ask:
            mock_ask.return_value = False
            prompts = {"delete": "Delete file?"}
            result = read_user_yes_no("delete", False, prompts=prompts, prefix="CONFIRM: ")
            assert result is False
            mock_ask.assert_called_once_with("CONFIRM: Delete file?", default=False)
```


# LLM-generated content at query #9
#--------------------------

```python
from unittest.mock import patch

@patch("json_prompt_module.JsonPrompt.ask")
def test_read_user_dict_valid_input():
    import json_prompt_module
    json_prompt_module.JsonPrompt.ask.return_value = {"key": "value"}
    result = read_user_dict("test_var", {"default": 1}, prompts={"test_var": "Custom Prompt"}, prefix="Pref: ")
    assert result == {"key": "value"}
    json_prompt_module.JsonPrompt.ask.assert_called_with(
        "Pref: Custom Prompt [cyan bold]([DEFAULT_DISPLAY])[/]",
        default={"default": 1},
        show_default=False
    )

@patch("json_prompt_module.JsonPrompt.ask")
def test_read_user_dict_uses_var_name_as_question():
    import json_prompt_module
    json_prompt_module.JsonPrompt.ask.return_value = {}
    result = read_user_dict("simple_var", {})
    assert result == {}
    json_prompt_module.JsonPrompt.ask.assert_called_with(
        "simple_var [cyan bold]([DEFAULT_DISPLAY])[/]",
        default={},
        show_default=False
    )

@patch("json_prompt_module.JsonPrompt.ask")
def test_read_user_dict_no_prompts_provided():
    import json_prompt_module
    json_prompt_module.JsonPrompt.ask.return_value = {"a": 1}
    result = read_user_dict("var", {}, prompts=None)
    assert result == {"a": 1}
    json_prompt_module.JsonPrompt.ask.assert_called_with(
        "var [cyan bold]([DEFAULT_DISPLAY])[/]",
        default={},
        show_default=False
    )

def test_read_user_dict_raises_type_error_on_invalid_default():
    try:
        read_user_dict("test", "not_a_dict")
        assert False
    except TypeError:
        assert True

@patch("json_prompt_module.JsonPrompt.ask")
def test_read_user_dict_with_empty_prompt_mapping():
    import json_prompt_module
    json_prompt_module.JsonPrompt.ask.return_value = {}
    result = read_user_dict("var", {}, prompts={"var": ""})
    assert result == {}
    json_prompt_module.JsonPrompt.ask.assert_called_with(
        "var [cyan bold]([DEFAULT_DISPLAY])[/]",
        default={},
        show_default=False
    )
```


# LLM-generated content at query #10
#--------------------------

```python
from unittest.mock import patch, MagicMock
from collections import OrderedDict

def test_prompt_choice_for_config_no_input_returns_first_option():
    env = MagicMock()
    cookiecutter_dict = {"project_name": "test_project"}
    options = ["{{ cookiecutter.project_name }}", "other_option"]
    key = "some_key"
    # Mocking render_variable behavior via the environment's from_string mock
    mock_template = MagicMock()
    mock_template.render.return_value = "test_project"
    env.from_string.side_effect = [MagicMock(render=lambda cookiecutter: "test_project"), MagicMock(render=lambda cookiecutter: "other_option")]
    
    result = prompt_choice_for_config(
        cookiecutter_dict=cookiecutter_dict,
        env=env,
        key=key,
        options=options,
        no_input=True
    )
    assert result == "test_project"

def test_prompt_choice_for_config_no_input_raises_error_on_empty_options():
    env = MagicMock()
    cookiecutter_dict = {}
    options = []
    key = "some_key"
    
    import pytest
    with pytest.raises(ValueError, match="The list of choices is empty"):
        prompt_choice_for_config(
            cookiecutter_dict=cookiecutter_dict,
            env=env,
            key=key,
            options=options,
            no_input=True
        )

@patch("prompt_choice_for_config.read_user_choice")
def test_prompt_choice_for_config_calls_read_user_choice_when_input_is_required(mock_read_user_choice):
    env = MagicMock()
    cookiecutter_dict = {"project_name": "test_project"}
    options = ["{{ cookiecutter.project_name }}", "option2"]
    key = "some_key"
    
    # Setup mock to return a specific value
    mock_read_user_choice.return_value = "test_project"
    
    # Mocking render_variable logic inside the list comprehension of prompt_choice_for_config
    # We need to ensure render_variable returns the expected values
    template1 = MagicMock()
    template1.render.return_value = "test_project"
    template2 = MagicMock()
    template2.render.return_value = "option2"
    env.from_string.side_effect = [template1, template2]

    result = prompt_choice_for_config(
        cookiecutter_dict=cookiecutter_dict,
        env=env,
        key=key,
        options=options,
        no_input=False,
        prompts={"some_key": "Custom Prompt"},
        prefix="[TEST] "
    )
    
    assert result == "test_project"
    mock_read_user_choice.assert_called_once_with(
        key, 
        ["test_project", "option2"], 
        {"some_key": "Custom Prompt"}, 
        "[TEST] "
    )
```


# LLM-generated content at query #11
#--------------------------

```python
def test_process_response_true_variants():
    prompt = YesNoPrompt()
    assert prompt.process_response("1") is True
    assert prompt.process_response("true") is True
    assert prompt.process_response("T") is True
    assert prompt.process_response("yes") is True
    assert prompt.process_response("y") is True
    assert prompt.process_response("on") is True
    assert prompt.process_response("  YES  ") is True

def test_process_response_false_variants():
    prompt = YesNoPrompt()
    assert prompt.process_response("0") is False
    assert prompt.process_response("false") is False
    assert prompt.process_response("f") is False
    assert prompt.process_response("no") is False
    assert prompt.process_response("n") is False
    assert prompt.process_response("off") is False
    assert prompt.process_response("  NO  ") is False

def test_process_response_invalid_input():
    prompt = YesNoPrompt()
    import pytest
    with pytest.raises(InvalidResponse):
        prompt.process_response("maybe")
    with pytest.raises(InvalidResponse):
        prompt.process_response("")
    with pytest.raises(InvalidResponse):
        prompt.process_response("123")
```


# LLM-generated content at query #12
#--------------------------

```python
from unittest.mock import patch, MagicMock

@patch('__main__.Prompt.ask')
def test_read_user_variable_returns_default_when_no_input(mock_ask):
    mock_ask.return_value = "default"
    result = read_user_variable("name", "default")
    assert result == "default"
    mock_ask.assert_called_once_with("name", default="default")

@patch('__main__.Prompt.ask')
def test_read_user_variable_uses_custom_prompt(mock_ask):
    mock_ask.return_value = "John"
    prompts = {"name": "Please enter your name"}
    result = read_user_variable("name", "default", prompts=prompts)
    assert result == "John"
    mock_ask.assert_called_once_with("Please enter your name", default="default")

@patch('__main__.Prompt.ask')
def test_read_user_variable_uses_prefix(mock_ask):
    mock_ask.return_value = "John"
    result = read_user_variable("name", "default", prefix="Enter: ")
    assert result == "John"
    mock_ask.assert_called_once_with("Enter: name", default="default")

@patch('__main__.Prompt.ask')
def test_read_user_variable_retries_on_none(mock_ask):
    mock_ask.side_effect = [None, "recovered"]
    result = read_user_variable("name", "default")
    assert result == "recovered"
    assert mock_ask.call_count == 2

@patch('__main__.Prompt.ask')
def test_read_user_variable_with_empty_prompts_dict(mock_ask):
    mock_ask.return_value = "val"
    result = read_user_variable("name", "default", prompts={})
    assert result == "val"
    mock_ask.assert_called_once_with("name", default="default")
```


# LLM-generated content at query #13
#--------------------------

```python
from unittest.mock import patch
from collections import OrderedDict
from itertools import starmap

def test_read_user_choice_empty_options_raises_error():
    with patch('builtins.print'):
        import pytest
        with pytest.raises(ValueError):
            read_user_choice("test", [])

@patch('rich.prompt.Prompt.ask')
def test_read_user_choice_returns_first_option_on_default(mock_prompt_ask):
    mock_prompt_ask.return_value = '1'
    options = ['apple', 'banana', 'cherry']
    result = read_user_choice("fruit", options)
    assert result == 'apple'
    mock_prompt_ask.assert_called_once()

@patch('rich.prompt.Prompt.ask')
def test_read_user_choice_returns_specific_option(mock_prompt_ask):
    mock_prompt_ask.return_value = '2'
    options = ['apple', 'banana', 'cherry']
    result = read_user_choice("fruit", options)
    assert result == 'banana'

@patch('rich.prompt.Prompt.ask')
def test_read_user_choice_with_string_prompt(mock_prompt_ask):
    mock_prompt_ask.return_value = '1'
    options = ['apple', 'banana']
    prompts = {"fruit": "Which fruit do you like?"}
    result = read_user_choice("fruit", options, prompts=prompts)
    assert result == 'apple'
    args, kwargs = mock_prompt_ask.call_args
    assert "Which fruit do you like?" in args[0]

@patch('rich.prompt.Prompt.ask')
def test_read_user_choice_with_dict_prompt_and_custom_labels(mock_prompt_ask):
    mock_prompt_assumed_val = '1'
    mock_prompt_ask.return_value = '1'
    options = ['apple', 'banana']
    prompts = {
        "fruit": {
            "__prompt__": "Pick a flavor:",
            "1": "Sweet Apple",
            "2": "Sour Banana"
        }
    }
    result = read_user_choice("fruit", options, prompts=prompts)
    assert result == 'apple'
    args, kwargs = mock_prompt_ask.call_args
    assert "Pick a flavor:" in args[0]
    assert "[bold]{}[/] - [bold]Sweet Apple[/]" in args[0] or "Sweet Apple" in args[0]

@patch('rich.prompt.Prompt.ask')
def test_read_user_choice_with_prefix(mock_prompt_ask):
    mock_prompt_ask.return_value = '1'
    options = ['apple']
    result = read_user_choice("fruit", options, prefix="[QUERY] ")
    assert result == 'apple'
    args, kwargs = mock_prompt_ask.call_args
    assert args[0].startswith("[QUERY] Select fruit")
```


# LLM-generated content at query #14
#--------------------------

```python
def test_read_user_choice_predicate_false():
    from collections import OrderedDict
    from itertools import starmap
    from rich.prompt import Prompt

    # Mocking Prompt.ask to avoid actual terminal interaction
    import unittest.mock as mock
    
    # Setup: prompts[var_name] must be a string so line 27 is skipped (the 'else' block)
    # or it must be a dict that does NOT contain "__prompt__"
    # To specifically target the predicate at line 27, we ensure 'if "__prompt__" in prompts[var_name]' is False.
    # This happens if prompts[var_name] is a dictionary without that key.
    
    var_name = "test_var"
    options = ["opt1", "opt2"]
    prompts = {
        "test_var": {"some_key": "some_value"}
    }
    # To make the predicate at line 27 evaluate to False, "__prompt__" must not be in prompts[var_name]
    
    with mock.patch("rich.prompt.Prompt.ask", return_value="1") as mock_ask:
        result = read_user_choice(var_name=var_name, options=options, prompts=prompts)
        
        assert result == "opt1"
        assert "__prompt__" not in prompts[var_name]
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import OrderedDict
from cookiecutter.prompt import choose_nested_template

def test_choose_nested_template_new_style_success():
    context = {
        'cookiecutter': {
            'templates': {
                'option1': {'path': 'templates/option1'},
                'option2': {'path': 'templates/option2'}
            }
        }
    }
    repo_dir = "/tmp/repo"
    
    with patch("cookiecutter.prompt.prompt_choice_for_template") as mock_prompt, \
         patch("cookiecutter.utils.create_env_with_context"), \
         patch("pathlib.Path.is_absolute", return_value=False), \
         patch("pathlib.Path.resolve") as mock_resolve:
        
        mock_prompt.return_value = "option1"
        # Mocking resolve to return a predictable path string
        mock_resolve.side_effect = [Path("/tmp/repo"), Path("/tmp/repo/templates/option1")]
        
        result = choose_nested_template(context, repo_dir, no_input=True)
        
        assert result == str(Path("/tmp/repo/templates/option1"))
        mock_prompt.assert_called_once()

def test_choose_nested_template_old_style_success():
    context = {
        'cookiecutter': {
            'template': ['choice1 (templates/choice1)', 'choice2 (templates/choice2)']
        }
    }
    repo_dir = "/tmp/repo"
    
    with patch("cookiecutter.prompt.prompt_choice_for_config") as mock_prompt, \
         patch("cookiecutter.utils.create_env_with_context"), \
         patch("pathlib.Path.is_absolute", return_value=False), \
         patch("pathlib.Path.resolve") as mock_resolve:
        
        mock_prompt.return_value = "choice1 (templates/choice1)"
        mock_resolve.side_effect = [Path("/tmp/repo"), Path("/tmp/repo/templates/choice1")]
        
        result = choose_nested_template(context, repo_dir, no_input=True)
        
        assert result == str(Path("/tmp/repo/templates/choice1"))
        mock_prompt.assert_called_once()

def test_choose_nested_template_illegal_path_raises_error():
    context = {
        'cookiecutter': {
            'templates': {
                'option1': {'path': '/absolute/path/to/template'}
            }
        }
    }
    repo_dir = "/tmp/repo"
    
    with patch("cookiecutter.prompt.prompt_choice_for_template") as mock_prompt, \
         patch("cookiecutter.utils.create_env_with_context"), \
         patch("pathlib.Path.is_absolute", return_value=True):
        
        mock_prompt.return_value = "option1"
        
        with pytest.raises(ValueError, match="Illegal template path"):
            choose_nested_template(context, repo_dir, no_input=True)

def test_choose_nested_template_none_path_raises_error():
    context = {
        'cookiecutter': {
            'templates': {
                'option1': {'path': None}
            }
        }
    }
    repo_dir = "/tmp/repo"
    
    with patch("cookiecutter.prompt.prompt_choice_for_template") as mock_prompt, \
         patch("cookiecutter.utils.create_env_with_context"):
        
        mock_prompt.return_value = "option1"
        
        with pytest.raises(ValueError, match="Illegal template path"):
            choose_nested_template(context, repo_dir, no_input=True)
```


# LLM-generated content at query #16
#--------------------------

```python
from unittest.mock import MagicMock, patch
from collections import OrderedDict
from cookiecutter.prompt import prompt_for_config

def test_prompt_for_config_no_input_simple_vars():
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_internal_var': 'secret'
        }
    }
    # We use no_input=True to avoid interaction with Prompt.ask
    result = prompt_for_config(context, no_input=True)
    assert result['project_name'] == 'my_project'
    assert result['_internal_var'] == 'secret'

def test_prompt_for_config_renders_template_with_no_input():
    context = {
        'cookiecutter': {
            'project_name': 'My Project',
            'repo_name': '{{ cookiecutter.project_name.replace(" ", "_") }}',
            '_internal': 'val'
        }
    }
    result = prompt_for_config(context, no_input=True)
    assert result['project_name'] == 'My Project'
    assert result['repo_name'] == 'My_Project'

def test_prompt_for_config_handles_list_choices_no_input():
    context = {
        'cookiecutter': {
            'type': ['web', 'api', 'cli'],
            '_internal': 'val'
        }
    }
    # with no_input=True, prompt_choice_for_config returns the first element
    result = prompt_for_config(context, no_input=True)
    assert result['type'] == 'web'

def test_prompt_for_config_handles_dict_variables_no_input():
    context = {
        'cookiecutter': {
            'settings': {'debug': 'true', 'port': '8080'},
            '_internal': 'val'
        }
    }
    result = prompt_for_config(context, no_input=True)
    assert result['settings']['debug'] == 'true'
    assert result['settings']['port'] == '8080'

def test_prompt_for_config_handles_double_underscore_keys():
    context = {
        'cookiecutter': {
            'project_name': 'test',
            '__template_var__': '{{ cookiecutter.project_name }}'
        }
    }
    result = prompt_for_config(context, no_input=True)
    assert result['__template_var__'] == 'test'

@patch('cookiecutter.prompt.read_user_variable')
def test_prompt_for_config_calls_read_user_variable_when_not_no_input(mock_read_user):
    context = {
        'cookiecutter': {
            'user_input': 'default_val'
        }
    }
    mock_read_user.return_value = 'user_provided_val'
    result = prompt_for_config(context, no_input=False)
    assert result['user_input'] == 'user_provided_val'
    mock_read_user.assert_called()

@patch('cookiecutter.prompt.read_user_choice')
def test_prompt_for_config_calls_read_user_choice_when_list_is_present(mock_read_choice):
    context = {
        'cookiecutter': {
            'choice_var': ['option1', 'option2']
        }
    }
    mock_read_choice.return_value = 'option2'
    result = prompt_for_config(context, no_input=False)
    assert result['choice_var'] == 'option2'
    mock_read_choice.assert_called()

@patch('cookiecutter.prompt.read_user_yes_no')
def test_prompt_for_config_calls_read_user_yes_no_for_bool(mock_read_yes_no):
    context = {
        'cookiecutter': {
            'is_enabled': True
        }
    }
    mock_read_yes_no.return_value = False
    result = prompt_for_config(context, no_input=False)
    assert result['is_enabled'] is False
    mock_read_yes_no.assert_called()

@patch('cookiecutter.prompt.JsonPrompt.ask')
def test_prompt_for_config_calls_read_user_dict_when_dict_is_present(mock_json_prompt):
    context = {
        'cookiecutter': {
            'config_dict': {'key': 'val'}
        }
    }
    mock_json_prompt.return_value = {'key': 'new_val'}
    result = prompt_for_config(context, no_input=False)
    assert result['config_dict'] == {'key': 'new_val'}
    mock_json_prompt.assert_called()
```


# LLM-generated content at query #17
#--------------------------

```python
def test_prompt_for_config_no_input_skips_read_user_dict():
    from collections import OrderedDict
    from unittest.mock import patch
    from cookiecutter.prompt import prompt_for_config

    context = {
        'cookiecutter': {
            'my_dict': {'key': 'value'},
            '_private_var': 'hidden'
        }
    }
    no_input = True

    with patch('cookiecutter.prompt.render_variable') as mock_render, \
         patch('cookiecutter.prompt.create_env_with_context') as mock_create_env:
        
        mock_render.return_value = {'key': 'value'}
        mock_create_env.return (None) # Mocking to avoid env complexity

        result = prompt_for_config(context, no_input=no_input)

        # The predicate at line 72 is: if not no_input and not key.startswith('__'):
        # Since no_input is True, the 'if' block containing read_user_dict should NOT be entered.
        # We verify this by ensuring read_user_dict was never called.
        from cookiecutter.prompt import read_user_dict
        with patch('cookiecutter.prompt.read_user_dict') as mock_read_dict:
            prompt_for_config(context, no_input=True)
            mock_read_dict.assert_not_called()

def test_prompt_for_config_input_enters_read_user_dict():
    from collections import OrderedDict
    from unittest.mock import patch
    from cookiecutter.prompt import prompt_for_config

    context = {
        'cookiecutter': {
            'my_dict': {'key': 'value'}
        }
    }
    no_input = False

    with patch('cookiecutter.prompt.render_variable') as mock_render, \
         patch('cookiecutter.prompt.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.prompt.read_user_dict') as mock_read_dict:
        
        mock_render.return_value = {'key': 'value'}
        mock_read_dict.return_value = {'key': 'new_value'}

        result = prompt_for_config(context, no_input=no_input)

        # Since no_input is False and key does not start with '__', 
        # the predicate at line 72 evaluates to True.
        mock_read_dict.assert_called_once()
        assert result['my_dict'] == {'key': 'new_value'}
```


# LLM-generated content at query #18
#--------------------------

```python
from unittest.mock import patch, MagicMock
from collections import OrderedDict
from cookiecutter.prompt import prompt_for_config

def test_prompt_for_config_no_input_simple_values():
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_private_var': 'hidden'
        }
    }
    with patch('cookiecutter.prompt.create_env_with_context') as mock_env:
        mock_env.return_value = MagicMock()
        result = prompt_for_config(context, no_input=True)
        assert result['project_name'] == 'my_project'
        assert result['_private_var'] == 'hidden'

def test_prompt_for_config_no_input_with_templates():
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'repo_name': '{{ cookiecutter.project_name.replace(" ", "_") }}'
        }
    }
    # Mocking render_variable to simulate template rendering without needing a real Jinja env
    with patch('cookiecutter.prompt.render_variable') as mock_render:
        mock_render.side_effect = [
            'my_project',  # for project_name
            'my_project',  # for repo_name (simulated result)
            'my_project'   # for the dict/logic inside render_variable calls
        ]
        result = prompt_for_config(context, no_input=True)
        assert result['repo_name'] == 'my_project'

def test_prompt_for_config_raises_undefined_error():
    from jinja2 import UndefinedError
    context = {
        'cookiecutter': {
            'project_name': '{{ non_existent_var }}'
        }
    }
    # We need to mock render_variable to raise the specific Jinja error 
    # that prompt_for_config catches and wraps.
    with patch('cookiecutter.prompt.render_variable') as mock_render:
        mock_render.side_effect = UndefinedError("Undefined variable")
        from cookiecutter.prompt import UndefinedVariableInTemplate
        try:
            prompt_for_config(context, no_input=True)
        except UndefinedVariableInTemplate as e:
            assert "Unable to render variable 'project_name'" in str(e)

def test_prompt_for_config_handles_lists_as_choices():
    context = {
        'cookiecutter': {
            'version': ['1.0', '2.0'],
        }
    }
    with patch('cookiecutter.prompt.prompt_choice_for_config') as mock_choice:
        mock_choice.return_value = '1.0'
        result = prompt_for_config(context, no_input=True)
        assert result['version'] == '1.0'

def test_prompt_for_config_handles_dicts():
    context = {
        'cookiecutter': {
            'config_dict': {'key': 'value'},
        }
    }
    # When no_input is True, it skips read_user_dict and just renders.
    result = prompt_for_config(context, no_input=True)
    assert result['config_dict'] == {'key': 'value'}

def test_prompt_for_config_with_prompts_extraction():
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '__prompts__': {'project_name': 'Enter project name: '}
        }
    }
    # Ensure __prompts__ is popped from the internal cookiecutter dict used for iteration
    with patch('cookiecutter.prompt.read_user_variable') as mock_read:
        mock_read.return_value = 'new_name'
        result = prompt_for_config(context, no_input=False)
        assert result['project_name'] == 'new_name'
        # The context passed to the function should have had __prompts__ removed from its cookiecutter key
        assert '__prompts__' not in context['cookiecutter']

def test_prompt_for_config_with_boolean_no_input():
    context = {
        'cookiecutter': {
            'is_enabled': True,
        }
    }
    result = prompt_for_config(context, no_input=True)
    assert result['is_enabled'] is True

def test_prompt_for_config_with_boolean_with_input():
    context = {
        'cookiecutter': {
            'is_enabled': True,
        }
    }
    with patch('cookiecutter.prompt.read_user_yes_no') as mock_yes_no:
        mock_yes_no.return_value = False
        result = prompt_for_config(context, no_input=False)
        assert result['is_enabled'] is False
```


# LLM-generated content at query #19
#--------------------------

```python
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.prompt import prompt_and_delete

@patch("cookiecutter.prompt.os.path.isdir")
@patch("cookiecutter.prompt.os.remove")
@patch("cookiecutter.prompt.rmtree")
@patch("cookiecutter.prompt.read_user_yes_no")
def test_prompt_and_delete_no_input_true(mock_read, mock_rmtree, mock_remove, mock_isdir):
    mock_read.assert_not_called()
    result = prompt_and_delete("test_path", no_input=True)
    assert result is True
    mock_rmtree.assert_not_called()
    mock_remove.assert_not_called()

@patch("cookiecutter.prompt.os.path.isdir")
@patch("cookiecutter.prompt.os.remove")
@patch("cookiecutter.prompt.rmtree")
@patch("cookiecutter.prompt.read_user_yes_no")
def test_prompt_and_delete_dir_deleted(mock_read, mock_rmtree, mock_remove, mock_isdir):
    mock_isdir.return_value = True
    mock_read.return_value = True
    result = prompt_and_delete("test_dir")
    assert result is True
    mock_rmtree.assert_called_once_with("test_dir")

@patch("cookiecutter.prompt.os.path.isdir")
@patch("cookiecutter.prompt.os.remove")
@patch("cookiecutter.prompt.rmtree")
@patch("cookiecutter.prompt.read_user_yes_no")
def test_prompt_and_delete_file_deleted(mock_read, mock_rmtree, mock_remove, mock_isdir):
    mock_isdir.return_value = False
    mock_read.return_value = True
    result = prompt_and_delete("test_file")
    assert result is True
    mock_remove.assert_called_once_with("test_file")

@patch("cookiecutter.prompt.os.path.isdir")
@patch("cookiecutter.prompt.os.remove")
@patch("cookiecutter.prompt.rmtree")
@patch("cookiecutter.prompt.read_user_yes_no")
def test_prompt_and_delete_do_not_delete_reuse_existing(mock_read, mock_rmtree, mock_remove, mock_isdir):
    mock_isdir.return_value = True
    mock_read.side_effect = [False, True]
    result = prompt_and_delete("test_dir")
    assert result is False
    mock_rmtree.assert_not_called()

@patch("cookiecutter.prompt.os.path.isdir")
@patch("cookiecutter.prompt.os.remove")
@patch("cookiecutter.prompt.rmtree")
@patch("cookiecutter.prompt.read_user_yes_no")
@patch("sys.exit")
def test_prompt_and_delete_do_not_delete_exit_program(mock_exit, mock_read, mock_rmtree, mock_remove, mock_isdir):
    mock_isdir.return_value = True
    mock_read.side_effect = [False, False]
    result = prompt_and_delete("test_dir")
    assert result is None
    mock_exit.assert_called_once()
```


