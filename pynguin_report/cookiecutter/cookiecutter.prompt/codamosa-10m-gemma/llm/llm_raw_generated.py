####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import OrderedDict

@pytest.mark.parametrize("context, repo_dir, no_input, expected_path", [
    # Case 1: New style (templates dict) with no input
    (
        {
            "cookiecutter": {
                "templates": {
                    "template_a": {"path": "sub/template_a", "title": "A"},
                    "template_boop": {"path": "sub/template_b", "title": "B"},
                }
            }
        },
        "/tmp/repo",
        True,
        str(Path("/tmp/repo/sub/template_a").resolve()),
    ),
    # Case 2: Old style (list of strings) with no input
    (
        {
            "cookiecutter": {
                "template": ["Choice One (path/to/one)", "Choice Two (path/to/two)"],
            }
        },
        "/tmp/repo",
        True,
        str(Path("/tmp/repo/path/to/one").resolve()),
    ),
])
def test_choose_nested_template(context, repo_dir, no_input, expected_path):
    # Mock create_env_with_context to avoid Jinja2 complexity
    with patch("cookiecutter.utils.create_env_with_context", return_value=MagicMock()):
        result = choose_nested_template(context, repo_dir, no_input)
        assert result == expected_path

def test_choose_nested_template_error_on_absolute_path():
    # Case 3: Error when template path is absolute
    context = {
        "cookiecutter": {
            "templates": {"t1": {"path": "/abs/path", "title": "T1"}}
        }
    }
    with pytest.raises(ValueError, match="Illegal template path"):
        choose_nested_template(context, "/tmp/repo", no_input=True)

def test_choose_nested_template_error_on_empty_config():
    # Case 4: Error when both 'templates' and 'template' are missing/empty
    context = {"cookiecutter": {}}
    with pytest.raises(Exception): # Either ValueError from prompt_choice or IndexError
        choose_nested_template(context, "/tmp/repo", no_input=True)

def test_choose_nested_template_interaction():
    # Case 5: Testing the interactive flow (simulating user input)
    context = {
        "cookiecutter": {
            "templates": {
                "t1": {"path": "tpl1", "title": "T1"},
                "t2": {"path": "tpl2", "title": "T2"},
            }
        }
    }
    # Mocking read_user_choice to simulate user selecting the second option ('2')
    with patch("cookiecutter.utils.create_env_with_context", return_value=MagicMock()), \
         patch("read_user_choice", return_value="t2"):
        result = choose_nested_template(context, "/tmp/repo", no_input=False)
        assert result == str(Path("/tmp/repo/tpl2").resolve())

def test_choose_nested_template_old_style_regex():
    # Case 6: Ensure regex correctly extracts path from old style strings
    context = {
        "cookiecutter": {
            "template": ["Option (extracted/path)"],
        }
    }
    with patch("cookiecutter.utils.create_env_with_context", return_value=MagicMock()), \
         patch("read_user_choice", return_value="0"): # '0' is the key in the choice map for first item
        result = choose_nested_template(context, "/tmp/repo", no_input=False)
        assert result == str(Path("/tmp/repo/extracted/path").resolve())
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_dict():
    # Test case 1: Basic functionality with default value and no prompts
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '{"key": "value"}'
        result = read_user_dict("my_var", {"default": "val"})
        assert result == {"key": "value"}
        mock_ask.assert_called_once()
        # Check if the prompt includes the (default) notation
        args, kwargs = mock_ask.call_args
        assert "my_var" in args[0]
        assert kwargs["default"] == {"default": "val"}

    # Test case 2: Using custom prompts
    prompts = {"my_var": "Custom Question"}
    with patch("rich.prompt.Prompt.ask") as mock_arg:
        mock_arg.return_value = '{"a": 1}'
        result = read_user_dict("my_var", {"a": 1}, prompts=prompts)
        assert result == {"a": 1}
        # The prompt string should be the custom question + default display suffix
        assert "Custom Question" in mock_arg.call_args[0][0]

    # Test case 3: Using prefix
    prefix = "[bold]Prefix: [/]"
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '{}'
        read_user_dict("my_var", {}, prefix=prefix)
        args, _ = mock_ask.call_args
        assert args[0].startswith(prefix)

    # Test case 4: Invalid default_value type (should raise TypeError)
    with pytest.raises(TypeError):
        read_user_dict("my_var", ["not", "a", "dict"])

    # Test case 5: Verify JsonPrompt behavior via the function call
    # This tests if the underlying JsonPrompt.process_response handles JSON correctly
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '{"nested": {"key": "val"}}'
        result = read_user_dict("my_var", {})
        assert result == {"nested": {"key": "val"}}

    # Test case 6: Invalid JSON input (should raise InvalidResponse from process_json)
    from rich.prompt import InvalidResponse
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '{"invalid": json' # Broken JSON
        with pytest.raises(InvalidResponse, match="Unable to decode to JSON"):
            read_user_dict("my_var", {})

    # Test case 7: JSON is valid but not a dictionary (should raise InvalidResponse)
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '["not", "a", "dict"]'
        with pytest.raises(InvalidResponse, match="Requires JSON dict"):
            read_user_dict("my_var", {})
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from collections import OrderedDict

@pytest.mark.parametrize("context, repo_dir, no_input, expected_path", [
    # Case 1: New style with templates dictionary (returns path from config)
    (
        {
            "cookiecutter": {
                "templates": {
                    "template_a": {"path": "subdir/template_a", "title": "A"},
                    "template_bo": {"path": "subdir/template_b", "title": "B"}
                }
            }
        },
        "/tmp",
        True,
        str(Path("/tmp/subdir/template_a").resolve())
    ),
    # Case 2: Old style with template list (regex extracts path from string)
    (
        {
            "cookiecutter": {
                "template": ["template_one (path/to/template_one)", "template_two (path/to/template_two)"]
            }
        },
        "/tmp",
        True,
        str(Path("/tmp/path/to/template_one").resolve())
    ),
])
def test_choose_nested_template(context, repo_dir, no_input, expected_path):
    with patch("cookiecutter.utils.create_env_with_context") as mock_env:
        mock_env.return_value = MagicMock()
        result = choose_nested_template(context, repo_dir, no_input=no_input)
        assert result == expected_path

def test_choose_nested_template_error():
    # Case 3: Illegal template path (absolute path)
    context = {
        "cookiecutter": {
            "templates": {"a": {"path": "/absolute/path"}}
        }
    }
    with pytest.raises(ValueError, match="Illegal template path"):
        choose_nested_template(context, "/tmp", no_input=True)

def test_choose_nested_template_regex_failure():
    # Case 4: Old style where regex fails to find parentheses
    context = {
        "cookiecutter": {
            "template": ["invalid_format"]
        }
    }
    with pytest.raises(AttributeError):
        choose_nested_template(context, "/tmp", no_input=True)

def test_choose_nested_template_interaction():
    # Case 5: Interactive mode (no_input=False)
    context = {
        "cookiecutter": {
            "templates": {
                "opt1": {"path": "t1"},
                "opt2": {"path": "t2"}
            }
        }
    }
    # Mock read_user_choice to simulate user selecting the second option ('2')
    with patch("cookiecutter.utils.create_env_with_context"), \
         patch("prompting.read_user_choice", return_value="opt2") as mock_choice:
        result = choose_nested_template(context, "/tmp", no_input=False)
        assert "t2" in result
        mock_choice.assert_called_once()
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from collections import OrderedDict
from jinja2 import Environment
from cookiecutter.exceptions import UndefinedVariableInTemplate

@pytest.fixture
def sample_context():
    return {
        "cookiecutter": {
            "project_name": "my_project",
            "use_git": True,
            "options": ["option1", "option2"],
            "__prompts__": {
                "project_name": "Enter Project Name"
            }
        }
    }

@pytest.fixture
def mock_env():
    env = MagicMock(spec=Environment)
    template = MagicMock()
    template.render.return_value = "rendered_value"
    env.from_string.return_value = template
    return env

@patch("cookiecutter.utils.create_env_with_context")
@patch("prompting.read_user_variable")
@patch("prompting.read_user_yes_no")
@patch("prompting.prompt_choice_for_config")
def test_prompt_for_config(
    mock_choice, 
    mock_yes_no, 
    mock_var, 
    mock_create_env, 
    sample_context, 
    mock_env
):
    # Setup mocks
    mock_create_env.return_value = mock_env
    mock_var.return_value = "user_input_val"
    mock_yes_no.return_value = True
    mock_choice.return_value = "option1"

    # Execution (no_input=False triggers prompts)
    result = prompt_for_config(sample_context, no_input=False)

    # Assertions
    assert isinstance(result, OrderedDict)
    assert result["project_name"] == "user_input_val"
    assert result["use_git"] is True
    assert result["options"] == "option1"
    
    # Verify interaction with prompts
    assert mock_var.called
    assert mock_yes_no.called

@patch("cookiecutter.utils.create_env_with_context")
def test_prompt_for_config_no_input(mock_create_env, sample_context, mock_env):
    # Setup mocks
    mock_create_env.return_value = mock_env
    
    # Override context to prevent actual prompting logic from needing user input
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "use_git": True,
            "__prompts__": {}
        }
    }

    # Execution (no_input=True should use values directly from context)
    result = prompt_for_config(context, no_input=True)

    assert result["project_name"] == "my_project"
    assert result["use_git"] is True

@patch("cookiecutter.utils.create_env_with_context")
def test_prompt_for_config_undefined_error(mock_create_env, sample_context, mock_env):
    from jinja2.exceptions import UndefinedError
    
    # Setup mocks to raise UndefinedError during rendering
    mock_create_env.return_value = mock_env
    mock_env.from_string.side_effect = UndefinedError("Template error")
    
    context = {
        "cookiecutter": {
            "broken_var": "{{ non_existent_var }}"
        }
    }

    with pytest.raises(UndefinedVariableInTemplate):
        prompt_for_config(context, no_input=False)

@patch("cookiecutter.utils.create_env_with_context")
@patch("prompting.JsonPrompt.ask")
def test_prompt_for_config_dict_handling(mock_json_ask, mock_create_env, mock_env):
    mock_create_env.return_value = mock_env
    mock_json_ask.return_value = {"key": "value"}

    context = {
        "cookiecutter": {
            "my_dict": {"subkey": "subval"},
            "__prompts__": {}
        }
    }

    result = prompt_for_config(context, no_input=False)
    assert result["my_dict"] == {"key": "value"}
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

def test_read_user_dict():
    """Test the read_user_dict function with various scenarios."""
    
    # Scenario 1: Basic functionality with no prompts or prefix
    # We mock JsonPrompt.ask to return a specific dictionary
    with patch("JsonPrompt.ask") as mock_ask:
        mock_ask.return_value = {"key": "value"}
        result = read_user_dict("my_var", default_value={"default": "val"})
        
        assert result == {"key": "value"}
        # Check if it called ask with the correct question (var_name + suffix)
        mock_ask.assert_called_once()
        args, kwargs = mock_ask.call_args
        assert "my_var" in args[0]
        assert kwargs["default"] == {"default": "val"}

    # Scenario 2: Functionality with prompts and prefix
    prompts = {"my_var": "Friendly Question"}
    prefix = "PROMPT: "
    with patch("JsonPrompt.ask") as mock_ask:
        mock_ask.return_value = {"a": 1}
        result = read_user_dict("my_var", default_value={}, prompts=prompts, prefix=prefix)
        
        assert result == {"a": 1}
        # The question should be the prompt string + the (default) suffix
        expected_question = f"{prefix}Friendly Question [cyan bold]({DEFAULT_DISPLAY})[/]"
        args, kwargs = mock_ask.call_args
        assert args[0] == expected_question

    # Scenario 3: Functionality when var_name is not in prompts (fallback to var_name)
    with patch("JsonPrompt.ask") as mock_ask:
        mock_ask.return_value = {}
        result = read_user_dict("simple_var", default_value={})
        
        args, kwargs = mock_ask.call_args
        assert "simple_var" in args[0]
        assert "[cyan bold]" in args[0]

    # Scenario 4: Type error when default_value is not a dict
    with pytest.raises(TypeError):
        read_user_dict("my_var", default_value="not_a_dict")

    # Scenario 5: Verifying prompt prefixing with existing prompts
    prompts = {"my_var": "Custom Prompt"}
    prefix = "[TEST] "
    with patch("JsonPrompt.ask") as mock_ask:
        mock_ask.return_value = {}
        read_user_dict("my_var", default_value={}, prompts=prompts, prefix=prefix)
        
        args, _ = mock_ask.call_args
        assert args[0].startswith("[TEST] Custom Prompt")
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import patch
from rich.prompt import InvalidResponse

def test_YesNoPrompt():
    prompt = YesNoPrompt("Question?")
    
    # Test valid 'yes' inputs
    for val in ["y", "yes", "true", "1", "t", "on", "  YES  "]:
        assert prompt.process_response(val) is True
        
    # Test valid 'no' inputs
    for val in ["n", "no", "false", "0", "f", "off", "  NO  "]:
        assert prompt.process_response(val) is False
        
    # Test invalid input raises InvalidResponse
    with pytest.raises(InvalidResponse):
        prompt.process_response("maybe")
    
    with pytest.raises(InvalidResponse):
        prompt.process_response("")

    with pytest.raises(InvalidResponse):
        prompt.process_response("random_string")
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch
from collections import OrderedDict

def test_read_user_choice():
    options = ["Option A", "Option B", "Option C"]
    
    # Test Case 1: Basic functionality with standard input
    # We mock Prompt.ask to return the key '2' which corresponds to 'Option B'
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "2"
        result = read_user_choice("my_var", options)
        assert result == "Option B"
        # Verify Prompt.ask was called with expected structure
        args, kwargs = mock_ask.call_args
        assert "Select my_var" in args[0]
        assert kwargs["choices"] == ["1", "2", "3"]

    # Test Case 2: Using custom prompts (string)
    prompts = {"my_var": "Custom Question"}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return<0xA0>return_value = "1"
        result = read_user_choice("my_var", options, prompts=prompts)
        assert result == "Option A"
        assert "Custom Question" in mock_ask.call_args[0][0]

    # Test Case 3: Using custom prompts (dictionary/complex)
    # This tests the logic where prompts[var_name] is a dict with __prompt__
    prompts_dict = {
        "my_var": {
            "__prompt__": "Pick one",
            "1": "Custom Label 1",
            "2": "Custom Label 2"
        }
    }
    # Note: read_user_choice builds its own choice_map based on the 'options' list passed,
    # but it uses prompts[var_name] to override the text of the lines.
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"
        result = read_user_choice("my_var", ["Alpha", "Beta"], prompts=prompts_dict)
        assert result == "Alpha"
        # Check if the formatted string contains the custom labels from the prompt dict
        full_prompt_text = mock_ask.call_args[0][0]
        assert "Pick one" in full_prompt_text
        assert "Custom Label 1" in full_prompt_text

    # Test Case 4: Using prefix
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "3"
        result = read_user_choice("my_var", options, prefix="[bold]Prefix: [/]")
        assert result == "Option C"
        assert "[bold]Prefix: [/]Select my_var" in mock_ask.call_args[0][0]

    # Test Case 5: Error handling for empty options
    with pytest.raises(ValueError):
        read_user_choice("my_var", [])

    # Test Case 6: Default value behavior (the first item in choice_map)
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"
        result = read_user_choice("my_var", options)
        assert result == "Option A"
        # The default in Prompt.ask should be the first key '1'
        assert mock_ask.call_args[1]["default"] == "1"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import json
from collections import OrderedDict
from rich.prompt import InvalidResponse

def test_process_json():
    # Test valid JSON string representing a dictionary
    valid_json = '{"key": "value", "number": 123, "bool": true}'
    expected_output = OrderedDict([
        ("key", "value"),
        ("number", 123),
        ("bool", True)
    ])
    assert process_json(valid_json) == expected_output

    # Test valid JSON string representing a different type (list) - should raise InvalidResponse
    list_json = '["item1", "item2"]'
    with pytest.raises(InvalidResponse, match="Requires JSON dict."):
        process_json(list_json)

    # Test invalid JSON syntax - should raise InvalidResponse
    invalid_syntax_json = '{"key": "value", broken}'
    with pytest.raises(InvalidResponse, match="Unable to decode to JSON."):
        process_json(invalid_syntax_json)

    # Test valid JSON string representing a primitive (string) - should raise InvalidResponse
    primitive_json = '"just a string"'
    with pytest.raises(InvalidResponse, match="Requires JSON dict."):
        process_json(primitive_json)

    # Test empty dictionary
    empty_json = '{}'
    assert process_json(empty_json) == {}
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_yes_no():
    """Tests the read_user_yes_no function behavior with various inputs."""
    
    # Test case 1: User enters 'y' (should return True)
    with patch("rich.prompt.Confirm.ask", return_value="y"):
        assert read_user_yes_no("test_var", default_value=False) is True

    # Test case 2: User enters 'no' (should return False)
    with patch("rich.prompt.Confirm.ask", return_value="no"):
        assert read_user_yes_no("test_var", default_value=True) is False

    # Test case 3: Using custom prompts dictionary
    prompts = {"my_var": "Custom Question"}
    with patch("rich.prompt.Confirm.ask") as mocked_ask:
        mocked_ask.return_value = "yes"
        result = read_user_yes_no("my_var", default_value=False, prompts=prompts)
        # Verify the prompt string contains the custom question
        args, kwargs = mocked_ask.call_args
        assert "Custom Question" in args[0]
        assert result is True

    # Test case 4: Using default value when no input is provided (simulated by return_value)
    with patch("rich.prompt.Confirm.ask", return_value="true"):
        assert read_user_yes_no("test_var", default_value=False) is True

    # Test case 5: Testing 'on' as a valid yes input (from YesNoPrompt logic)
    with patch("rich.prompt.Confirm.ask", return_value="on"):
        assert read_user_yes_no("test_var", default_value=False) is True

    # Test case 6: Testing '0' as a valid no input (from YesNoPrompt logic)
    with patch("rich.prompt.Confirm.ask", return_value="0"):
        assert read_user_yes_no("test_var", default_value=True) is False

    # Test case 7: Testing prefix usage
    prefix = "PROMPT: "
    with patch("rich.prompt.Confirm.ask") as mocked_ask:
        mocked_ask.return_value = "y"
        read_user_yes_no("var", default_value=False, prefix=prefix)
        args, kwargs = mocked_ask.call_args
        assert args[0].startswith(prefix)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_YesNoPrompt():
    prompt = YesNoPrompt("Test Question")
    
    # Test truthy values
    for true_val in ["1", "true", "t", "yes", "y", "on"]:
        with patch("rich.prompt.Confirm.ask", return_value=true_val):
            assert prompt.process_response(true_val) is True

    # Test falsy values
    for false_val in ["0", "false", "f", "no", "n", "off"]:
        with patch("rich.prompt.Confirm.ask", return_value=false_val):
            assert prompt.process_response(false_val) is False

    # Test case insensitivity and whitespace
    with patch("rich.prompt.Confirm.ask", return_value="  YES  "):
        assert prompt.process_response("  YES  ") is True
    
    with patch("rich.prompt.Confirm.ask", return_value="No"):
        assert prompt.process_response("No") is False

    # Test invalid response raises InvalidResponse
    from rich.prompt import InvalidResponse
    with pytest.raises(InvalidResponse):
        prompt.process_response("maybe")
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import json
from collections import OrderedDict
from rich.prompt import InvalidResponse

def test_process_json():
    # Test valid JSON string representing a dictionary
    valid_json = '{"key": "value", "number": 123, "bool": true}'
    expected_output = OrderedDict([
        ("key", "value"),
        ("number", 123),
        ("bool", True)
    ])
    assert process_json(valid_json) == expected_output

    # Test valid JSON string representing a nested dictionary
    nested_json = '{"outer": {"inner": "content"}}'
    expected_nested = OrderedDict([
        ("outer", OrderedDict([("inner", "content")]))
    ])
    assert process_json(nested_json) == expected_nested

    # Test invalid JSON syntax (syntax error)
    invalid_syntax = '{"key": "value",}'  # Trailing comma
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(invalid_syntax)
    assert "Unable to decode to JSON" in str(excinfo.value)

    # Test valid JSON but not a dictionary (e.g., a list)
    not_a_dict_json = '["item1", "item2"]'
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(not_a_dict_json)
    assert "Requires JSON dict" in str(excinfo.value)

    # Test valid JSON but not a dictionary (e.g., a string)
    string_json = '"just a string"'
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(string_json)
    assert "Requires JSON dict" in str(excinfo.value)

    # Test valid JSON but not a dictionary (e.g., an integer)
    int_json = '123'
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(int_json)
    assert "Requires JSON dict" in str(excinfo.value)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch

@pytest.mark.parametrize("input_val, expected", [
    ("yes", True),
    ("y", True),
    ("1", True),
    ("true", True),
    ("t", True),
    ("on", True),
    ("no", False),
    ("n", False),
    ("0", False),
    ("false", False),
    ("f", False),
    ("off", False),
])
def test_read_user_yes_no(input_val, expected):
    """Test that read_user_yes_no correctly parses various truthy and falsy inputs."""
    with patch("rich.prompt.Confirm.ask") as mock_ask:
        mock_ask.return_value = input_val
        result = read_user_yes_no("test_var", default_value=True)
        assert result is expected

def test_read_user_yes_no_with_prompts():
    """Test that read_user_yes_no uses the custom prompt from the prompts dictionary."""
    prompts = {"my_var": "Custom Question"}
    with patch("rich.prompt.Confirm.ask") as mock_ask:
        mock_ask.return_value = "yes"
        read_user_yes_no("my_var", default_value=True, prompts=prompts)
        # Check that the question passed to Prompt.ask includes the custom prompt
        args, kwargs = mock_ask.call_args
        assert "Custom Question" in args[0]

def test_read_user_yes_no_uses_var_name_as_default():
    """Test that read_user_yes_no uses the var_name itself if no prompts are provided."""
    with patch("rich.prompt.Confirm.ask") as mock_ask:
        mock_ask.return_value = "no"
        read_user_yes_no("simple_var", default_value=False)
        args, kwargs = mock_ask.call_args
        assert "simple_var" in args[0]

def test_read_user_yes_no_with_prefix():
    """Test that the prefix is correctly prepended to the question."""
    with patch("rich.prompt.Confirm.ask") as mock_ask:
        mock_ask.return_value = "true"
        read_user_yes_no("var", default_value=True, prefix="PRE: ")
        args, kwargs = mock_ask.call_args
        assert args[0].startswith("PRE: var")
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from collections import OrderedDict
from jinja2 import Environment

@pytest.fixture
def mock_env():
    env = Environment()
    return env

@pytest.fixture
def sample_context():
    return {
        'cookiecutter': {
            'project_name': 'my_project',
            'version': '0.1.0',
            'use_git': True,
            'author': 'developer',
            '_private_var': 'secret',
            '__meta__': 'meta_info'
        }
    }

def test_prompt_for_config(mock_env, sample_context):
    """
    Tests prompt_for_config with no_input=True to verify the logic of 
    rendering variables and populating the cookiecutter dictionary
    without interactive prompts.
    """
    # We use no_input=True to avoid needing to mock Prompt.ask for every single variable
    # This tests the rendering, the loop logic, and the handling of private/meta keys.
    
    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        result = prompt_for_config(sample_context, no_input=True)
        
        # Check that the result is an OrderedDict
        assert isinstance(result, OrderedDict)
        
        # Check that public variables are present
        assert result['project_name'] == 'my_project'
        assert result['version'] == '0.1.0'
        assert result['use_git'] is True
        assert result['author'] == 'developer'
        
        # Check that private variables (starting with _) are present but not processed as prompts
        assert result['_private_var'] == 'secret'
        
        # Check that __prefixed__ variables were rendered/processed
        # In our case, if no input is provided, they should be returned as rendered strings
        assert '__meta__' in result

def test_prompt_for_config_with_choices(mock_env):
    """
    Tests prompt_for_config when a list (choice) is present.
    """
    context = {
        'cookiecutter': {
            'template_type': ['web', 'api', 'cli']
        }
    }
    
    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        # Mocking read_user_choice to simulate selecting 'api'
        with patch('rich.prompt.Prompt.ask', return_value='2'): # '2' corresponds to 'api' in choice_map
            result = prompt_for_config(context, no_input=False)
            assert result['template_type'] == 'api'

def test_prompt_for_config_error_handling(mock_env):
    """
    Tests that UndefinedError is correctly caught and re-raised 
    as UndefinedVariableInTemplate.
    """
    from cookiecutter.exceptions import UndefinedVariableInTemplate
    
    context = {
        'cookiecutter': {
            'broken_var': '{{ non_existent_variable }}'
        }
    }
    
    # Mocking the context to cause an undefined error during rendering
    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        with pytest.raises(UndefinedVariableInTemplate):
            prompt_for_config(context, no_input=True)

def test_prompt_for_config_dict_processing(mock_env, sample_context):
    """
    Tests the second pass of prompt_for_config which handles dictionary types.
    """
    context = {
        'cookiecutter': {
            'settings': {'mode': 'production', 'debug': False}
        }
    }
    
    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        # Mock JsonPrompt.ask to return the dict directly (simulating valid JSON input)
        with patch('JsonPrompt.ask', return_value={'mode': 'production', 'debug': False}):
            result = prompt_for_config(context, no_input=False)
            assert result['settings']['mode'] == 'production'
            assert result['settings']['debug'] is False
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

@pytest.mark.parametrize("path_val, is_dir", [
    ("test_file.txt", False),
    ("test_dir", True),
])
def test_prompt_and_delete(path_val, is_dir):
    """Test the prompt_and_delete function for various scenarios."""
    
    # Scenario 1: no_input=True (Automatic deletion)
    with patch("os.path.isdir", return_value=is_dir), \
         patch("os.remove" if not is_dir else "shutil.rmtree") as mock_delete, \
         patch("cookiecutter.utils.rmtree" if is_dump_dir_logic(is_dir) else "builtins.print"): 
        # Note: Since rmtree is imported from cookiecutter.utils in the source, we patch that specifically
        pass

    # Re-writing actual test logic due to complex dependencies in the provided snippet
    
    # Setup Mock Path
    mock_path = MagicMock(spec=Path)
    
    # --- TEST CASE 1: no_input=True, path is a file, delete succeeds ---
    with patch("os.path.isdir", return_value=False), \
         patch("os.remove") as mock_remove:
        assert prompt_and_delete(path_val, no_input=True) is True
        mock_remove.assert_called_once()

    # --- TEST CASE 2: no_input=True, path is a directory, delete succeeds ---
    with patch("os.path.isdir", return_value=True), \
         patch("cookiecutter.utils.rmtree") as mock_rmtree:
        assert prompt_and_delete(path_val, no_input=True) is True
        mock_rmtree.assert_called_once()

    # --- TEST CASE 3: no_input=False, user says YES to delete ---
    with patch("os.path.isdir", return_value=False), \
         patch("os.remove"), \
         patch("read_user_yes_no", return_value=True):
        assert prompt_and_delete(path.Path(path_val), no_input=False) is True

    # --- TEST CASE 4: no_input=False, user says NO to delete, but YES to reuse ---
    with patch("os.path.isdir", return_value=False), \
         patch("read_user_yes_no", side_effect=[False, True]):
        assert prompt_and_delete(path.Path(path_val), no_input=False) is False

    # --- TEST CASE 5: no_input=False, user says NO to delete, and NO to reuse (System Exit) ---
    with patch("os.path.isdir", return_value=False), \
         patch("read_user_yes_no", side_effect=[False, False]), \
         patch("sys.exit") as mock_exit:
        assert prompt_and_delete(path.Path(path_val), no_input=False) is None
        mock_exit.assert_called_once()

def is_dump_dir_logic(is_dir):
    # Helper for the patch logic in the test above
    return is_dir
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

@pytest.mark.parametrize("input_val, should_delete, expected_return", [
    ("yes", True, True),
    ("no", False, None), # Will trigger second prompt or sys.exit
])
def test_prompt_and_delete(input_val, should_delete, expected_return):
    """Test the deletion logic of prompt_and_delete."""
    test_path = Path("/tmp/test_dir")
    
    # Mocking file existence and os operations
    with patch("os.path.isdir", return_value=True), \
         patch("cookiecutter.utils.rmtree") as mock_rmtree, \
         patch("os.remove") as mock_remove, \
         patch("read_user_yes_no", return_value=input_val == "yes"), \
         patch("sys.exit") as mock_exit:
        
        # Scenario 1: User says yes to deleting (no_input=False)
        if input_val == "yes":
            result = prompt_and_delete(str(test_path), no_input=False)
            assert result is True
            mock_rmtree.assert_called_once_with(test_path)
        
        # Scenario 2: User says no to deleting, then yes to reusing
        else:
            # We need a second mock for the 're-use' prompt
            with patch("read_user_yes_no", side_effect=["no", "yes"]):
                result = prompt_and_delete(str(test_path), no_input=False)
                assert result is False
                mock_exit.assert_not_called()

        # Scenario 3: User says no to deleting, then no to reusing (Exit)
        with patch("read_user_yes_no", side_effect=["no", "no"]):
            prompt_and_delete(str(test_path), no_input=False)
            mock_exit.assert_called()

def test_prompt_and_delete_no_input():
    """Test that prompt_and_delete deletes automatically when no_input is True."""
    test_file = Path("/tmp/test_file.txt")
    
    with patch("os.path.isdir", return_value=False), \
         patch("os.remove") as mock_remove:
        
        result = prompt_and_delete(str(test_file), no_input=True)
        
        assert result is True
        mock_remove.assert_called_once()

def test_prompt_and_delete_file_path():
    """Test that the function handles files correctly using os.remove."""
    test_file = Path("/tmp/test_file.txt")
    
    with patch("os.path.isdir", return_value=False), \
         patch("os.remove") as mock_remove:
        
        # Force user to say yes via side_effect if we weren't using no_input=True
        with patch("read_user_yes_no", return_value="yes"):
            result = prompt_and_delete(str(test_file), no_input=False)
            assert result is True
            mock_remove.assert_called_once()
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import patch
from collections import OrderedDict

def test_read_user_choice():
    options = ["Option A", "Option B", "Option C"]
    
    # Test Case 1: Basic functionality with manual selection
    # We mock Prompt.ask to return '2', which maps to 'Option B' in our loop
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "2"
        result = read_user_choice("my_var", options)
        assert result == "Option B"
        # Verify the prompt structure contains the expected question
        args, kwargs = mock_ask.call_args
        assert "Select my_var" in args[0]
        assert kwargs["choices"] == ["1", "2", "3"]

    # Test Case 2: Using custom prompts (string)
    prompts = {"my_var": "Please choose a flavor"}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"
        result = read_user_choice("my_var", options, prompts=prompts)
        assert result == "Option A"
        assert "Please choose a flavor" in mock_ask.call_args[0][0]

    # Test Case 3: Using custom prompts (dictionary with __prompt__)
    prompts = {"my_var": {"__prompt__": "Pick one", "1": "First"}}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"
        result = read_user_choice("my_var", options, prompts=prompts)
        assert result == "Option A"
        # Verify that the choice lines were reconstructed from the prompt dict
        prompt_text = mock_ask.call_args[0][0]
        assert "[bold magenta]1[/] - [bold]First[/]" in prompt_text

    # Test Case 4: Using prefix
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "3"
        result = read_user_choice("my_var", options, prefix="[blue]PROMPT:[/] ")
        assert result == "Option C"
        assert "[blue]PROMPT:[/] Select my_var" in mock_ask.call_args[0][0]

    # Test Case 5: Error handling for empty options
    with pytest.raises(ValueError):
        read_user_choice("my_var", [])

    # Test Case 6: Default behavior (first item)
    # Note: the code uses next(iter(choices)) as default, which is '1'
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"
        result = read_user_choice("my_var", options)
        assert result == "Option A"
        assert kwargs["default"] == "1"
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_variable():
    # Test case 1: Basic usage with default value and no prompts
    with patch("rich.prompt.Prompt.ask", return_value="my_value") as mock_ask:
        result = read_user_variable("project_name", "default_val")
        assert result == "my_value"
        mock_ask.assert_called_once_with("project_name", default="default_val")

    # Test case 2: Usage with custom prompts dictionary
    prompts = {"project_name": "Enter your project name:"}
    with patch("rich.prompt.Prompt.ask", return, value="new_name") as mock_ask:
        # We need to handle the side_effect to simulate a returned value
        mock_ask.side_effect = ["new_name"]
        result = read_user_variable("project_name", "default_val", prompts=prompts)
        assert result == "new_name"
        mock_ask.assert_called_once_with("Enter your project name:", default="default_val")

    # Test case 3: Usage with prefix
    with patch("rich.prompt.Prompt.ask", return_value="val") as mock_ask:
        result = read_user_variable("var", "def", prefix="PRE: ")
        assert result == "val"
        mock_ask.assert_called_once_with("PRE: var", default="def")

    # Test case 4: Testing the loop (simulating a None return then a valid return)
    # Note: Prompt.ask in rich usually doesn't return None for standard inputs, 
    # but the function logic explicitly checks `if variable is not None`.
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.side_effect = [None, "valid_input"]
        result = read_user_variable("var", "def")
        assert result == "valid_input"
        assert mock_ask.call_count == 2

    # Test case 5: Complex prompt (dict mapping)
    prompts = {"var": {"__prompt__": "Custom Prompt"}}
    with patch("rich.prompt.Prompt.ask", return_value="val") as mock_ask:
        result = read_user_variable("var", "def", prompts=prompts)
        assert result == "val"
        # In the current implementation, if it's a dict, it falls back to var name 
        # unless specific logic in read_user_variable handles dict keys.
        # Looking at code: `question = prompts[var_name] if ... else var_name`
        # Since prompts['var'] is a dict, question becomes the dict object.
        # This test verifies it calls Prompt.ask with that object (which stringifies).
        args, kwargs = mock_ask.call_args
        assert "var" in args[0] or isinstance(args[0], dict)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from collections import OrderedDict
from jinja2 import Environment

@pytest.fixture
def mock_env():
    env = Environment()
    return env

@pytest.fixture
def sample_context():
    # We use a dict structure that mimics the cookiecutter context
    # __prompts__ is used by the function to look for custom labels
    return {
        'cookiecutter': {
            'project_name': 'my_project',
            'version': '0.1.0',
            '_private_var': 'hidden',
            '__custom_label__': 'Custom Label',
            '__prompts__': {
                'project_name': 'Enter Project Name'
            }
        }
    }

def test_prompt_for_config(mock_env, sample_context):
    """
    Tests prompt_for_config by mocking the interactive Prompt.ask calls.
    We simulate a non-interactive flow (no_input=True) to verify 
    the logic of variable rendering and dictionary construction.
    """
    # Patching create_env_with_context to return our mock env
    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        # We set no_input=True so it doesn't enter the infinite loops of Prompt.ask
        # and uses the default values provided in the context.
        result = prompt_for_config(sample_context, no_input=True)
        
        # Verify that private variables (starting with _) are included in result 
        # but were not processed as prompts
        assert result['_private_var'] == 'hidden'
        
        # Verify standard variables are present
        assert result['project_name'] == 'my_project'
        assert result['version'] == '0.1.0'
        
        # Verify that the __rendered__ keys (from template logic) are handled
        # Note: In our context, '__custom_label__' is a key in cookiecutter.
        # The function renders it.
        assert 'Custom Label' in result.values()

def test_prompt_for_config_interactive(mock_env, sample_context):
    """
    Tests prompt_for_config with simulated user input via mocking Prompt.ask.
    """
    context = {
        'cookiecutter': {
            'user_input': 'default_val',
            '__prompts__': {}
        }
    }

    # Mocking the Prompt.ask to return a specific value for user interaction
    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        with patch('rich.prompt.Prompt.ask', return_value='user_provided_val'):
            result = prompt_for_config(context, no_input=False)
            
            assert result['user_input'] == 'user_provided_val'

def test_prompt_for_config_error_handling(mock_env):
    """
    Tests that UndefinedError in rendering raises UndefinedVariableInTemplate.
    """
    # Context with a variable that refers to a non-existent key in cookiecutter
    context = {
        'cookiecutter': {
            'broken_var': '{{ cookiecutter.non_existent }}',
            '__prompts__': {}
        }
    }

    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        from cookiecutter.exceptions import UndefinedVariableInTemplate
        with pytest.raises(UndefinedVariableInTemplate):
            prompt_for_config(context, no_input=False)

def test_prompt_for_config_empty_list_choice(mock_env, sample_context):
    """
    Tests behavior when a choice variable is empty.
    """
    context = {
        'cookiecutter': {
            'choices': [],
            '__prompts__': {}
        }
    }
    
    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        # prompt_choice_for_config raises ValueError if options are empty
        with pytest.raises(ValueError, match="The list of choices is empty"):
            prompt_for_config(context, no_input=True)
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
import json
from collections import OrderedDict
from rich.prompt import InvalidResponse

def test_process_json():
    # Test valid JSON dictionary string
    valid_json = '{"key": "value", "number": 123, "bool": true}'
    expected_output = OrderedDict([
        ("key", "value"),
        ("number", 123),
        ("bool", True)
    ])
    assert process_json(valid_json) == expected_output

    # Test valid JSON with nested structures
    nested_json = '{"outer": {"inner": [1, 2, 3]}}'
    expected_nested = OrderedDict([
        ("outer", OrderedDict([
            ("inner", [1, 2, 3])
        ]))
    ])
    assert process_json(nestedly_json := nested_json) == expected_nested

    # Test invalid JSON syntax (raises InvalidResponse via json.JSONDecodeError)
    invalid_syntax = '{"key": "value",}'  # Trailing comma is invalid in standard JSON
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(invalid_syntax)
    assert "Unable to decode to JSON" in str(excinfo.value)

    # Test valid JSON but not a dictionary (e.g., a list or string)
    not_a_dict = '["item1", "item2"]'
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(not_a_dict)
    assert "Requires JSON dict" in str(excinfo.value)

    # Test valid JSON but a single primitive value
    primitive = '"just a string"'
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(primitive)
    assert "Requires JSON dict" in str(excinfo.value)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_variable():
    # Test Case 1: Return default value when Prompt.ask returns None (simulating empty input)
    # We mock Prompt.ask to return None on first call and 'my_value' on second call
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.side_effect = [None, "my_value"]
        result = read_user_variable("var_name", default_value="default")
        assert result == "my_value"
        assert mock_ask.call_count == 2

    # Test Case 2: Use custom prompt from prompts dictionary
    prompts = {"project_name": "Enter your project name:"}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "Project X"
        result = read_user_variable("project_name", default_value="default", prompts=prompts)
        # Verify the question passed to Prompt.ask contains the custom prompt text
        mock_ask.assert_called_with("Enter your project name:", default="default")
        assert result == "Project X"

    # Test Case 3: Use prefix in the question
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "Value"
        result = read_user_variable("var_name", default_value="default", prefix="[bold]Prompt: ")
        mock_ask.assert_called_with("[bold]Prompt: var_name", default="default")
        assert result == "Value"

    # Test Case 4: Fallback to var_name when no prompts are provided
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "Simple"
        result = read_user_variable("simple_var", default_value="default", prompts=None)
        mock_ask.assert_called_with("simple_var", default="default")
        assert result == "Simple"

    # Test Case 5: Verify immediate return if Prompt.ask returns a valid value immediately
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "Immediate"
        result = read_user_variable("var", "default")
        assert result == "Immediate"
        assert mock_ask.call_count == 1
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_variable():
    """Tests the read_user_variable function for various input scenarios."""
    
    # Scenario 1: Standard behavior - user provides an input via Prompt.ask
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "my_project"
        result = read_user_variable("project_name", default_value="default_val")
        assert result == "my_project"
        mock_ask.assert_called_once_with("project_name", default="default_val")

    # Scenario 2: Using a custom prompt string from the prompts dictionary
    prompts = {"project_name": "Enter your project title:"}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "New Title"
        result = read_user_variable("project_name", default_value="default", prompts=prompts)
        assert result == "New Title"
        mock_ask.assert_called_once_with("Enter your project title:", default="default")

    # Scenario 3: Using a prefix in the prompt
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "value"
        result = read_user_variable("var", "def", prefix="PRE_: ")
        assert result == "value"
        mock_ask.assert_called_once_with("PRE_: var", default="def")

    # Scenario 4: Handling None return (simulating the loop continuing until a value is provided)
    # The function has a 'while True' and breaks only if variable is not None.
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        # First call returns None, second call returns "valid"
        mock_ask.side_effect = [None, "valid"]
        result = read_user_variable("var", "def")
        assert result == "valid"
        assert mock_ask.call_count == 2

    # Scenario 5: Complex prompts dictionary with nested logic (if applicable)
    # Testing the branch where prompts[var_name] exists but is an empty string or similar
    prompts = {"empty_prompt": ""}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "test"
        # If prompt key exists but evaluates to False (like empty string), it falls back to var_name
        result = read_user_variable("empty_prompt", "def", prompts=prompts)
        assert result == "test"
        mock_ask.assert_called_once_with("empty_prompt", default="def")
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

@pytest.mark.parametrize("path_val", ["test_file.txt", "test_dir"])
def test_prompt_and_delete(path_val):
    """Test the prompt_and_delete function for various scenarios."""
    
    # Setup common mocks
    mock_path = Path(path_val)
    
    # Scenario 1: no_input=True, path is a file, should delete file
    with patch("os.remove") as mock_remove, \
         patch("os.path.isdir", return_value=False), \
         patch("prompt_and_delete.__module__", "test_module"): # dummy for scope
        
        # We need to ensure the function is testing the logic of prompt_and_delete
        from your_module import prompt_and_delete 
        
        result = prompt_and_delete(path_val, no_input=True)
        assert result is True
        mock_remove.assert_called_once()

    # Scenario 2: no_input=True, path is a directory, should delete dir
    with patch("cookiecutter.utils.rmtree") as mock_rmtree, \
         patch("os.path.isdir", return_value=True):
        
        from your_module import prompt_and_delete
        result = prompt_and_delete(path_val, no_input=True)
        assert result is True
        mock_rmtree.assert_called_once()

    # Scenario 3: no_input=False, user says 'yes' to delete (via read_user_yes_no)
    with patch("your_module.read_user_yes_no", return_value=True), \
         patch("os.path.isdir", return_value=False), \
         patch("os.remove") as mock_remove:
        
        from your_module import prompt_and_delete
        result = prompt_and_delete(path_val, no_input=False)
        assert result is True
        mock_remove.assert_called_once()

    # Scenario 4: no_input=False, user says 'no' to delete, but 'yes' to reuse
    with patch("your_module.read_user_yes_no", side_effect=[False, True]), \
         patch("os.path.isdir", return_value=True), \
         patch("cookiecutter.utils.rmtree") as mock_rmtree:
        
        from your_module import prompt_and_delete
        result = prompt_and_delete(path_val, no_input=False)
        assert result is False
        # rmtree should NOT be called because user said 'no' to deletion
        mock_rmtree.assert_not_called()

    # Scenario 5: no_input=False, user says 'no' to delete AND 'no' to reuse -> sys.exit()
    with patch("your_module.read_user_yes_no", side_effect=[False, False]), \
         patch("sys.exit") as mock_exit:
        
        from your_module import prompt_and_delete
        prompt_and_delete(path_val, no_input=False)
        mock_exit.assert_called_once()
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import patch
from collections import OrderedDict

def test_read_user_choice():
    options = ["Option A", "Option B", "Option C"]
    prompts = {"my_var": "Please select an option"}
    prefix = "PROMPT: "

    # Test Case 1: Basic functionality with default Prompt.ask behavior (selecting '2')
    # We mock Prompt.ask to return the key of our choice in the mapping
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_assumed_input = "2"
        mock_ask.return_value = mock_assumed_input
        
        result = read_user_choice("my_var", options, prompts=prompts, prefix=prefix)
        
        assert result == "Option B"
        # Verify the prompt string construction
        expected_prompt_start = f"{prefix}Please select an option"
        args, kwargs = mock_ask.call_args
        assert args[0].startswith(expected_prompt_start)
        assert kwargs["choices"] == ["1", "2", "3"]

    # Test Case 2: No prompts provided (uses var_name as question)
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"
        result = read_user_choice("my_var", options, prompts=None)
        assert result == "Option A"
        assert mock_ask.call_args[0][0] == "Select my_var"

    # Test Case 3: Complex prompts (dictionary style with __prompt__)
    complex_prompts = {
        "my_var": {
            "__prompt__": "Pick one",
            "1": "First Item",
            "2": "Second Item"
        }
    }
    # Note: In the source, choice_lines is regenerated if __prompt__ exists in a dict
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_async_input = "1"
        mock_ask.return_value = mock_async_input
        result = read_user_choice("my_var", options, prompts=complex_prompts)
        assert result == "Option A"
        # Verify the prompt uses the __prompt__ key
        assert "Pick one" in mock_ask.call_args[0][0]

    # Test Case 4: Empty options should raise ValueError
    with pytest.raises(ValueError):
        read_user_choice("my_var", [], prompts=None)

    # Test Case 5: String-based prompt mapping
    simple_prompts = {"my_var": "Custom Question"}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "3"
        result = read_user_choice("my_var", options, prompts=simple_prompts)
        assert result == "Option C"
        assert "Custom Question" in mock_ask.call_args[0][0]
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

@pytest.mark.parametrize("action_to_delete, action_to_reuse, expected_return, should_exit", [
    # Case 1: no_input=True -> deletes and returns True
    (True, None, True, False),
    # Case 2: user says yes to delete -> deletes and returns True
    ("yes", None, True, False),
    # Case 3: user says no to delete, but yes to reuse -> does not delete and returns False
    ("no", "yes", False, False),
    # Case 4: user says no to delete, no to reuse -> exits
    ("no", "no", None, True),
])
def test_prompt_and_delete(action_to_delete, action_to_reuse, expected_return, should_exit):
    path = "/fake/path/to/file"
    
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.remove") as mock_remove, \
         patch("cookiecutter.utils.rmtree") as mock_rmtree, \
         patch("sys.exit") as mock_exit, \
         patch("read_user_yes_no") as mock_yes_no:
        
        # Setup mocks
        mock_isdir.return_value = True
        
        # Logic for simulating prompt flow
        if action_to_delete is True: # no_input=True case
            pass 
        elif action_to_delete == "yes":
            mock_yes_no.return_value = True
        elif action_to_delete == "no":
            # First prompt (is it okay to delete?) returns No
            # Second prompt (do you want to reuse?) returns the value of action_to_reuse
            mock_yes_no.side_effect = [False, action_to_reuse]
        
        # Execute function
        no_input_flag = True if action_to_delete is True else False
        result = prompt_and_delete(path, no_input=no_input_flag)
        
        # Assertions
        if expected_return is not None:
            assert result == expected_return
            
        if should_exit:
            mock_exit.assert_called_once()
        else:
            assert mock_exit.call_count == 0

        # Check if deletion logic was triggered correctly
        if action_to_delete is True or action_to_delete == "yes":
            # Since we mocked isdir to True, rmtree should be called
            mock_rmtree.assert_called()
        elif action_to_delete == "no" and action_to_reuse == "yes":
            # Should not have deleted anything
            mock_rmtree.assert_not_called()
            mock_remove.assert_not_called()

def test_prompt_and_delete_file_removal():
    """Test the branch where path is a file, not a directory."""
    path = "/fake/file.txt"
    
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.remove") as mock_remove, \
         patch("read_user_yes_no") as mock_yes_no:
        
        mock_isdir.return_value = False
        mock_yes_no.return_value = True
        
        result = prompt_and_delete(path, no_input=False)
        
        assert result is True
        mock_remove.assert_called_once_with(path)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_YesNoPrompt():
    """Tests the process_response method of YesNoPrompt for various inputs."""
    prompt = YesNoPrompt("Question?")
    
    # Test truthy values
    for true_val in ["1", "true", "T", "yes", "Y", "on", "  YES  "]:
        assert prompt.process_response(true_val) is True

    # Test falsy values
    for false_val in ["0", "false", "F", "no", "N", "off", "  no  "]:
        assert prompt.process_response(false_val) is False

    # Test invalid values raise InvalidResponse
    from rich.prompt import InvalidResponse
    with pytest.raises(InvalidResponse):
        prompt.process_response("maybe")
    
    with pytest.raises(InvalidResponse):
        prompt.process_response("")
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from rich.prompt import InvalidResponse

def test_YesNoPrompt_process_response():
    prompt = YesNoPrompt("Question?")
    
    # Test truthy values
    for val in ["1", "true", "t", "yes", "y", "on", "  YES  ", "True"]:
        assert prompt.process_response(val) is True
        
    # Test falsy values
    for val in ["0", "false", "f", "no", "n", "off", "  no  ", "False"]:
        assert prompt.process_response(val) is False
        
    # Test invalid values
    with pytest.raises(InvalidResponse):
        prompt.process_response("maybe")
    
    with pytest.raises(InvalidResponse):
        prompt.process_response("")

    with pytest.raises(InvalidResponse):
        prompt.process_response("random_string")
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from collections import OrderedDict
from jinja2 import Environment

@pytest.fixture
def mock_env():
    env = Environment()
    return env

@pytest.fixture
def base_context():
    return {
        'cookiecutter': {
            'project_name': 'my_project',
            'version': '0.1.0',
            '_internal_var': 'secret',
            '__prompt_info__': 'some info'
        }
    }

def test_prompt_for_config(mock_env, base_conext):
    """
    Tests the prompt_for_config function by mocking user input 
    and verifying that variables are processed and rendered correctly.
    """
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'use_git': True,
            'options': ['opt1', 'opt2'],
            'metadata': {'key': 'value'},
            '__prompts__': {'project_name': 'Enter Project Name'}
        }
    }

    # Mocking Prompt.ask for various types of inputs
    # 1. read_user_variable (for project_name) -> returns 'new_name'
    # 2. read_user_yes_no (for use_git) -> returns True
    # 3. prompt_choice_for_config (for options) -> returns 'opt1'
    # 4. JsonPrompt.ask (for metadata dict) -> returns '{"key": "new_value"}'
    
    with patch('rich.prompt.Prompt.ask') as mock_ask, \
         patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        
        # Side effect implementation to simulate sequential user inputs
        mock_ask.side_effect = [
            'new_name',      # project_name (read_user_variable)
            True,            # use_git (read_user_yes_no/Confirm)
            '1',             # options (read_user_choice - returns the index key)
            '{"key": "new_value"}' # metadata (JsonPrompt.ask)
        ]

        result = prompt_for_config(context, no_input=False)

        # Assertions
        assert isinstance(result, OrderedDict)
        assert result['project_name'] == 'new_name'
        assert result['use_git'] is True
        assert result['options'] == 'opt1'
        assert result['metadata'] == {'key': 'new_value'}
        assert result['_internal_var'] == 'secret' # Check that private vars are preserved

def test_prompt_for_config_no_input(mock_env):
    """Tests prompt_for_config with no_input=True (automation mode)."""
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'version': '1.0.0'
        }
    }

    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        result = prompt_for_config(context, no_input=True)
        
        assert result['project_name'] == 'my_project'
        assert result['version'] == '1.0.0'

def test_prompt_for_config_rendering_error(mock_env):
    """Tests that UndefinedError in template rendering raises UndefinedVariableInTemplate."""
    context = {
        'cookiecutter': {
            'project_name': '{{ non_existent_var }}'
        }
    }

    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        from cookiecutter.exceptions import UndefinedVariableInTemplate
        with pytest.raises(UndefinedVariableInTemplate):
            prompt_for_config(context, no_input=False)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_JsonPrompt():
    """Tests the behavior of JsonPrompt's process_response method."""
    
    # Test valid JSON dict input
    valid_json = '{"key": "value", "number": 123}'
    expected_output = {"key": "value", "number": 123}
    assert JsonPrompt.process_response(valid_json) == expected_output

    # Test valid JSON list input (should raise InvalidResponse because it's not a dict)
    invalid_type_json = '["item1", "item2"]'
    with pytest.raises(InvalidResponse, match="Requires JSON dict."):
        Jsonlama = JsonPrompt.process_response(invalid_type_json)

    # Test invalid JSON syntax (should raise InvalidResponse due to decoding error)
    malformed_json = '{"key": "value"'  # Missing closing brace
    with pytest.raises(InvalidResponse, match="Unable to decode to JSON."):
        JsonPrompt.process_response(malformed_json)

    # Test empty input (should raise InvalidResponse due to decoding error)
    empty_input = ""
    with pytest.raises(InvalidResponse, match="Unable to decode to JSON."):
        JsonPrompt.process_response(empty_input)

    # Verify class attributes
    assert JsonPrompt.default is None
    assert JsonPrompt.response_type is dict
    assert "[prompt.invalid]" in JsonPrompt.validate_error_message
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import patch
from collections import OrderedDict

def test_read_user_choice():
    options = ["Option A", "Option B", "Option C"]
    
    # Test Case 1: Standard selection with no prompts provided
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"
        result = read_user_choice("my_var", options)
        assert result == "Option A"
        # Verify the prompt contains the expected structure
        args, kwargs = mock_ask.call_args
        assert "Select my_var" in args[0]
        assert kwargs["choices"] == ["1", "2", "3"]

    # Test Case 2: Selection using custom prompts (string)
    custom_prompts = {"my_var": "Please pick one:"}
    with patch("rich.prompt.Prompt.ask") as mock_lag:
        mock_lag.return_value = "2"
        result = read_user_choice("my_var", options, prompts=custom_prompts)
        assert result == "Option B"
        assert "Please pick one:" in mock_lag.call_args[0][0]

    # Test Case 3: Selection using complex prompts (dict with __prompt__)
    complex_prompts = {
        "my_var": {
            "__prompt__": "Custom Question",
            "1": "Friendly A",
            "2": "Friendly B",
            "3": "Friendly C"
        }
    }
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "3"
        result = read_user_choice("my_var", options, prompts=complex_prompts)
        assert result == "Option C"
        # Verify the prompt uses the __prompt__ key and custom labels
        prompt_text = mock_ask.call_args[0][0]
        assert "Custom Question" in prompt_text
        assert "[bold magenta]1[/] - [bold]Friendly A[/]" in prompt_text

    # Test Case 4: Selection with prefix
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "2"
        result = read_user_choice("my_var", options, prefix="[RED] ")
        assert result == "Option B"
        assert "[RED] Select my_var" in mock_ask.call_args[0][0]

    # Test Case 5: Error handling for empty options
    with pytest.raises(ValueError):
        read_user_choice("my_var", [])
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch
from collections import OrderedDict

def test_read_user_choice():
    options = ["Option A", "Option B", "Option C"]
    
    # Test case 1: Basic selection with no custom prompts
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"
        result = read_user_choice("my_var", options)
        assert result == "Option A"
        
        # Verify the prompt construction (checking if it contains the key part)
        args, kwargs = mock_ask.call_args
        assert "Select my_var" in args[0]
        assert kwargs["choices"] == ["1", "2", "3"]

    # Test case 2: Selection with custom string prompts
    prompts = {"my_var": "Please pick a fruit"}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "2"
        result = read_user_choice("my_var", options, prompts=prompts)
        assert result == "Option B"
        assert "Please pick a fruit" in mock_ask.call_args[0][0]

    # Test case 3: Selection with complex dictionary prompts (mapping keys to labels)
    # This tests the logic for: choice_lines = ... if p in prompts[var_name]
    prompts = {
        "my_var": {
            "__prompt__": "Choose your destiny",
            "1": "First Choice",
            "2": "Second Choice",
            "3": "Third Choice"
        }
    }
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "3"
        result = read_user_choice("my_var", options, prompts=prompts)
        assert result == "Option C"
        
        # Check if the prompt contains the custom lines
        full_prompt = mock_ask.call_args[0][0]
        assert "Choose your destiny" in full_prompt
        assert "[bold magenta]1[/] - [bold]First Choice[/]" in full_prompt

    # Test case 4: Error handling for empty options
    with pytest.raises(ValueError):
        read_user_choice("empty", [])

    # Test case 5: Prefix usage
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = "1"
        result = read_user_choice("var", options, prefix="[bold]PROMPT: [/]")
        assert result == "Option A"
        assert "[bold]PROMPT: [/]Select var" in mock_ask.call_args[0][0]
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from collections import OrderedDict

@pytest.mark.parametrize("context, repo_dir, no_input, expected_path", [
    # Case 1: New style (templates dict) with no input
    (
        {
            "cookiecutter": {
                "templates": {
                    "tpl1": {"path": "template_a"},
                    "tpl2": {"path": "template_b"}
                }
            }
        },
        "/tmp/repo",
        True,
        str(Path("/tmp/repo/template_a").resolve())
    ),
    # Case 2: Old style (list of strings) with no input
    (
        {
            "cookiecutter": {
                "template": ["template_x (path/to/x)", "template_y (path/to/y)"]
            }
        },
        "/tmp/repo",
        True,
        str(Path("/tmp/repo/path/to/x").resolve())
    ),
])
def test_choose_nested_template(context, repo_dir, no_input, expected_path):
    # Patching dependencies to avoid real file system / jinja2 environment creation
    with patch("cookiecutter.utils.create_env_with_context"), \
         patch("prompting.read_user_choice") as mock_choice:
        
        # For the 'template' list case, read_user_choice needs to return the first item
        if "template" in context["cookiecutter"] and isinstance(context["cookietemplate"], list):
             mock_choice.return_value = context["cookiecutter"]["template"][0]

        result = choose_nested_template(context, repo_dir, no_input)
        assert result == expected_path

def test_choose_nested_template_error_absolute_path():
    # Case 3: Error when path is absolute
    context = {
        "cookiecutter": {
            "templates": {"tpl1": {"path": "/absolute/path"}}
        }
    }
    with patch("cookiecutter.utils.create_env_with_context"):
        with pytest.raises(ValueError, match="Illegal template path"):
            choose_nested_template(context, "/tmp/repo", no_input=True)

def test_choose_nested_template_error_invalid_regex():
    # Case 4: Error when old style string doesn't contain parenthesis for regex
    context = {
        "cookiecutter": {
            "template": ["invalid_string"]
        }
    }
    with patch("cookiecutter.utils.create_env_with_context"):
        with pytest.raises(AttributeError):
            choose_nested_template(context, "/tmp/repo", no_input=True)

def test_choose_nested_template_prompting():
    # Case 5: Interactive mode (no_input=False)
    context = {
        "cookiecutter": {
            "templates": {
                "tpl1": {"path": "tpl1_path"}
            }
        }
    }
    with patch("cookiecutter.utils.create_env_with_context"), \
         patch("prompting.read_user_choice") as mock_choice:
        
        # Simulate user selecting the first key 'tpl1'
        mock_choice.return_value = "tpl1"
        
        result = choose_nested_template(context, "/tmp/repo", no_input=False)
        assert result == str(Path("/tmp/repo/tpl1_path").resolve())
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_variable():
    # Test 1: Basic functionality - returns value from Prompt.ask
    with patch("rich.prompt.Prompt.ask", return_value="my_project"):
        result = read_user_variable("project_name", "default_val")
        assert result == "my_project"

    # Test 2: Returns default value when prompt returns None
    # Note: The function has a while True loop that breaks if variable is not None.
    # We simulate one None then one valid string to avoid infinite loop.
    with patch("rich.prompt.Prompt.ask", side_effect=[None, "valid_input"]):
        result = read_user_variable("project_name", "default_val")
        assert result == "valid_input"

    # Test 3: Uses custom prompt from prompts dictionary
    prompts = {"project_name": "Enter your project name:"}
    with patch("rich.prompt.Prompt.ask", return_value="decorated") as mock_ask:
        result = read_user_variable("project_name", "default", prompts=prompts)
        assert result == "decorated"
        # Check if the question passed to Prompt.ask used the custom prompt
        mock_ask.assert_called_with("Enter your project name:", default="default")

    # Test 4: Uses variable name as question when no prompts provided
    with patch("rich.prompt.Prompt.ask", return_value="simple") as mock_ask:
        result = read_user_variable("simple_var", "default")
        assert result == "simple"
        mock_ask.assert_called_with("simple_var", default="default")

    # Test 5: Respects the prefix argument
    with patch("rich.prompt.Prompt.ask", return_value="prefixed") as mock_ask:
        result = read_user_variable("var", "def", prefix="[bold]Prefix: ")
        assert result == "prefixed"
        mock_ask.assert_called_with("[bold]Prefix: var", default="def")

    # Test 6: Handles empty prompts dictionary gracefully
    with patch("rich.prompt.Prompt.ask", return_value="empty_dict_test") as mock_ask:
        result = read_user_variable("var", "def", prompts={})
        assert result == "empty_dict_test"
        mock_ask.assert_called_with("var", default="def")
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_variable():
    # Test Case 1: Standard behavior without prompts or prefix
    with patch("rich.prompt.Prompt.ask", return_value="my_value") as mock_ask:
        result = read_user_variable("var_name", "default_val")
        assert result == "my_value"
        mock_ask.assert_called_once_with("var_name", default="default_val")

    # Test Case 2: Using a custom prompt from the prompts dictionary
    prompts = {"var_name": "Custom Question"}
    with patch("rich.prompt.Prompt.ask", return_value="user_input") as mock_ask:
        result = read_user_variable("var_name", "default_val", prompts=prompts)
        assert result == "user_input"
        mock_ask.assert_called_once_with("Custom Question", default="default_val")

    # Test Case 3: Using a prefix and custom prompt
    prompts = {"var_name": "Custom Question"}
    prefix = "Prefix: "
    with patch("rich.prompt.Prompt.ask", return_value="user_input") as mock_ask:
        result = read_user_variable("var_name", "default_val", prompts=prompts, prefix=prefix)
        assert result == "user_input"
        mock_ask.assert_called_once_with("Prefix: Custom Question", default="default_val")

    # Test Case 4: Handling None return from Prompt.ask (simulating retry loop)
    # The function has a 'while True' loop that breaks when variable is not None.
    with patch("rich.prompt.Prompt.ask", side_effect=[None, "recovered_value"]) as mock_ask:
        result = read_user_variable("var_name", "default_val")
        assert result == "recovered_value"
        assert mock_ask.call_count == 2

    # Test Case 5: Variable name not in prompts (fallback to var_name)
    prompts = {"other_var": "Other Question"}
    with patch("rich.prompt.Prompt.ask", return_value="val") as mock_ask:
        result = read_user_variable("var_name", "default_val", prompts=prompts)
        assert result == "val"
        mock_ask.assert_called_once_with("var_name", default="default_val")

    # Test Case 6: Empty prompts dictionary or None
    with patch("rich.prompt.Prompt.ask", return_value="val") as mock_ask:
        result = read_user_variable("var_name", "default_val", prompts=None)
        assert result == "val"
        mock_ask.assert_called_once_with("var_name", default="default_val")
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_variable():
    # Test Case 1: Basic usage with default value and no prompts
    with patch("rich.prompt.Prompt.ask", return_value="my_val") as mock_ask:
        result = read_user_variable("var_name", "default_val")
        assert result == "my_val"
        mock_ask.assert_called_once_with("var_name", default="default_val")

    # Test Case 2: Usage with custom prompts mapping
    prompts = {"var_name": "Custom Question"}
    with patch("rich.prompt.Prompt.ask", return, value="user_input") as mock_ask:
        # Note: We use a side_effect to simulate the loop breaking on first valid input
        mock_ask.side_effect = ["user_input"]
        result = read_user_variable("var_name", "default_val", prompts=prompts)
        assert result == "user_input"
        mock_ask.assert_called_once_with("Custom Question", default="default_val")

    # Test Case 3: Usage with prefix
    with patch("rich.prompt.Prompt.ask", return_value="prefixed_val") as mock_ask:
        result = read_user_variable("var_name", "default_val", prefix="PRE: ")
        assert result == "prefixed_val"
        mock_ask.assert_called_once_with("PRE: var_name", default="default_val")

    # Test Case 4: Simulating a None response that requires loop continuation
    # The function loops until variable is not None.
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.side_effect = [None, "valid_input"]
        result = read_user_variable("var_name", "default_val")
        assert result == "valid_input"
        assert mock_ask.call_count == 2

    # Test Case 5: Using prompts with a dictionary that contains the key but is empty/None for that key
    prompts = {"var_name": None}
    with patch("rich.prompt.Prompt.ask", return_value="fallback") as mock_ask:
        result = read_user_variable("var_name", "default_val", prompts=prompts)
        assert result == "fallback"
        # Should fall back to var_name because prompts[var_name] is None/Falsy
        mock_ask.assert_called_once_with("var_name", default="default_val")

    # Test Case 6: Using complex prompt dictionary (checking the 'if prompts and var_name in prompts' logic)
    prompts = {"var_name": "Pretty Prompt"}
    with patch("rich.prompt.Prompt.ask", return_value="ok") as mock_ask:
        result = read_user_variable("var_name", "default", prompts=prompts)
        assert result == "ok"
        mock_ask.assert_called_once_with("Pretty Prompt", default="default")
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch
from rich.prompt import InvalidResponse

def test_JsonPrompt():
    """Tests the process_response static method of JsonPrompt."""
    # Test valid JSON string representing a dictionary
    valid_json = '{"key": "value", "number": 123, "bool": true}'
    expected_output = {"key": "value", "number": 123, "bool": True}
    assert JsonPrompt.process_response(valid_json) == expected_output

    # Test valid JSON string representing a different dictionary structure
    nested_json = '{"outer": {"inner": "data"}}'
    expected_nested = {"outer": {"inner": "data"}}
    assert JsonPrompt.process_response(nested_json) == expected_nested

    # Test invalid JSON syntax (should raise InvalidResponse)
    invalid_syntax = '{"key": "value",}'  # Trailing comma is invalid in standard JSON
    with pytest.raises(InvalidResponse) as excinfo:
        JsonPrompt.process_response(invalid_syntax)
    assert "Unable to decode to JSON" in str(excinfo.value)

    # Test valid JSON that is not a dictionary (should raise InvalidResponse)
    not_a_dict_json = '"just a string"'
    with pytest.raises(InvalidResponse) as excinfo:
        JsonPrompt.process_response(not_a_dict_json)
    assert "Requires JSON dict" in str(excinfo.value)

    # Test integer input (valid JSON, but not a dict)
    int_json = '123'
    with pytest.raises(InvalidResponse) as excinfo:
        JsonPrompt.process_response(int_json)
    assert "Requires JSON dict" in str(excinfo.value)

    # Test class attributes
    assert JsonPrompt.response_type == dict
    assert "[prompt.invalid]" in JsonPrompt.validate_error_message
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_JsonPrompt():
    """Tests the JsonPrompt class properties and response processing."""
    # Test static attributes
    assert JsonPrompt.default is None
    assert JsonPrompt.response_type is dict
    assert JsonPrompt.validate_error_message == "[prompt.invalid]  Please enter a valid JSON string"

    # Test process_response with valid JSON
    valid_json = '{"key": "value", "number": 123}'
    expected_dict = {"key": "value", "number": 123}
    assert JsonPrompt.process_response(valid_json) == expected_dict

    # Test process_response with valid JSON list (should fail because it's not a dict per process_json logic)
    invalid_json_type = '["item1", "item2"]'
    with pytest.raises(InvalidResponse, match="Requires JSON dict."):
        JsonPrompt.processrass_response(invalid_json_type)

    # Test process_response with invalid JSON syntax
    invalid_syntax = '{"key": "value"'  # Missing closing brace
    with pytest.raises(InvalidResponse, match="Unable to decode to JSON."):
        JsonPrompt.process_response(invalid_syntax)

    # Test process_response with non-string input (e.g., None or int) 
    # json.loads fails on non-strings/bytes
    with pytest.raises(InvalidResponse):
        JsonPrompt.process_response(None)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch
from collections import OrderedDict

def test_read_user_dict(monkeypatch):
    """Tests the read_user_dict function for various input scenarios."""
    
    # Test Case 1: Standard usage with a default dictionary and no prompts
    default_val = {"key": "value"}
    # Mocking JsonPrompt.ask to return a specific JSON string that represents a dict
    with patch("rich.prompt.PromptBase.ask", return_value='{"new_key": "new_value"}') as mock_ask:
        result = read_user_dict("test_var", default_val)
        assert result == {"new_key": "new_value"}
        mock_ask.assert_called_once()

    # Test Case 2: Usage with prompts and prefix
    prompts = {"test_var": "Enter your config:"}
    prefix = "[bold]Prompt:[/]"
    with patch("rich.prompt.PromptBase.ask", return_value='{"a": 1}') as mock_ask:
        result = read_user_dict("test_var", default_val, prompts=prompts, prefix=prefix)
        assert result == {"a": 1}
        # Verify the prompt string construction
        expected_prompt_part = "[bold]Prompt:[/][bold]Enter your config:[/] [cyan bold](default)[/]"
        # We check if the first argument to ask contains our constructed question
        args, kwargs = mock_ask.call_args
        assert "Enter your config:" in args[0]
        assert prefix in args[0]

    # Test Case 3: Invalid default_value type (should raise TypeError)
    with pytest.raises(TypeError):
        read_user_dict("test_var", "not a dict")

    # Test Case 4: Testing the error handling inside JsonPrompt via process_json
    # Since read_user_dict calls JsonPrompt.ask, we test if InvalidResponse is raised
    # when the input is not valid JSON.
    from rich.prompt import InvalidResponse
    with patch("rich.prompt.PromptBase.ask", return_value='{invalid_json}') as mock_ask:
        with pytest.raises(InvalidResponse) as excinfo:
            read_user_dict("test_var", default_val)
        assert "Unable to decode to JSON" in str(excinfo.value)

    # Test Case 5: Testing the error handling when JSON is valid but not a dictionary
    with patch("rich.prompt.PromptBase.ask", return_value='"just a string"') as mock_ask:
        with pytest.raises(InvalidResponse) as excinfo:
            read_user_dict("test_var", default_val)
        assert "Requires JSON dict" in str(excinfo.value)

    # Test Case 6: Verifying behavior when var_name is not in prompts
    with patch("rich.prompt.PromptBase.ask", return_value='{"ok": true}') as mock_ask:
        result = read_user_dict("unknown_var", default_val, prompts={"other": "val"})
        assert result == {"ok": True}
        # Should use the var_name itself as the question
        args, _ = mock_ask.call_args
        assert "unknown_var" in args[0]
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import patch
from collections import OrderedDict

def test_read_user_dict():
    """Test the read_user_dict function for various input scenarios."""
    
    # Scenario 1: Basic usage with default value as a dict and no prompts/prefix
    default_val = {"key": "value"}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_args = [
            'my_var [cyan bold](default)[/]', 
            {'default': default_val}, 
            False
        ]
        # Simulate user entering a JSON string representing a dict
        mock_ask.return_value = '{"new_key": "new_value"}'
        
        result = read_user_dict("my_var", default_val)
        
        assert result == {"new_key": "new_value"}
        mock_ask.assert_called_once()

    # Scenario 2: Using custom prompts and prefix
    prompts = {"my_var": "Custom Question"}
    prefix = "PROMPT: "
    default_val = {"a": 1}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        # The function constructs the question string with the prefix and suffix
        expected_question = "PROMPT: Custom Question [cyan bold](default)[/]"
        mock_ask.return_value = '{"b": 2}'
        
        result = read_user_dict("my_var", default_val, prompts=prompts, prefix=prefix)
        
        assert result == {"b": 2}
        # Check if the prompt string was constructed correctly
        args, kwargs = mock_ask.call_args
        assert args[0] == expected_question

    # Scenario 3: Testing TypeError when default_value is not a dict
    with pytest.raises(TypeError):
        read_user_dict("my_var", "not a dict")

    # Scenario 4: Using prompts that don't contain the var_name (fallback to var_name)
    prompts = {"other_var": "Other"}
    default_val = {"key": "value"}
    with patch("rich.prompt.Prompt.ask") as mock_as:
        mock_as.return_value = '{"x": 1}'
        # Should use 'my_var' because it's not in prompts
        read_user_dict("my_var", default_val, prompts=prompts)
        args, _ = mock_as.call_args
        assert "my_var" in args[0]

    # Scenario 5: Integration with JsonPrompt logic via Prompt.ask
    # This tests if the string returned by Prompt.ask is correctly parsed as JSON
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '{"nested": {"a": 1}}'
        result = read_user_dict("test_var", {"default": "val"})
        assert result == {"nested": {"a": 1}}
        assert isinstance(result, dict)

```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_dict():
    """Tests the read_user_dict function for various scenarios."""
    
    # Scenario 1: Standard prompt with no special prompts or prefix
    default_val = {"key": "value"}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '{"new_key": "new_value"}'
        result = read_user_dict("my_var", default_val)
        
        assert result == {"new_key": "new_value"}
        # Check if the question includes the default display hint
        expected_question = "my_var [cyan bold](default)[/]"
        mock_ask.assert_called_once_with(expected_question, default=default_val, show_default=False)

    # Scenario 2: Prompt with a custom prompt string in the prompts dictionary
    prompts = {"my_var": "Enter your configuration"}
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '{"a": 1}'
        result = read_user_dict("my_var", default_val, prompts=prompts)
        
        assert result == {"a": 1}
        expected_question = "Enter your configuration [cyan bold](default)[/]"
        mock_ask.assert_called_with(expected_question, default=default_val, show_default=False)

    # Scenario 3: Prompt with a prefix
    prefix = "Config: "
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '{}'
        result = read_user_dict("my_var", default_val, prefix=prefix)
        
        assert result == {}
        expected_question = "Config: my_var [cyan bold](default)[/]"
        mock_ask.assert_called_with(expected_question, default=default_val, show_default=False)

    # Scenario 4: Invalid default_value type (should raise TypeError)
    with pytest.raises(TypeError):
        read_user_dict("my_var", "not_a_dict")

    # Scenario 5: JSON parsing error during input (JsonPrompt behavior)
    # Note: JsonPrompt.process_response is called by Prompt.ask internally
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        # We simulate the logic of JsonPrompt which calls process_json
        # If the user enters invalid JSON, process_json raises InvalidResponse
        from rich.prompt import InvalidResponse
        
        # Mocking the underlying behavior of JsonPrompt via the prompt's return value 
        # is tricky because Prompt.ask handles the loop, but we can test 
        # the core logic used by JsonPrompt: process_json
        with pytest.raises(InvalidResponse) as excinfo:
            process_json("{invalid_json}")
        assert "Unable to decode to JSON" in str(excinfo.value)

    # Scenario 6: Valid JSON but not a dictionary (should raise InvalidResponse)
    with pytest.raises(InvalidResponse) as excinfo:
        process_json('["not", "a", "dict"]')
    assert "Requires JSON dict" in str(excinfo.value)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from rich.prompt import InvalidResponse

def test_YesNoPrompt_process_response():
    prompt = YesNoPrompt("Question?")
    
    # Test truthy values
    for val in ["1", "true", "t", "yes", "y", "on", "  YES  ", "TRUE"]:
        assert prompt.process_response(val) is True
        
    # Test falsy values
    for val in ["0", "false", "f", "no", "n", "off", "  no  ", "FALSE"]:
        assert prompt.process_response(val) is False
        
    # Test invalid response
    with pytest.raises(InvalidResponse):
        prompt.process_response("maybe")
    
    with pytest.raises(InvalidResponse):
        prompt.process_response("")

    with pytest.raises(InvalidResponse):
        prompt.process_response("random_string")
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_dict():
    """Tests the read_user_dict function with various inputs and mock responses."""
    
    # Test Case 1: Standard usage with a simple default dictionary
    # Mocking JsonPrompt.ask to return a valid JSON string converted to dict
    with patch("rich.prompt.PromptBase.ask") as mock_ask:
        mock_ask.return_value = {"key": "value"}
        default_val = {"key": "old_value"}
        result = read_user_dict("my_var", default_val)
        
        assert result == {"key": "value"}
        # Verify it calls ask with the correct formatted question
        mock_ask.assert_called()
        args, kwargs = mock_args_from_call(mock_ask)
        assert "my_var" in args[0]

    # Test Case 2: Usage with prompts and prefix
    prompts = {"my_var": "Custom Question"}
    prefix = "Prefix: "
    with patch("rich.prompt.PromptBase.ask") as mock_ask:
        mock_ask.return_value = {"a": 1}
        result = read_user_dict("my_var", {"a": 0}, prompts=prompts, prefix=prefix)
        
        assert result == {"a": 1}
        # Check if prefix and custom prompt are combined correctly
        args, _ = mock_args_from_call(mock_ask)
        assert args[0].startswith("Prefix: Custom Question")

    # Test Case 3: Error handling - non-dict default value should raise TypeError
    with pytest.raises(TypeError):
        read_user_dict("my_var", ["not", "a", "dict"])

    # Test Case 4: Verify behavior when prompts are missing/empty (uses var_name)
    with patch("rich.prompt.PromptBase.ask") as mock_ask:
        mock_ask.return    value = {"ok": True}
        result = read_user_dict("simple_var", {"ok": False}, prompts=None)
        
        assert result == {"ok": True}
        args, _ = mock_args_from_call(mock_ask)
        assert "simple_var" in args[0]

def mock_args_from_call(mock_obj):
    """Helper to extract arguments from a mock call."""
    return mock_obj.call_args[0], mock_obj.call_args[1]
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from collections import OrderedDict
from jinja2 import Environment

@pytest.fixture
def mock_env():
    env = Environment()
    return env

@pytest.fixture
def sample_context():
    return {
        'cookiecutter': {
            'project_name': 'my_project',
            'version': '0.1.0',
            '_internal_var': 'secret',
            '__metadata__': 'meta_{{ cookiecutter.project_name }}',
            'options_list': ['opt1', 'opt2'],
            'nested_dict': {'key': 'val'},
            '__prompts__': {
                'project_name': 'Enter project name'
            }
        }
    }

def test_prompt_for_config(sample_context, mock_env):
    """
    Tests prompt_for_config by mocking the interactive input functions.
    We simulate a 'no_input=True' scenario to avoid actual terminal interaction
    and verify that variables are rendered and processed correctly.
    """
    # We use no_input=True to bypass Prompt.ask calls which require stdin
    # This allows us to test the logic of variable rendering, 
    # ordering, and template processing.
    
    # Setup context with a dependency on another variable
    context = {
        'cookiecutter': {
            'project_name': 'TestProject',
            'repo_name': '{{ cookiecutter.project_name.lower().replace(" ", "_") }}',
            'is_active': True,
            'choices': ['alpha', 'beta'],
            '__prompts__': {
                'project_name': 'Name?'
            }
        }
    }

    # Mock create_env_with_context to return our controlled env
    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        # We run with no_input=True so the function doesn't call Prompt.ask
        result = prompt_for_config(context, no_input=True)

        # Assertions
        assert isinstance(result, OrderedDict)
        assert result['project_name'] == 'TestProject'
        # Check that jinja rendering worked for repo_name
        assert result['repo_name'] == 'testproject'
        # Check boolean handling
        assert result['is_active'] is True
        # Check list/choice handling (in no_input mode, it takes the first)
        assert result['choices'] == 'alpha'

def test_prompt_for_config_error(sample_context, mock_env):
    """Tests that UndefinedError in templates raises UndefinedVariableInTemplate."""
    # Create a context where a variable refers to something that doesn't exist
    bad_context = {
        'cookiecutter': {
            'a': 'value',
            'b': '{{ cookiecutter.non_existent }}'
        }
    }

    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        from cookiecutter.exceptions import UndefinedVariableInTemplate
        with pytest.raises(UndefinedVariableInTemplate):
            prompt_for_config(bad_context, no_input=True)

def test_prompt_for_config_dict_processing(sample_context, mock_env):
    """Tests that dictionary variables are processed."""
    context = {
        'cookiecutter': {
            'project_name': 'Base',
            'settings': {'env': 'prod'}
        }
    }

    with patch('cookiecutter.utils.create_env_with_context', return_value=mock_env):
        # We must mock JsonPrompt.ask because with no_input=False, 
        # it tries to interact with the user.
        with patch('cookiecutter.prompting.JsonPrompt.ask') as mock_json_ask:
            mock_json_ask.return_value = {'env': 'prod'}
            
            result = prompt_for_config(context, no_input=mask_interaction())
            # Note: In actual test execution, we'd use a helper to mask 
            # all Prompt.ask calls for the 'no_input=False' path.
            pass

def mask_interaction():
    """Helper to return a dummy function that simulates no input."""
    return True
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
import json
from collections import OrderedDict
from rich.prompt import InvalidResponse

def test_process_json():
    # Test valid JSON dictionary
    valid_json = '{"key": "value", "number": 123, "bool": true}'
    expected_output = OrderedDict([
        ("key", "value"),
        ("number", 123),
        ("bool", True)
    ])
    assert process_json(valid_json) == expected_output

    # Test valid JSON list (should raise InvalidResponse because it's not a dict)
    invalid_type_json = '["item1", "item2"]'
    with pytest.raises(InvalidResponse, match="Requires JSON dict."):
        process_json(invalid_type_json)

    # Test malformed JSON (syntax error)
    malformed_json = '{"key": "value",}'  # Trailing comma is invalid in standard json.loads
    with pytest.raises(InvalidResponse, match="Unable to decode to JSON."):
        process_json(malformed_json)

    # Test empty string
    empty_string = ""
    with pytest.raises(InvalidResponse, match="Unable to decode to JSON."):
        process_json(empty_string)

    # Test valid JSON dictionary with nested structures
    nested_json = '{"outer": {"inner": "val"}, "list": [1, 2]}'
    expected_nested = OrderedDict([
        ("outer", OrderedDict([("inner", "val")])),
        ("list", [1, 2])
    ])
    assert process_json(nested_json) == expected_nested

    # Test JSON that is just a primitive (not a dict)
    primitive_json = '"just a string"'
    with pytest.raises(InvalidResponse, match="Requires JSON dict."):
        process_json(primitive_json)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_variable():
    # Test case 1: Default behavior (no prompts, no prefix)
    with patch("rich.prompt.Prompt.ask", return_value="my_value") as mock_ask:
        result = read_user_variable("var_name", "default_val")
        assert result == "my_value"
        mock_ask.assert_called_once_with("var_name", default="default_val")

    # Test case 2: Using prompts dictionary to override question name
    prompts = {"var_name": "Custom Question"}
    with patch("rich.prompt.Prompt.ask", return_value="user_input") as mock_ask:
        result = read_user_variable("var_name", "default_val", prompts=prompts)
        assert result == "user_input"
        mock_ask.assert_called_once_with("Custom Question", default="default_val")

    # Test case 3: Using prefix and custom prompt
    prompts = {"var_name": "Custom Question"}
    prefix = "PROMPT: "
    with patch("rich.prompt.Prompt.ask", return_value="user_input") as mock_ask:
        result = read_user_variable("var_name", "default_val", prompts=prompts, prefix=prefix)
        assert result == "user_input"
        mock_ask.assert_called_once_with("PROMPT: Custom Question", default="default_val")

    # Test case 4: Handling None response (looping logic)
    # The function uses a 'while True' loop that breaks only when variable is not None.
    # We simulate the first call returning None and the second returning a value.
    with patch("rich.prompt.Prompt.ask", side_effect=[None, "valid_response"]) as mock_ask:
        result = read_user_variable("var_name", "default_val")
        assert result == "valid_response"
        assert mock_ask.call_count == 2

    # Test case 5: Variable name not in prompts (should use var_name as question)
    prompts = {"other_var": "Different Question"}
    with patch("rich.prompt.Prompt.ask", return_value="val") as mock_ask:
        result = read_user_variable("var_name", "default_val", prompts=prompts)
        assert result == "val"
        mock_ask.assert_called_once_with("var_name", default="default_val")
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_YesNoPrompt():
    """Test the process_response method of YesNoPrompt."""
    prompt = YesNoPrompt("Question?")

    # Test truthy values
    for value in ["1", "true", "t", "yes", "y", "on", "  YES  "]:
        assert prompt.process_response(value) is True

    # Test falsy values
    for value in ["0", "false", "f", "no", "n", "off", "NO"]:
        assert prompt.process_response(value) is False

    # Test invalid values raise InvalidResponse
    from rich.prompt import InvalidResponse
    with pytest.raises(InvalidResponse):
        prompt.process_response("maybe")
    
    with pytest.raises(InvalidResponse):
        prompt.process_response("")
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import patch
from collections import OrderedDict

def test_read_user_variable():
    # Test Case 1: Standard behavior - return default when input is None (simulated)
    # Note: Prompt.ask returns the value; we simulate user pressing enter for default
    with patch("rich.prompt.Prompt.ask", return_value="default_val") as mock_ask:
        result = read_user_variable("my_var", "default_val")
        assert result == "default_val"
        mock_ask.assert_called_once_with("my_var", default="default_val")

    # Test Case 2: Using custom prompts dictionary
    prompts = {"my_var": "What is your name?"}
    with patch("rich.prompt.Prompt.ask", return_value="John Doe") as mock_ask:
        result = read_user_variable("my_var", "default_val", prompts=prompts)
        assert result == "John Doe"
        mock_ask.assert_called_once_with("What is your name?", default="default_val")

    # Test Case 3: Using prefix
    with patch("rich.prompt.Prompt.ask", return_value="Value") as mock_ask:
        result = read_user_variable("my_var", "default_val", prefix="Enter: ")
        assert result == "Value"
        mock_ask.assert_called_once_with("Enter: my_var", default="default_val")

    # Test Case 4: Using prefix and prompts together
    prompts = {"my_var": "Name?"}
    with patch("rich.prompt.Prompt.ask", return_value="John") as mock_ask:
        result = read_user_variable("my_var", "default_val", prompts=prompts, prefix="SET ")
        assert result == "John"
        mock_ask.assert_called_once_with("SET Name?", default="default_val")

    # Test Case 5: Simulating a loop where first input is None (if logic allowed) 
    # and second is valid. Since Prompt.ask(default=...) usually returns the default 
    # rather than None when empty, we test the return of a specific value.
    with patch("rich.prompt.Prompt.ask", side_effect=["first_input"]) as mock_ask:
        result = read_user_variable("var", "def")
        assert result == "first_input"
        assert mock_ask.call_count == 1

    # Test Case 6: Variable not in prompts (fallback to var_name)
    with patch("rich.prompt.Prompt.ask", return_value="Value") as mock_ask:
        result = read_user_variable("unknown_var", "def", prompts={"other": "desc"})
        assert result == "Value"
        mock_ask.assert_called_once_with("unknown_var", default="def")
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from rich.prompt import InvalidResponse

def test_YesNoPrompt_process_response():
    prompt = YesNoPrompt("Question?")
    
    # Test truthy values
    for value in ["1", "true", "t", "yes", "y", "on", "  YES  ", "True"]:
        assert prompt.process_response(value) is True
        
    # Test falsy values
    for value in ["0", "false", "f", "no", "n", "off", " NO ", "False"]:
        assert prompt.process_response(value) is False
        
    # Test invalid values
    with pytest.raises(InvalidResponse):
        prompt.process_response("maybe")
        
    with pytest.raises(InvalidResponse):
        prompt.process_response("")

    with pytest.raises(InvalidResponse):
        prompt.process_response("apple")
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from collections import OrderedDict
from rich.prompt import InvalidResponse

def test_process_json():
    # Test valid JSON string representing a dictionary
    valid_json = '{"key": "value", "number": 123, "bool": true}'
    expected_output = OrderedDict([
        ("key", "value"),
        ("number", 123),
        ("bool", True)
    ])
    assert process_json(valid_json) == expected_output

    # Test valid JSON string representing a nested dictionary
    nested_json = '{"outer": {"inner": "val"}}'
    expected_nested = OrderedDict([
        ("outer", OrderedDict([("inner", "val")]))
    ])
    assert process_json(nestedly_json) == expected_nested

    # Test invalid JSON syntax (raises InvalidResponse)
    invalid_syntax = '{"key": "value",}'  # Trailing comma is invalid in standard JSON
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(invalid_syntax)
    assert "Unable to decode to JSON" in str(excinfo.value)

    # Test valid JSON but not a dictionary (e.g., a list)
    list_json = '["item1", "item2"]'
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(list_json)
    assert "Requires JSON dict" in str(excinfo.value)

    # Test valid JSON but not a dictionary (e.g., a string)
    string_json = '"just a string"'
    with pytest.raises(InvalidResponse) as excinfo:
        process_json(string_json)
    assert "Requires JSON dict" in str(excinfo.value)

    # Test empty input/malformed input
    with pytest.raises(InvalidResponse):
        process_json("")
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_read_user_variable():
    """Tests various scenarios for reading user variables."""
    
    # Test Case 1: Standard behavior with no prompts or prefix
    # Simulates user entering "my_project"
    with patch("rich.prompt.Prompt.ask", return_value="my_project") as mock_ask:
        result = read_user_variable("project_name", default_value="default_val")
        assert result == "my_project"
        mock_ask.assert_called_once_with("project_name", default="default_val")

    # Test Case 2: Using a custom prompt from the prompts dictionary
    prompts = {"project_name": "Enter your project name:"}
    with patch("rich.prompt.Prompt.ask", return either_value="my_custom_project") as mock_ask:
        # We use side_effect to simulate returning a value
        mock_ask.side_effect = ["my_custom_project"]
        result = read_user_variable("project_name", default_value="default_val", prompts=prompts)
        assert result == "my_custom_project"
        mock_ask.assert_called_with("Enter your project name:", default="default_val")

    # Test Case 3: Using a prefix and custom prompt
    prompts = {"project_name": "Name:"}
    prefix = "PROMPT: "
    with patch("rich.prompt.Prompt.ask", return_value="prefixed_val") as mock_ask:
        result = read_user_variable("project_name", default_value="default_val", prompts=prompts, prefix=prefix)
        assert result == "prefixed_val"
        mock_ask.assert_called_with("PROMPT: Name:", default="default_val")

    # Test Case 4: Simulating user hitting Enter (returning None/Default)
    # The function uses a while True loop, so we simulate the first call returning None 
    # and the second call returning a valid value to break the loop.
    with patch("rich.prompt.Prompt.ask", side_effect=[None, "recovered_value"]) as mock_ask:
        result = read_user_variable("project_name", default_value="default_val")
        assert result == "recovered_value"
        assert mock_ask.call_count == 2

    # Test Case 5: Verify behavior when prompts dict is provided but key is missing
    with patch("rich.prompt.Prompt.ask", return_value="fallback") as mock_ask:
        prompts = {"other_key": "Something else"}
        result = read_user_variable("project_name", default_value="default_val", prompts=prompts)
        assert result == "fallback"
        # Should fallback to var_name because project_name is not in prompts
        mock_ask.assert_called_with("project_name", default="default_val")
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_JsonPrompt():
    """Test the behavior and properties of the JsonPrompt class."""
    
    # Test static method process_response with valid JSON string (dict)
    valid_json = '{"key": "value", "number": 123}'
    expected_dict = {"key": "value", "number": 123}
    assert JsonPrompt.process_response(valid_json) == expected_dict

    # Test static method process_response with valid JSON string (list) - should raise InvalidResponse
    invalid_json_type = '["item1", "item2"]'
    with pytest.raises(InvalidResponse, match="Requires JSON dict."):
        JsonjaPrompt.process_response(invalid_json_type)

    # Test static method process_response with invalid JSON syntax - should raise InvalidResponse
    malformed_json = '{"key": "value"'  # Missing closing brace
    with pytest.raises(InvalidResponse, match="Unable to decode to JSON."):
        JsonPrompt.process_response(malformed_json)

    # Test class attributes
    assert JsonPrompt.default is None
    assert JsonPrompt.response_type is dict
    assert "[prompt.invalid]" in JsonPrompt.validate_error_message

    # Test the PromptBase integration via .ask (mocking rich interaction)
    with patch("rich.prompt.Prompt.ask") as mock_ask:
        mock_ask.return_value = '{"a": 1}'
        result = JsonPrompt.ask("Enter JSON:")
        assert result == {"a": 1}
        mock_ask.assert_called_once()
```


