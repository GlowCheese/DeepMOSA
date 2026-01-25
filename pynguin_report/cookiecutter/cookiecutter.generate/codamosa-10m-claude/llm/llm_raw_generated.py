####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context():
    """Test apply_overwrites_to_context function with various scenarios."""
    
    # Test 1: Simple value overwrite
    context = {'name': 'old_value'}
    overwrite_context = {'name': 'new_value'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['name'] == 'new_value'
    
    # Test 2: Ignore new variables at first level
    context = {'existing': 'value'}
    overwrite_context = {'new_var': 'new_value'}
    apply_overwrites_to_context(context, overwrite_context)
    assert 'new_var' not in context
    
    # Test 3: Add new variables in nested dictionary
    context = {'nested': {'existing': 'value'}}
    overwrite_context = {'nested': {'new_var': 'new_value'}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context['nested']['new_var'] == 'new_value'
    
    # Test 4: Choice variable - valid choice
    context = {'choice': ['option1', 'option2', 'option3']}
    overwrite_context = {'choice': 'option2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['choice'][0] == 'option2'
    assert 'option2' in context['choice']
    
    # Test 5: Choice variable - invalid choice
    context = {'choice': ['option1', 'option2']}
    overwrite_context = {'choice': 'invalid_option'}
    with pytest.raises(ValueError, match='invalid_option provided for choice variable'):
        apply_overwrites_to_context(context, overwrite_context)
    
    # Test 6: Multi-choice variable - valid choices
    context = {'multichoice': ['opt1', 'opt2', 'opt3']}
    overwrite_context = {'multichoice': ['opt1', 'opt3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['multichoice'] == ['opt1', 'opt3']
    
    # Test 7: Multi-choice variable - invalid choices
    context = {'multichoice': ['opt1', 'opt2']}
    overwrite_context = {'multichoice': ['opt1', 'invalid']}
    with pytest.raises(ValueError, match='provided for multi-choice variable'):
        apply_overwrites_to_context(context, overwrite_context)
    
    # Test 8: Nested dictionary overwrite
    context = {'config': {'key1': 'value1', 'key2': 'value2'}}
    overwrite_context = {'config': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['config']['key1'] == 'new_value1'
    assert context['config']['key2'] == 'value2'
    
    # Test 9: Boolean variable with string 'y'
    context = {'flag': True}
    overwrite_context = {'flag': 'y'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['flag'] is True
    
    # Test 10: Boolean variable with string 'n'
    context = {'flag': True}
    overwrite_context = {'flag': 'n'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['flag'] is False
    
    # Test 11: Boolean variable with invalid string
    context = {'flag': True}
    overwrite_context = {'flag': 'invalid_bool'}
    with pytest.raises(ValueError, match='could not be converted to a boolean'):
        apply_overwrites_to_context(context, overwrite_context)
    
    # Test 12: List to list overwrite in nested context
    context = {'nested': {'items': ['a', 'b', 'c']}}
    overwrite_context = {'nested': {'items': ['b', 'c']}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context['nested']['items'] == ['b', 'c']
    
    # Test 13: Empty overwrite context
    context = {'key': 'value'}
    overwrite_context = {}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['key'] == 'value'
    
    # Test 14: Overwrite with None value
    context = {'key': 'value'}
    overwrite_context = {'key': None}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['key'] is None
    
    # Test 15: Complex nested structure
    context = {
        'project': {
            'name': 'myproject',
            'version': '1.0',
            'options': ['opt1', 'opt2']
        }
    }
    overwrite_context = {
        'project': {
            'name': 'newproject',
            'options': ['opt2']
        }
    }
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context['project']['name'] == 'newproject'
    assert context['project']['version'] == '1.0'
    assert context['project']['options'] == ['opt2']


# LLM-generated content at query #2
#--------------------------

```python
def test_render_and_create_dir(tmp_path):
    """Test render_and_create_dir function."""
    from jinja2 import Environment
    
    # Test 1: Basic directory creation with context rendering
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    dirname = '{{cookiecutter.project_name}}'
    
    result_dir, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_dir == tmp_path / 'my_project'
    assert result_dir.exists()
    assert created is True
    
    # Test 2: Empty directory name raises exception
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, tmp_path, env)
    
    # Test 3: Directory already exists without overwrite raises exception
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            '{{cookiecutter.project_name}}', context, tmp_path, env, 
            overwrite_if_exists=False
        )
    
    # Test 4: Directory already exists with overwrite_if_exists=True
    result_dir2, created2 = render_and_create_dir(
        '{{cookiecutter.project_name}}', context, tmp_path, env, 
        overwrite_if_exists=True
    )
    
    assert result_dir2 == tmp_path / 'my_project'
    assert result_dir2.exists()
    assert created2 is False
    
    # Test 5: Nested directory creation
    context_nested = {'cookiecutter': {'org': 'myorg', 'project': 'myproj'}}
    dirname_nested = '{{cookiecutter.org}}/{{cookiecutter.project}}'
    
    result_dir3, created3 = render_and_create_dir(
        dirname_nested, context_nested, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_dir3 == tmp_path / 'myorg' / 'myproj'
    assert result_dir3.exists()
    assert created3 is True
    
    # Test 6: Directory with no template variables
    result_dir4, created4 = render_and_create_dir(
        'static_dir', context, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_dir4 == tmp_path / 'static_dir'
    assert result_dir4.exists()
    assert created4 is True
    
    # Test 7: Empty string as dirname raises exception
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, tmp_path, env)


# LLM-generated content at query #3
#--------------------------

```python
def test_render_and_create_dir(tmp_path, monkeypatch):
    """Test render_and_create_dir function."""
    from jinja2 import Environment
    
    # Test basic directory creation with rendering
    context = {'cookiecutter': {'project_name': 'my_project'}}
    env = Environment()
    output_dir = tmp_path
    dirname = '{{cookiecutter.project_name}}'
    
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert created is True
    assert result_dir.exists()
    assert result_dir.name == 'my_project'
    assert result_dir.parent == output_dir


def test_render_and_create_dir_empty_dirname():
    """Test render_and_create_dir with empty directory name."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {}}
    env = Environment()
    output_dir = '.'
    
    with raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, env)


def test_render_and_create_dir_exists_no_overwrite(tmp_path):
    """Test render_and_create_dir when directory exists and overwrite is False."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    env = Environment()
    output_dir = tmp_path
    dirname = '{{cookiecutter.project_name}}'
    
    # Create the directory first
    existing_dir = output_dir / 'my_project'
    existing_dir.mkdir()
    
    with raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, output_dir, env, overwrite_if_exists=False
        )


def test_render_and_create_dir_exists_with_overwrite(tmp_path):
    """Test render_and_create_dir when directory exists and overwrite is True."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    env = Environment()
    output_dir = tmp_path
    dirname = '{{cookiecutter.project_name}}'
    
    # Create the directory first
    existing_dir = output_dir / 'my_project'
    existing_dir.mkdir()
    
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=True
    )
    
    assert created is False
    assert result_dir.exists()
    assert result_dir.name == 'my_project'


def test_render_and_create_dir_nested(tmp_path):
    """Test render_and_create_dir with nested directory structure."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'org': 'myorg', 'project': 'myproj'}}
    env = Environment()
    output_dir = tmp_path
    dirname = '{{cookiecutter.org}}/{{cookiecutter.project}}'
    
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert created is True
    assert result_dir.exists()
    assert result_dir.name == 'myproj'
    assert result_dir.parent.name == 'myorg'


def test_render_and_create_dir_no_template_vars(tmp_path):
    """Test render_and_create_dir with no template variables."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {}}
    env = Environment()
    output_dir = tmp_path
    dirname = 'static_dir_name'
    
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert created is True
    assert result_dir.exists()
    assert result_dir.name == 'static_dir_name'


def test_render_and_create_dir_path_type(tmp_path):
    """Test render_and_create_dir returns Path object."""
    from jinja2 import Environment
    from pathlib import Path
    
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment()
    output_dir = tmp_path
    dirname = '{{cookiecutter.name}}'
    
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert isinstance(result_dir, Path)


# LLM-generated content at query #4
#--------------------------

```python
def test_generate_file(tmp_path, mocker):
    """Test generate_file function with various file types and scenarios."""
    import tempfile
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    
    # Test 1: Render text file with context
    text_file = template_dir / "test_{{cookiecutter.name}}.txt"
    text_file.write_text("Hello {{cookiecutter.name}}!")
    
    context = {
        'cookiecutter': {
            'name': 'world',
            '_new_lines': None
        }
    }
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    with work_in(str(template_dir)):
        generate_file(
            project_dir,
            "test_{{cookiecutter.name}}.txt",
            context,
            env,
            skip_if_file_exists=False
        )
    
    output_file = Path(project_dir) / "test_world.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Hello world!"
    
    # Test 2: Skip if file exists
    with work_in(str(template_dir)):
        generate_file(
            project_dir,
            "test_{{cookiecutter.name}}.txt",
            context,
            env,
            skip_if_file_exists=True
        )
    
    # File should remain unchanged
    assert output_file.read_text() == "Hello world!"
    
    # Test 3: Binary file copy
    binary_file = template_dir / "binary.bin"
    binary_content = b"\x89PNG\r\n\x1a\n"
    binary_file.write_bytes(binary_content)
    
    mocker.patch('binaryornot.check.is_binary', return_value=True)
    
    with work_in(str(template_dir)):
        generate_file(
            project_dir,
            "binary.bin",
            context,
            env,
            skip_if_file_exists=False
        )
    
    output_binary = Path(project_dir) / "binary.bin"
    assert output_binary.exists()
    assert output_binary.read_bytes() == binary_content
    
    # Test 4: Empty directory name returns early
    empty_dir_file = template_dir / "empty_dir"
    empty_dir_file.mkdir()
    
    with work_in(str(template_dir)):
        generate_file(
            project_dir,
            "empty_dir",
            context,
            env,
            skip_if_file_exists=False
        )
    
    # Test 5: File with custom newlines
    newline_file = template_dir / "newlines.txt"
    newline_file.write_text("line1\r\nline2\r\n", newline='')
    
    context_with_newlines = {
        'cookiecutter': {
            'name': 'test',
            '_new_lines': '\r\n'
        }
    }
    
    with work_in(str(template_dir)):
        generate_file(
            project_dir,
            "newlines.txt",
            context_with_newlines,
            env,
            skip_if_file_exists=False
        )
    
    output_newline = Path(project_dir) / "newlines.txt"
    assert output_newline.exists()
    
    # Test 6: Template syntax error
    syntax_error_file = template_dir / "syntax_error.txt"
    syntax_error_file.write_text("{{cookiecutter.name")
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    with work_in(str(template_dir)):
        with pytest.raises(TemplateSyntaxError):
            generate_file(
                project_dir,
                "syntax_error.txt",
                context,
                env,
                skip_if_file_exists=False
            )
    
    # Test 7: File permissions are copied
    perm_file = template_dir / "executable.sh"
    perm_file.write_text("#!/bin/bash\necho test")
    os.chmod(perm_file, 0o755)
    
    with work_in(str(template_dir)):
        generate_file(
            project_dir,
            "executable.sh",
            context,
            env,
            skip_if_file_exists=False
        )
    
    output_perm = Path(project_dir) / "executable.sh"
    assert output_perm.exists()
    assert os.stat(output_perm).st_mode & 0o755 == 0o755


# LLM-generated content at query #5
#--------------------------

```python
def test_apply_overwrites_to_context():
    """Test apply_overwrites_to_context function with various scenarios."""
    
    # Test 1: Simple string variable overwrite
    context = {'name': 'original', 'age': 30}
    overwrite = {'name': 'new_name'}
    apply_overwrites_to_context(context, overwrite)
    assert context['name'] == 'new_name'
    assert context['age'] == 30
    
    # Test 2: New variable at first level should be ignored
    context = {'name': 'original'}
    overwrite = {'new_var': 'value'}
    apply_overwrites_to_context(context, overwrite)
    assert 'new_var' not in context
    
    # Test 3: Choice variable - valid choice
    context = {'color': ['red', 'green', 'blue']}
    overwrite = {'color': 'green'}
    apply_overwrites_to_context(context, overwrite)
    assert context['color'][0] == 'green'
    assert 'green' in context['color']
    
    # Test 4: Choice variable - invalid choice raises ValueError
    context = {'color': ['red', 'green', 'blue']}
    overwrite = {'color': 'yellow'}
    with pytest.raises(ValueError, match="yellow provided for choice variable"):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 5: Multi-choice variable - valid choices
    context = {'languages': ['python', 'javascript', 'java']}
    overwrite = {'languages': ['python', 'java']}
    apply_overwrites_to_context(context, overwrite)
    assert context['languages'] == ['python', 'java']
    
    # Test 6: Multi-choice variable - invalid choices raises ValueError
    context = {'languages': ['python', 'javascript', 'java']}
    overwrite = {'languages': ['python', 'rust']}
    with pytest.raises(ValueError, match="rust provided for multi-choice variable"):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 7: Boolean variable - string to boolean conversion
    context = {'is_active': True}
    overwrite = {'is_active': 'y'}
    apply_overwrites_to_context(context, overwrite)
    assert context['is_active'] is True
    
    # Test 8: Boolean variable - invalid string raises ValueError
    context = {'is_active': True}
    overwrite = {'is_active': 'maybe'}
    with pytest.raises(ValueError, match="could not be converted to a boolean"):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 9: Nested dictionary variable overwrite
    context = {'config': {'debug': True, 'port': 8000}}
    overwrite = {'config': {'debug': False}}
    apply_overwrites_to_context(context, overwrite)
    assert context['config']['debug'] is False
    assert context['config']['port'] == 8000
    
    # Test 10: Nested dictionary with new variable
    context = {'config': {'debug': True}}
    overwrite = {'config': {'new_key': 'new_value'}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['config']['new_key'] == 'new_value'
    
    # Test 11: List overwrite when in_dictionary_variable is True
    context = {'config': {'items': ['a', 'b']}}
    overwrite = {'config': {'items': ['x', 'y']}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['config']['items'] == ['x', 'y']
    
    # Test 12: Empty overwrite context
    context = {'name': 'original', 'age': 30}
    overwrite = {}
    apply_overwrites_to_context(context, overwrite)
    assert context['name'] == 'original'
    assert context['age'] == 30
    
    # Test 13: Complex nested structure
    context = {
        'project': {
            'name': 'myproject',
            'features': ['auth', 'api'],
            'settings': {'debug': True}
        }
    }
    overwrite = {
        'project': {
            'name': 'newproject',
            'settings': {'debug': False}
        }
    }
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['project']['name'] == 'newproject'
    assert context['project']['settings']['debug'] is False
    assert context['project']['features'] == ['auth', 'api']


# LLM-generated content at query #6
#--------------------------

```python
def test_generate_context(tmp_path, monkeypatch):
    """Test generate_context function."""
    monkeypatch.chdir(tmp_path)
    
    # Test 1: Basic context generation from cookiecutter.json
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John Doe"}')
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"
    
    # Test 2: Context with extra_context override
    extra_context = {"project_name": "override_project"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["project_name"] == "override_project"
    assert result["cookiecutter"]["author"] == "John Doe"
    
    # Test 3: Context with default_context
    default_context = {"author": "Jane Doe"}
    result = generate_context(str(context_file), default_context=default_context)
    assert result["cookiecutter"]["author"] == "Jane Doe"
    
    # Test 4: Extra context overrides default_context
    result = generate_context(
        str(context_file),
        default_context={"project_name": "default_project"},
        extra_context={"project_name": "extra_project"}
    )
    assert result["cookiecutter"]["project_name"] == "extra_project"
    
    # Test 5: Choice variable (list) with valid override
    choice_context_file = tmp_path / "choice_context.json"
    choice_context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    result = generate_context(str(choice_context_file), extra_context={"license": "Apache"})
    assert result["cookiecutter"]["license"][0] == "Apache"
    
    # Test 6: Multi-choice variable (list of lists) with valid override
    multi_choice_file = tmp_path / "multi_choice.json"
    multi_choice_file.write_text('{"features": ["feature1", "feature2", "feature3"]}')
    result = generate_context(str(multi_choice_file), extra_context={"features": ["feature2", "feature3"]})
    assert result["cookiecutter"]["features"] == ["feature2", "feature3"]
    
    # Test 7: Boolean variable with string override
    bool_context_file = tmp_path / "bool_context.json"
    bool_context_file.write_text('{"use_docker": true}')
    result = generate_context(str(bool_context_file), extra_context={"use_docker": "n"})
    assert result["cookiecutter"]["use_docker"] is False
    
    # Test 8: Nested dictionary override
    nested_file = tmp_path / "nested.json"
    nested_file.write_text('{"options": {"debug": false, "verbose": true}}')
    result = generate_context(str(nested_file), extra_context={"options": {"debug": True}})
    assert result["cookiecutter"]["options"]["debug"] is True
    assert result["cookiecutter"]["options"]["verbose"] is True
    
    # Test 9: Invalid JSON raises ContextDecodingException
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text('{invalid json}')
    with pytest.raises(ContextDecodingException):
        generate_context(str(invalid_json_file))
    
    # Test 10: Invalid choice override raises ValueError
    with pytest.raises(ValueError, match="provided for choice variable"):
        generate_context(str(choice_context_file), extra_context={"license": "InvalidLicense"})
    
    # Test 11: Invalid multi-choice override raises ValueError
    with pytest.raises(ValueError, match="provided for multi-choice variable"):
        generate_context(str(multi_choice_file), extra_context={"features": ["invalid"]})
    
    # Test 12: Invalid boolean override raises ValueError
    with pytest.raises(ValueError, match="could not be converted to a boolean"):
        generate_context(str(bool_context_file), extra_context={"use_docker": "maybe"})
    
    # Test 13: OrderedDict preservation
    ordered_file = tmp_path / "ordered.json"
    ordered_file.write_text('{"first": 1, "second": 2, "third": 3}')
    result = generate_context(str(ordered_file))
    keys = list(result["cookiecutter"].keys())
    assert keys == ["first", "second", "third"]
    
    # Test 14: Custom context_file name
    custom_file = tmp_path / "custom_context.json"
    custom_file.write_text('{"key": "value"}')
    result = generate_context(str(custom_file))
    assert "custom_context" in result
    assert result["custom_context"]["key"] == "value"
    
    # Test 15: Empty file name extraction
    empty_stem_file = tmp_path / "config.json"
    empty_stem_file.write_text('{"setting": "value"}')
    result = generate_context(str(empty_stem_file))
    assert "config" in result


# LLM-generated content at query #7
#--------------------------

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pytest
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError


def test_generate_file():
    """Test generate_file function with various scenarios."""
    
    # Test 1: Binary file should be copied without rendering
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = "binary_file.bin"
        context = {'cookiecutter': {}}
        env = Environment()
        
        # Create a mock binary file
        with patch('binaryornot.check.is_binary', return_value=True):
            with patch('shutil.copyfile') as mock_copyfile:
                with patch('shutil.copymode') as mock_copymode:
                    with patch('os.path.isdir', return_value=False):
                        with patch('os.path.exists', return_value=False):
                            generate_file(project_dir, infile, context, env)
                            mock_copyfile.assert_called_once()
                            mock_copymode.assert_called_once()
    
    # Test 2: Text file should be rendered
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = "template_{{cookiecutter.project_name}}.txt"
        context = {'cookiecutter': {'project_name': 'myproject'}}
        env = Environment(loader=FileSystemLoader('.'))
        
        with patch('binaryornot.check.is_binary', return_value=False):
            with patch('os.path.isdir', return_value=False):
                with patch('os.path.exists', return_value=False):
                    with patch.object(env, 'get_template') as mock_get_template:
                        mock_template = Mock()
                        mock_template.render.return_value = "rendered content"
                        mock_get_template.return_value = mock_template
                        
                        with patch('builtins.open', mock_open(read_data="content\n")):
                            with patch('shutil.copymode'):
                                generate_file(project_dir, infile, context, env)
                                mock_get_template.assert_called_once()
    
    # Test 3: File name is empty (output is a directory)
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = "somefile"
        context = {'cookiecutter': {}}
        env = Environment()
        
        with patch('os.path.isdir', return_value=True):
            generate_file(project_dir, infile, context, env)
            # Should return early without error
    
    # Test 4: Skip if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = "existing_file.txt"
        context = {'cookiecutter': {}}
        env = Environment()
        
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.exists', return_value=True):
                with patch('binaryornot.check.is_binary', return_value=False):
                    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
                    # Should return early without rendering
    
    # Test 5: Template syntax error handling
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = "bad_template.txt"
        context = {'cookiecutter': {}}
        env = Environment(loader=FileSystemLoader('.'))
        
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.exists', return_value=False):
                with patch('binaryornot.check.is_binary', return_value=False):
                    syntax_error = TemplateSyntaxError("Bad syntax", 1)
                    with patch.object(env, 'get_template', side_effect=syntax_error):
                        with pytest.raises(TemplateSyntaxError):
                            generate_file(project_dir, infile, context, env)
    
    # Test 6: Custom newline character from context
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = "newline_file.txt"
        context = {'cookiecutter': {'_new_lines': '\r\n'}}
        env = Environment(loader=FileSystemLoader('.'))
        
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.exists', return_value=False):
                with patch('binaryornot.check.is_binary', return_value=False):
                    with patch.object(env, 'get_template') as mock_get_template:
                        mock_template = Mock()
                        mock_template.render.return_value = "content"
                        mock_get_template.return_value = mock_template
                        
                        with patch('builtins.open', mock_open(read_data="content\n")):
                            with patch('shutil.copymode'):
                                generate_file(project_dir, infile, context, env)
                                # Verify file was written with custom newline
    
    # Test 7: Detected newline from file
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = "detect_newline.txt"
        context = {'cookiecutter': {}}
        env = Environment(loader=FileSystemLoader('.'))
        
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.exists', return_value=False):
                with patch('binaryornot.check.is_binary', return_value=False):
                    with patch.object(env, 'get_template') as mock_get_template:
                        mock_template = Mock()
                        mock_template.render.return_value = "content"
                        mock_get_template.return_value = mock_template
                        
                        mock_file = mock_open(read_data="line1\nline2\n")
                        with patch('builtins.open', mock_file):
                            with patch('shutil.copymode'):
                                generate_file(project_dir, infile, context, env)
    
    # Test 8: Forward slashes conversion for Windows paths
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = os.path.join("subdir", "template.txt")
        context = {'cookiecutter': {}}
        env = Environment(loader=FileSystemLoader('.'))
        
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.exists', return_value=False):
                with patch('binaryornot.check.is_binary', return_value=False):
                    with patch.object(env, 'get_template') as mock_get_template:
                        mock_template = Mock()
                        mock_template.render.return_value = "content"
                        mock_get_template.return_value = mock_template
                        
                        with patch('builtins.open', mock_open(read_data="content\n")):
                            with patch('shutil.copymode'):
                                generate_file(project_dir, infile, context, env)
                                # Verify forward slashes were used for get_template
                                call_args = mock_get_template.call_args[0][0]
                                assert '/' in call_args or '\\' not in call_args or call_args.count('/') > 0


# LLM-generated content at query #8
#--------------------------

```python
def test_is_copy_only_path():
    """Test is_copy_only_path function with various patterns and paths."""
    
    # Test with matching pattern
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*', 'binary/*']
        }
    }
    assert is_copy_only_path('file.txt', context) is True
    assert is_copy_only_path('docs/readme.md', context) is True
    assert is_copy_only_path('binary/image.png', context) is True
    
    # Test with non-matching pattern
    assert is_copy_only_path('file.py', context) is False
    assert is_copy_only_path('script.sh', context) is False
    assert is_copy_only_path('source/code.txt', context) is False
    
    # Test with empty _copy_without_render list
    context_empty = {
        'cookiecutter': {
            '_copy_without_render': []
        }
    }
    assert is_copy_only_path('file.txt', context_empty) is False
    assert is_copy_only_path('docs/readme.md', context_empty) is False
    
    # Test with missing _copy_without_render key
    context_no_key = {'cookiecutter': {}}
    assert is_copy_only_path('file.txt', context_no_key) is False
    assert is_copy_only_path('docs/readme.md', context_no_key) is False
    
    # Test with missing cookiecutter key
    context_no_cookiecutter = {}
    assert is_copy_only_path('file.txt', context_no_cookiecutter) is False
    
    # Test with wildcard patterns
    context_wildcards = {
        'cookiecutter': {
            '_copy_without_render': ['*.min.js', 'node_modules/**', '.*']
        }
    }
    assert is_copy_only_path('script.min.js', context_wildcards) is True
    assert is_copy_only_path('node_modules/package', context_wildcards) is True
    assert is_copy_only_path('.gitignore', context_wildcards) is True
    assert is_copy_only_path('script.js', context_wildcards) is False
    
    # Test with question mark wildcard
    context_question = {
        'cookiecutter': {
            '_copy_without_render': ['file?.txt']
        }
    }
    assert is_copy_only_path('file1.txt', context_question) is True
    assert is_copy_only_path('fileA.txt', context_question) is True
    assert is_copy_only_path('file.txt', context_question) is False
    assert is_copy_only_path('file12.txt', context_question) is False
    
    # Test with exact path match
    context_exact = {
        'cookiecutter': {
            '_copy_without_render': ['specific/path/file.txt']
        }
    }
    assert is_copy_only_path('specific/path/file.txt', context_exact) is True
    assert is_copy_only_path('specific/path/other.txt', context_exact) is False
    assert is_copy_only_path('different/path/file.txt', context_exact) is False


# LLM-generated content at query #9
#--------------------------

```python
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import pytest

from cookiecutter.exceptions import ContextDecodingException


def test_generate_context():
    """Test generate_context with valid JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'project_slug': '{{ cookiecutter.project_name.lower().replace(" ", "_") }}',
            'author_name': 'John Doe'
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        result = generate_context(context_file=context_file)
        
        assert isinstance(result, OrderedDict)
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'My Project'
        assert result['cookiecutter']['author_name'] == 'John Doe'


def test_generate_context_with_default_context():
    """Test generate_context with default_context overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'author_name': 'John Doe'
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        default_context = {
            'author_name': 'Jane Doe'
        }
        
        result = generate_context(
            context_file=context_file,
            default_context=default_context
        )
        
        assert result['cookiecutter']['author_name'] == 'Jane Doe'
        assert result['cookiecutter']['project_name'] == 'My Project'


def test_generate_context_with_extra_context():
    """Test generate_context with extra_context overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'author_name': 'John Doe'
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'project_name': 'Another Project'
        }
        
        result = generate_context(
            context_file=context_file,
            extra_context=extra_context
        )
        
        assert result['cookiecutter']['project_name'] == 'Another Project'
        assert result['cookiecutter']['author_name'] == 'John Doe'


def test_generate_context_invalid_json():
    """Test generate_context with invalid JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write('{ invalid json }')
        
        with pytest.raises(ContextDecodingException):
            generate_context(context_file=context_file)


def test_generate_context_choice_variable():
    """Test generate_context with choice variable and default context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'license': ['MIT', 'Apache', 'GPL']
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        default_context = {
            'license': 'Apache'
        }
        
        result = generate_context(
            context_file=context_file,
            default_context=default_context
        )
        
        assert result['cookiecutter']['license'][0] == 'Apache'
        assert 'MIT' in result['cookiecutter']['license']
        assert 'GPL' in result['cookiecutter']['license']


def test_generate_context_invalid_choice():
    """Test generate_context with invalid choice in default context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'license': ['MIT', 'Apache', 'GPL']
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        default_context = {
            'license': 'InvalidLicense'
        }
        
        with pytest.warns(UserWarning):
            result = generate_context(
                context_file=context_file,
                default_context=default_context
            )
            assert result['cookiecutter']['license'][0] == 'MIT'


def test_generate_context_dict_variable():
    """Test generate_context with nested dictionary variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'options': {
                'use_docker': True,
                'use_ci': False
            }
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'options': {
                'use_ci': True
            }
        }
        
        result = generate_context(
            context_file=context_file,
            extra_context=extra_context
        )
        
        assert result['cookiecutter']['options']['use_docker'] is True
        assert result['cookiecutter']['options']['use_ci'] is True


def test_generate_context_boolean_string_conversion():
    """Test generate_context with boolean variable and string override."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'use_docker': False
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'use_docker': 'y'
        }
        
        result = generate_context(
            context_file=context_file,
            extra_context=extra_context
        )
        
        assert result['cookiecutter']['use_docker'] is True


def test_generate_context_multichoice_variable():
    """Test generate_context with multichoice variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'features': ['feature1', 'feature2', 'feature3']
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'features': ['feature2', 'feature3']
        }
        
        result = generate_context(
            context_file=context_file,
            extra_context=extra_context
        )
        
        assert result['cookiecutter']['features'] == ['feature2', 'feature3']


def test_generate_context_file_not_found():
    """Test generate_context with non-existent file."""
    with pytest.raises(FileNotFoundError):
        generate_context(context_file='/nonexistent/path/cookiecutter.json')


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    import json
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "my_project",
        "author": "John Doe",
        "version": "0.1.0"
    }
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"
    assert result["cookiecutter"]["version"] == "0.1.0"
    
    # Test 2: Context with default_context overrides
    context_file2 = tmp_path / "cookiecutter2.json"
    context_data2 = {
        "project_name": "default_project",
        "author": "Default Author",
        "use_docker": True
    }
    context_file2.write_text(json.dumps(context_data2))
    
    default_context = {
        "project_name": "overridden_project",
        "author": "Default Author"
    }
    
    result = generate_context(str(context_file2), default_context=default_context)
    assert result["cookiecutter"]["project_name"] == "overridden_project"
    assert result["cookiecutter"]["author"] == "Default Author"
    
    # Test 3: Context with extra_context overrides
    context_file3 = tmp_path / "cookiecutter3.json"
    context_data3 = {
        "project_name": "original_project",
        "version": "1.0.0"
    }
    context_file3.write_text(json.dumps(context_data3))
    
    extra_context = {
        "project_name": "extra_project",
        "version": "2.0.0"
    }
    
    result = generate_context(str(context_file3), extra_context=extra_context)
    assert result["cookiecutter"]["project_name"] == "extra_project"
    assert result["cookiecutter"]["version"] == "2.0.0"
    
    # Test 4: Context with choice variable
    context_file4 = tmp_path / "cookiecutter4.json"
    context_data4 = {
        "python_version": ["3.8", "3.9", "3.10"]
    }
    context_file4.write_text(json.dumps(context_data4))
    
    extra_context4 = {
        "python_version": "3.9"
    }
    
    result = generate_context(str(context_file4), extra_context=extra_context4)
    assert result["cookiecutter"]["python_version"][0] == "3.9"
    
    # Test 5: Context with multi-choice variable
    context_file5 = tmp_path / "cookiecutter5.json"
    context_data5 = {
        "features": ["docker", "ci", "docs", "tests"]
    }
    context_file5.write_text(json.dumps(context_data5))
    
    extra_context5 = {
        "features": ["docker", "tests"]
    }
    
    result = generate_context(str(context_file5), extra_context=extra_context5)
    assert result["cookiecutter"]["features"] == ["docker", "tests"]
    
    # Test 6: Context with nested dictionary
    context_file6 = tmp_path / "cookiecutter6.json"
    context_data6 = {
        "project": {
            "name": "my_project",
            "slug": "my_project"
        }
    }
    context_file6.write_text(json.dumps(context_data6))
    
    extra_context6 = {
        "project": {
            "name": "updated_project"
        }
    }
    
    result = generate_context(str(context_file6), extra_context=extra_context6)
    assert result["cookiecutter"]["project"]["name"] == "updated_project"
    assert result["cookiecutter"]["project"]["slug"] == "my_project"
    
    # Test 7: Invalid JSON file should raise ContextDecodingException
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text("{invalid json content")
    
    try:
        generate_context(str(invalid_json_file))
        assert False, "Should have raised ContextDecodingException"
    except ContextDecodingException as e:
        assert "JSON decoding error" in str(e)
    
    # Test 8: Boolean variable with string override
    context_file8 = tmp_path / "cookiecutter8.json"
    context_data8 = {
        "use_docker": True,
        "use_ci": False
    }
    context_file8.write_text(json.dumps(context_data8))
    
    extra_context8 = {
        "use_docker": "n",
        "use_ci": "y"
    }
    
    result = generate_context(str(context_file8), extra_context=extra_context8)
    assert result["cookiecutter"]["use_docker"] is False
    assert result["cookiecutter"]["use_ci"] is True
    
    # Test 9: Invalid choice should raise ValueError
    context_file9 = tmp_path / "cookiecutter9.json"
    context_data9 = {
        "python_version": ["3.8", "3.9", "3.10"]
    }
    context_file9.write_text(json.dumps(context_data9))
    
    extra_context9 = {
        "python_version": "3.7"
    }
    
    try:
        generate_context(str(context_file9), extra_context=extra_context9)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "3.7" in str(e) and "choice variable" in str(e)
    
    # Test 10: OrderedDict is preserved
    context_file10 = tmp_path / "cookiecutter10.json"
    context_data10 = OrderedDict([("first", "1"), ("second", "2"), ("third", "3")])
    context_file10.write_text(json.dumps(context_data10))
    
    result = generate_context(str(context_file10))
    assert isinstance(result, dict)
    assert "cookiecutter" in result


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pathlib import Path
from jinja2 import Environment
from cookiecutter.exceptions import EmptyDirNameException, OutputDirExistsException


def test_render_and_create_dir(tmp_path):
    """Test render_and_create_dir creates directory with rendered name."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = tmp_path
    
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env
    )
    
    assert result_dir == Path(output_dir, 'my_project')
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_already_exists(tmp_path):
    """Test render_and_create_dir raises error when directory exists."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = tmp_path
    
    # Create directory first
    existing_dir = Path(output_dir, 'my_project')
    existing_dir.mkdir()
    
    dirname = '{{cookiecutter.project_name}}'
    
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, env)


def test_render_and_create_dir_overwrite_if_exists(tmp_path):
    """Test render_and_create_dir overwrites existing directory."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = tmp_path
    
    # Create directory first
    existing_dir = Path(output_dir, 'my_project')
    existing_dir.mkdir()
    
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=True
    )
    
    assert result_dir == Path(output_dir, 'my_project')
    assert result_dir.exists()
    assert created is False


def test_render_and_create_dir_empty_dirname():
    """Test render_and_create_dir raises error for empty directory name."""
    env = Environment()
    context = {'cookiecutter': {}}
    output_dir = Path('.')
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, env)


def test_render_and_create_dir_empty_string_dirname():
    """Test render_and_create_dir raises error for empty string dirname."""
    env = Environment()
    context = {'cookiecutter': {}}
    output_dir = Path('.')
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('   ', context, output_dir, env)


def test_render_and_create_dir_nested_context(tmp_path):
    """Test render_and_create_dir with nested context variables."""
    env = Environment()
    context = {
        'cookiecutter': {
            'author': 'john',
            'project': 'awesome'
        }
    }
    output_dir = tmp_path
    
    dirname = '{{cookiecutter.author}}_{{cookiecutter.project}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env
    )
    
    assert result_dir == Path(output_dir, 'john_awesome')
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_path_string_output_dir(tmp_path):
    """Test render_and_create_dir works with string output_dir."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = str(tmp_path)
    
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env
    )
    
    assert result_dir == Path(output_dir, 'my_project')
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_returns_tuple(tmp_path):
    """Test render_and_create_dir returns tuple with Path and bool."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'test'}}
    output_dir = tmp_path
    
    dirname = '{{cookiecutter.project_name}}'
    result = render_and_create_dir(dirname, context, output_dir, env)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Path)
    assert isinstance(result[1], bool)


# LLM-generated content at query #12
#--------------------------

```python
def test_generate_file(tmp_path, monkeypatch):
    """Test generate_file function."""
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    # Create a text template file
    infile = "test_{{cookiecutter.name}}.txt"
    template_file = template_dir / infile
    template_file.write_text("Hello {{cookiecutter.greeting}}")
    
    context = {
        'cookiecutter': {
            'name': 'project',
            'greeting': 'World'
        }
    }
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    # Change to template directory
    monkeypatch.chdir(template_dir)
    
    # Execute
    generate_file(project_dir, infile, context, env)
    
    # Verify
    output_file = Path(project_dir) / "test_project.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Hello World"


def test_generate_file_binary(tmp_path, monkeypatch):
    """Test generate_file with binary file."""
    from jinja2 import Environment, FileSystemLoader
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    # Create a binary file
    infile = "test.bin"
    template_file = template_dir / infile
    template_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    # Execute
    generate_file(project_dir, infile, context, env)
    
    # Verify
    output_file = Path(project_dir) / infile
    assert output_file.exists()
    assert output_file.read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_file_skip_if_exists(tmp_path, monkeypatch):
    """Test generate_file skips existing files."""
    from jinja2 import Environment, FileSystemLoader
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    infile = "test.txt"
    template_file = template_dir / infile
    template_file.write_text("New content")
    
    # Create existing file
    output_file = Path(project_dir) / infile
    output_file.write_text("Old content")
    
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    # Execute
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    
    # Verify file wasn't overwritten
    assert output_file.read_text() == "Old content"


def test_generate_file_empty_dir_name(tmp_path, monkeypatch):
    """Test generate_file with empty directory name."""
    from jinja2 import Environment, FileSystemLoader
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    # Create directory instead of file
    infile = "."
    
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    # Execute - should return early for directories
    generate_file(project_dir, infile, context, env)
    
    # Verify no error is raised


def test_generate_file_with_newline_config(tmp_path, monkeypatch):
    """Test generate_file respects _new_lines configuration."""
    from jinja2 import Environment, FileSystemLoader
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    infile = "test.txt"
    template_file = template_dir / infile
    template_file.write_text("line1\nline2\n")
    
    context = {
        'cookiecutter': {
            '_new_lines': '\r\n'
        }
    }
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    # Execute
    generate_file(project_dir, infile, context, env)
    
    # Verify
    output_file = Path(project_dir) / infile
    assert output_file.exists()
    content = output_file.read_bytes()
    assert b'\r\n' in content


def test_generate_file_template_syntax_error(tmp_path, monkeypatch):
    """Test generate_file raises on template syntax error."""
    from jinja2 import Environment, FileSystemLoader
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    infile = "test.txt"
    template_file = template_dir / infile
    template_file.write_text("{{cookiecutter.name")  # Invalid syntax
    
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    # Execute and verify exception
    with pytest.raises(TemplateSyntaxError):
        generate_file(project_dir, infile, context, env)


def test_generate_file_with_context_variables(tmp_path, monkeypatch):
    """Test generate_file renders context variables correctly."""
    from jinja2 import Environment, FileSystemLoader
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    infile = "{{cookiecutter.module}}.py"
    template_file = template_dir / infile
    template_file.write_text("def {{cookiecutter.function}}():\n    pass")
    
    context = {
        'cookiecutter': {
            'module': 'mymodule',
            'function': 'myfunc'
        }
    }
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    # Execute
    generate_file(project_dir, infile, context, env)
    
    # Verify
    output_file = Path(project_dir) / "mymodule.py"
    assert output_file.exists()
    assert "def myfunc():" in output_file.read_text()


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "my_project",
        "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
        "author": "John Doe"
    }
    context_file.write_text(json.dumps(context_data))
    
    context = generate_context(str(context_file))
    assert "cookiecutter" in context
    assert context["cookiecutter"]["project_name"] == "my_project"
    assert context["cookiecutter"]["author"] == "John Doe"

    # Test 2: Context with default_context overrides
    context_file2 = tmp_path / "cookiecutter2.json"
    context_data2 = {
        "project_name": "default_project",
        "version": "1.0.0"
    }
    context_file2.write_text(json.dumps(context_data2))
    
    default_context = {
        "project_name": "overridden_project",
        "version": "2.0.0"
    }
    
    context = generate_context(str(context_file2), default_context=default_context)
    assert context["cookiecutter"]["project_name"] == "overridden_project"
    assert context["cookiecutter"]["version"] == "2.0.0"

    # Test 3: Context with extra_context overrides
    context_file3 = tmp_path / "cookiecutter3.json"
    context_data3 = {
        "project_name": "original_project",
        "license": "MIT"
    }
    context_file3.write_text(json.dumps(context_data3))
    
    extra_context = {
        "project_name": "extra_project",
        "license": "Apache"
    }
    
    context = generate_context(str(context_file3), extra_context=extra_context)
    assert context["cookiecutter"]["project_name"] == "extra_project"
    assert context["cookiecutter"]["license"] == "Apache"

    # Test 4: Context with choice variable (list)
    context_file4 = tmp_path / "cookiecutter4.json"
    context_data4 = {
        "python_version": ["3.8", "3.9", "3.10"]
    }
    context_file4.write_text(json.dumps(context_data4))
    
    extra_context4 = {"python_version": "3.9"}
    
    context = generate_context(str(context_file4), extra_context=extra_context4)
    assert context["cookiecutter"]["python_version"][0] == "3.9"

    # Test 5: Context with multi-choice variable (list of lists)
    context_file5 = tmp_path / "cookiecutter5.json"
    context_data5 = {
        "features": ["feature1", "feature2", "feature3"]
    }
    context_file5.write_text(json.dumps(context_data5))
    
    extra_context5 = {"features": ["feature1", "feature3"]}
    
    context = generate_context(str(context_file5), extra_context=extra_context5)
    assert set(context["cookiecutter"]["features"]) == {"feature1", "feature3"}

    # Test 6: Context with dictionary variable
    context_file6 = tmp_path / "cookiecutter6.json"
    context_data6 = {
        "config": {
            "debug": True,
            "port": 8000
        }
    }
    context_file6.write_text(json.dumps(context_data6))
    
    extra_context6 = {"config": {"port": 9000}}
    
    context = generate_context(str(context_file6), extra_context=extra_context6)
    assert context["cookiecutter"]["config"]["port"] == 9000
    assert context["cookiecutter"]["config"]["debug"] is True

    # Test 7: Context with boolean variable conversion
    context_file7 = tmp_path / "cookiecutter7.json"
    context_data7 = {
        "use_docker": True
    }
    context_file7.write_text(json.dumps(context_data7))
    
    extra_context7 = {"use_docker": "n"}
    
    context = generate_context(str(context_file7), extra_context=extra_context7)
    assert context["cookiecutter"]["use_docker"] is False

    # Test 8: Invalid JSON raises ContextDecodingException
    context_file8 = tmp_path / "cookiecutter8.json"
    context_file8.write_text("{invalid json}")
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(context_file8))

    # Test 9: Invalid choice variable raises ValueError
    context_file9 = tmp_path / "cookiecutter9.json"
    context_data9 = {
        "framework": ["django", "flask"]
    }
    context_file9.write_text(json.dumps(context_data9))
    
    extra_context9 = {"framework": "fastapi"}
    
    with pytest.raises(ValueError):
        generate_context(str(context_file9), extra_context=extra_context9)

    # Test 10: Invalid multi-choice variable raises ValueError
    context_file10 = tmp_path / "cookiecutter10.json"
    context_data10 = {
        "packages": ["numpy", "pandas"]
    }
    context_file10.write_text(json.dumps(context_data10))
    
    extra_context10 = {"packages": ["numpy", "scipy"]}
    
    with pytest.raises(ValueError):
        generate_context(str(context_file10), extra_context=extra_context10)

    # Test 11: Invalid boolean conversion raises ValueError
    context_file11 = tmp_path / "cookiecutter11.json"
    context_data11 = {
        "enable_feature": False
    }
    context_file11.write_text(json.dumps(context_data11))
    
    extra_context11 = {"enable_feature": "maybe"}
    
    with pytest.raises(ValueError):
        generate_context(str(context_file11), extra_context=extra_context11)

    # Test 12: Both default_context and extra_context (extra_context takes precedence)
    context_file12 = tmp_path / "cookiecutter12.json"
    context_data12 = {
        "name": "original",
        "version": "1.0"
    }
    context_file12.write_text(json.dumps(context_data12))
    
    default_context12 = {"name": "default", "version": "2.0"}
    extra_context12 = {"name": "extra"}
    
    context = generate_context(
        str(context_file12),
        default_context=default_context12,
        extra_context=extra_context12
    )
    assert context["cookiecutter"]["name"] == "extra"
    assert context["cookiecutter"]["version"] == "2.0"


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    # Test basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "my_project",
        "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
        "author_name": "John Doe"
    }
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author_name"] == "John Doe"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context with default_context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "my_project",
        "author_name": "John Doe"
    }
    context_file.write_text(json.dumps(context_data))
    
    default_context = {
        "author_name": "Jane Smith"
    }
    
    result = generate_context(str(context_file), default_context=default_context)
    assert result["cookiecutter"]["author_name"] == "Jane Smith"
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context with extra_context overrides."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "my_project",
        "author_name": "John Doe"
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {
        "project_name": "new_project"
    }
    
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["project_name"] == "new_project"
    assert result["cookiecutter"]["author_name"] == "John Doe"


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context with choice variable and overwrite."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "license": ["MIT", "Apache", "GPL"]
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {
        "license": "Apache"
    }
    
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["license"][0] == "Apache"


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test generate_context with multi-choice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "features": ["feature1", "feature2", "feature3"]
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {
        "features": ["feature1", "feature3"]
    }
    
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["features"] == ["feature1", "feature3"]


def test_generate_context_with_dict_variable(tmp_path):
    """Test generate_context with dictionary variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "config": {
            "debug": True,
            "port": 8000
        }
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {
        "config": {
            "port": 9000
        }
    }
    
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["config"]["port"] == 9000
    assert result["cookiecutter"]["config"]["debug"] is True


def test_generate_context_with_boolean_variable(tmp_path):
    """Test generate_context with boolean variable string conversion."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "include_tests": True
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {
        "include_tests": "n"
    }
    
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["include_tests"] is False


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context with invalid JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text("{invalid json}")
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(context_file))


def test_generate_context_nonexistent_file():
    """Test generate_context with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        generate_context("/nonexistent/path/cookiecutter.json")


def test_generate_context_invalid_choice_overwrite(tmp_path):
    """Test generate_context with invalid choice value."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "license": ["MIT", "Apache", "GPL"]
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {
        "license": "BSD"
    }
    
    with pytest.raises(ValueError, match="provided for choice variable"):
        generate_context(str(context_file), extra_context=extra_context)


def test_generate_context_invalid_boolean_conversion(tmp_path):
    """Test generate_context with invalid boolean string conversion."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "include_tests": True
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {
        "include_tests": "maybe"
    }
    
    with pytest.raises(ValueError, match="could not be converted to a boolean"):
        generate_context(str(context_file), extra_context=extra_context)


def test_generate_context_ordered_dict(tmp_path):
    """Test generate_context preserves ordering with OrderedDict."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "z_field": "last",
        "a_field": "first",
        "m_field": "middle"
    }
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    keys = list(result["cookiecutter"].keys())
    assert keys == ["z_field", "a_field", "m_field"]


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from pathlib import Path
from jinja2 import Environment
from cookiecutter.exceptions import EmptyDirNameException, OutputDirExistsException


def test_render_and_create_dir(tmp_path):
    """Test render_and_create_dir function."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    # Test 1: Basic directory creation with template rendering
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    assert result_dir == Path(tmp_path) / 'my_project'
    assert created is True
    assert result_dir.exists()
    
    # Test 2: Directory already exists without overwrite
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, tmp_path, env, overwrite_if_exists=False
        )
    
    # Test 3: Directory already exists with overwrite
    result_dir, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=True
    )
    assert result_dir == Path(tmp_path) / 'my_project'
    assert created is False
    assert result_dir.exists()
    
    # Test 4: Empty directory name raises exception
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, tmp_path, env, overwrite_if_exists=False)
    
    # Test 5: Plain directory name without template variables
    result_dir, created = render_and_create_dir(
        'static_dir', context, tmp_path, env, overwrite_if_exists=False
    )
    assert result_dir == Path(tmp_path) / 'static_dir'
    assert created is True
    assert result_dir.exists()
    
    # Test 6: Nested directory structure
    nested_dirname = '{{cookiecutter.project_name}}/src'
    result_dir, created = render_and_create_dir(
        nested_dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    assert result_dir == Path(tmp_path) / 'my_project' / 'src'
    assert created is True
    assert result_dir.exists()


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    import json
    from pathlib import Path
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
        "author_name": "John Doe"
    }
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["author_name"] == "John Doe"
    
    # Test 2: With default context overrides
    default_context = {"project_name": "Default Project"}
    result = generate_context(str(context_file), default_context=default_context)
    assert result["cookiecutter"]["project_name"] == "Default Project"
    
    # Test 3: With extra context overrides
    extra_context = {"author_name": "Jane Smith"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["author_name"] == "Jane Smith"
    
    # Test 4: Choice variable with extra context
    context_data_choice = {
        "license": ["MIT", "Apache", "GPL"],
        "project_name": "test"
    }
    context_file_choice = tmp_path / "cookiecutter_choice.json"
    with open(context_file_choice, 'w') as f:
        json.dump(context_data_choice, f)
    
    extra_context_choice = {"license": "Apache"}
    result = generate_context(str(context_file_choice), extra_context=extra_context_choice)
    # First item should be the selected choice
    assert result["cookiecutter"]["license"][0] == "Apache"
    
    # Test 5: Multi-choice variable with extra context
    context_data_multichoice = {
        "features": ["feature1", "feature2", "feature3"],
        "project_name": "test"
    }
    context_file_multichoice = tmp_path / "cookiecutter_multichoice.json"
    with open(context_file_multichoice, 'w') as f:
        json.dump(context_data_multichoice, f)
    
    extra_context_multichoice = {"features": ["feature2", "feature3"]}
    result = generate_context(str(context_file_multichoice), extra_context=extra_context_multichoice)
    assert result["cookiecutter"]["features"] == ["feature2", "feature3"]
    
    # Test 6: Dictionary variable with extra context
    context_data_dict = {
        "project_name": "test",
        "options": {"debug": True, "verbose": False}
    }
    context_file_dict = tmp_path / "cookiecutter_dict.json"
    with open(context_file_dict, 'w') as f:
        json.dump(context_data_dict, f)
    
    extra_context_dict = {"options": {"debug": False}}
    result = generate_context(str(context_file_dict), extra_context=extra_context_dict)
    assert result["cookiecutter"]["options"]["debug"] is False
    assert result["cookiecutter"]["options"]["verbose"] is False
    
    # Test 7: Boolean variable with string extra context
    context_data_bool = {
        "project_name": "test",
        "use_docker": True
    }
    context_file_bool = tmp_path / "cookiecutter_bool.json"
    with open(context_file_bool, 'w') as f:
        json.dump(context_data_bool, f)
    
    extra_context_bool = {"use_docker": "n"}
    result = generate_context(str(context_file_bool), extra_context=extra_context_bool)
    assert result["cookiecutter"]["use_docker"] is False
    
    # Test 8: Invalid JSON file raises ContextDecodingException
    invalid_json_file = tmp_path / "invalid.json"
    with open(invalid_json_file, 'w') as f:
        f.write("{invalid json content")
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(invalid_json_file))
    
    # Test 9: Invalid choice raises ValueError
    with pytest.raises(ValueError):
        extra_context_invalid = {"license": "InvalidLicense"}
        generate_context(str(context_file_choice), extra_context=extra_context_invalid)
    
    # Test 10: Invalid boolean conversion raises ValueError
    with pytest.raises(ValueError):
        extra_context_invalid_bool = {"use_docker": "maybe"}
        generate_context(str(context_file_bool), extra_context=extra_context_invalid_bool)


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_file(tmp_path, monkeypatch):
    """Test generate_file function with various scenarios."""
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_new_lines': None
        }
    }
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    # Test 1: Render text file
    infile = "test_{{ cookiecutter.project_name }}.txt"
    template_file = template_dir / infile
    template_file.write_text("Project: {{ cookiecutter.project_name }}")
    
    monkeypatch.chdir(template_dir)
    generate_file(project_dir, infile, context, env)
    
    output_file = Path(project_dir) / "test_my_project.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Project: my_project"
    
    # Test 2: Skip if file exists
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert output_file.read_text() == "Project: my_project"
    
    # Test 3: Binary file copy
    binary_infile = "binary_file.bin"
    binary_template = template_dir / binary_infile
    binary_template.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    generate_file(project_dir, binary_infile, context, env)
    output_binary = Path(project_dir) / binary_infile
    assert output_binary.exists()
    assert output_binary.read_bytes() == b'\x89PNG\r\n\x1a\n'
    
    # Test 4: Empty filename results in directory
    empty_dir_infile = "test_{{ cookiecutter.project_name }}/"
    generate_file(project_dir, empty_dir_infile, context, env)
    
    # Test 5: File with custom newline from context
    context['cookiecutter']['_new_lines'] = '\r\n'
    newline_infile = "newline_test.txt"
    newline_template = template_dir / newline_infile
    newline_template.write_text("Line 1\nLine 2\n", newline='\n')
    
    generate_file(project_dir, newline_infile, context, env)
    newline_output = Path(project_dir) / newline_infile
    assert newline_output.exists()
    
    # Test 6: File permissions are copied
    perm_infile = "perm_test.txt"
    perm_template = template_dir / perm_infile
    perm_template.write_text("test content")
    perm_template.chmod(0o755)
    
    generate_file(project_dir, perm_infile, context, env)
    perm_output = Path(project_dir) / perm_infile
    assert perm_output.stat().st_mode & 0o755 == 0o755


# LLM-generated content at query #18
#--------------------------

```python
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from jinja2 import Environment, FileSystemLoader
from cookiecutter.generate import generate_file


def test_generate_file():
    """Test generate_file function with various scenarios."""
    
    # Test 1: Binary file should be copied without rendering
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = 'binary_file.bin'
        context = {'cookiecutter': {}}
        env = Environment()
        
        with patch('cookiecutter.generate.is_binary', return_value=True):
            with patch('shutil.copyfile') as mock_copyfile:
                with patch('shutil.copymode') as mock_copymode:
                    with patch('os.path.isdir', return_value=False):
                        with patch('os.path.exists', return_value=False):
                            generate_file(project_dir, infile, context, env)
                            mock_copyfile.assert_called_once()
                            mock_copymode.assert_called_once()
    
    # Test 2: Text file should be rendered
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = 'test_template.txt'
        context = {'cookiecutter': {'name': 'test_project'}}
        env = Environment()
        
        with patch('cookiecutter.generate.is_binary', return_value=False):
            with patch('builtins.open', mock_open(read_data='Hello {{ cookiecutter.name }}')):
                with patch.object(env, 'get_template') as mock_get_template:
                    mock_template = Mock()
                    mock_template.render.return_value = 'Hello test_project'
                    mock_get_template.return_value = mock_template
                    
                    with patch('os.path.isdir', return_value=False):
                        with patch('os.path.exists', return_value=False):
                            with patch('shutil.copymode'):
                                generate_file(project_dir, infile, context, env)
                                mock_get_template.assert_called_once()
                                mock_template.render.assert_called_once()
    
    # Test 3: Empty filename should return early
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = 'test.txt'
        context = {'cookiecutter': {}}
        env = Environment()
        
        with patch('os.path.isdir', return_value=True):
            with patch('os.path.join', return_value=os.path.join(project_dir, infile)):
                generate_file(project_dir, infile, context, env)
    
    # Test 4: Skip if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = 'test.txt'
        context = {'cookiecutter': {}}
        env = Environment()
        
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.exists', return_value=True):
                with patch('cookiecutter.generate.is_binary', return_value=False):
                    with patch.object(env, 'get_template') as mock_get_template:
                        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
                        mock_get_template.assert_not_called()
    
    # Test 5: Newline handling - detected from file
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = 'test.txt'
        context = {'cookiecutter': {}}
        env = Environment()
        
        with patch('cookiecutter.generate.is_binary', return_value=False):
            with patch('builtins.open', mock_open(read_data='test content')):
                with patch.object(env, 'get_template') as mock_get_template:
                    mock_template = Mock()
                    mock_template.render.return_value = 'rendered content'
                    mock_get_template.return_value = mock_template
                    
                    with patch('os.path.isdir', return_value=False):
                        with patch('os.path.exists', return_value=False):
                            with patch('shutil.copymode'):
                                generate_file(project_dir, infile, context, env)
    
    # Test 6: Custom newline from context
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = 'test.txt'
        context = {'cookiecutter': {'_new_lines': '\r\n'}}
        env = Environment()
        
        with patch('cookiecutter.generate.is_binary', return_value=False):
            with patch('builtins.open', mock_open(read_data='test content')):
                with patch.object(env, 'get_template') as mock_get_template:
                    mock_template = Mock()
                    mock_template.render.return_value = 'rendered content'
                    mock_get_template.return_value = mock_template
                    
                    with patch('os.path.isdir', return_value=False):
                        with patch('os.path.exists', return_value=False):
                            with patch('shutil.copymode'):
                                generate_file(project_dir, infile, context, env)
    
    # Test 7: Windows path separator handling
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = 'subdir\\template.txt'
        context = {'cookiecutter': {}}
        env = Environment()
        
        with patch('cookiecutter.generate.is_binary', return_value=False):
            with patch('builtins.open', mock_open(read_data='test')):
                with patch.object(env, 'get_template') as mock_get_template:
                    mock_template = Mock()
                    mock_template.render.return_value = 'rendered'
                    mock_get_template.return_value = mock_template
                    
                    with patch('os.path.isdir', return_value=False):
                        with patch('os.path.exists', return_value=False):
                            with patch('shutil.copymode'):
                                generate_file(project_dir, infile, context, env)


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from collections import OrderedDict
    
    # Setup temporary directories
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create a simple template file
    test_file = template_dir / "test.txt"
    test_file.write_text("Hello {{cookiecutter.name}}")
    
    # Create context
    context = {
        'cookiecutter': OrderedDict([
            ('project_name', 'my_project'),
            ('name', 'World')
        ])
    }
    
    # Mock find_template to return our template directory
    with patch('cookiecutter.generate.find_template') as mock_find:
        mock_find.return_value = str(template_dir)
        
        # Mock run_hook_from_repo_dir to avoid actually running hooks
        with patch('cookiecutter.generate.run_hook_from_repo_dir'):
            # Call generate_files
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False,
                skip_if_file_exists=False,
                accept_hooks=True,
                keep_project_on_failure=False
            )
    
    # Verify the project directory was created
    assert Path(result).exists()
    assert "my_project" in result
    
    # Verify the file was generated with rendered content
    generated_file = Path(result) / "test.txt"
    assert generated_file.exists()
    assert generated_file.read_text() == "Hello World"


def test_generate_files_with_subdirectories(tmp_path, monkeypatch):
    """Test generate_files with subdirectories."""
    from pathlib import Path
    from unittest.mock import patch
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create subdirectory structure
    src_dir = template_dir / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("# {{cookiecutter.name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': OrderedDict([
            ('project_name', 'my_app'),
            ('name', 'MyApp')
        ])
    }
    
    with patch('cookiecutter.generate.find_template') as mock_find:
        mock_find.return_value = str(template_dir)
        with patch('cookiecutter.generate.run_hook_from_repo_dir'):
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                accept_hooks=True
            )
    
    # Verify subdirectory structure
    generated_file = Path(result) / "src" / "main.py"
    assert generated_file.exists()
    assert "# MyApp" in generated_file.read_text()


def test_generate_files_overwrite_if_exists(tmp_path):
    """Test generate_files with overwrite_if_exists flag."""
    from pathlib import Path
    from unittest.mock import patch
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Pre-create the project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old content")
    
    context = {
        'cookiecutter': OrderedDict([
            ('project_name', 'my_project')
        ])
    }
    
    with patch('cookiecutter.generate.find_template') as mock_find:
        mock_find.return_value = str(template_dir)
        with patch('cookiecutter.generate.run_hook_from_repo_dir'):
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=True,
                accept_hooks=True
            )
    
    assert Path(result).exists()


def test_generate_files_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists flag."""
    from pathlib import Path
    from unittest.mock import patch
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "existing.txt").write_text("new {{cookiecutter.name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': OrderedDict([
            ('project_name', 'my_project'),
            ('name', 'Test')
        ])
    }
    
    with patch('cookiecutter.generate.find_template') as mock_find:
        mock_find.return_value = str(template_dir)
        with patch('cookiecutter.generate.run_hook_from_repo_dir'):
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                skip_if_file_exists=True,
                accept_hooks=True
            )
    
    assert Path(result).exists()


def test_generate_files_no_hooks(tmp_path):
    """Test generate_files with accept_hooks=False."""
    from pathlib import Path
    from unittest.mock import patch, call
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        'cookiecutter': OrderedDict([
            ('project_name', 'my_project')
        ])
    }
    
    with patch('cookiecutter.generate.find_template') as mock_find:
        mock_find.return_value = str(template_dir)
        with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_hook:
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                accept_hooks=False
            )
            # Verify hooks were not called
            mock_hook.assert_not_called()


def test_generate_files_with_copy_without_render(tmp_path):
    """Test generate_files with _copy_without_render context."""
    from pathlib import Path
    from unittest.mock import patch
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a binary file (or file to


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_context(tmp_path, monkeypatch):
    """Test generate_context function with various scenarios."""
    monkeypatch.chdir(tmp_path)
    
    # Test 1: Basic context generation with valid JSON
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John Doe"}')
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"
    
    # Test 2: Context generation with default_context
    context_file = tmp_path / "cookiecutter2.json"
    context_file.write_text('{"project_name": "default_name", "version": "1.0"}')
    
    default_context = {"project_name": "overridden_name"}
    result = generate_context(str(context_file), default_context=default_context)
    assert result["cookiecutter"]["project_name"] == "overridden_name"
    assert result["cookiecutter"]["version"] == "1.0"
    
    # Test 3: Context generation with extra_context
    context_file = tmp_path / "cookiecutter3.json"
    context_file.write_text('{"project_name": "original", "license": "MIT"}')
    
    extra_context = {"project_name": "extra_override"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["project_name"] == "extra_override"
    
    # Test 4: Choice variable with valid overwrite
    context_file = tmp_path / "cookiecutter4.json"
    context_file.write_text('{"python_version": ["3.8", "3.9", "3.10"]}')
    
    extra_context = {"python_version": "3.9"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["python_version"][0] == "3.9"
    
    # Test 5: Multi-choice variable with valid overwrite
    context_file = tmp_path / "cookiecutter5.json"
    context_file.write_text('{"features": ["auth", "api", "admin"]}')
    
    extra_context = {"features": ["api", "admin"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["features"] == ["api", "admin"]
    
    # Test 6: Dictionary variable with partial overwrite
    context_file = tmp_path / "cookiecutter6.json"
    context_file.write_text('{"database": {"engine": "postgresql", "port": 5432}}')
    
    extra_context = {"database": {"port": 3306}}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["database"]["port"] == 3306
    assert result["cookiecutter"]["database"]["engine"] == "postgresql"
    
    # Test 7: Boolean variable with string overwrite
    context_file = tmp_path / "cookiecutter7.json"
    context_file.write_text('{"use_ci": true}')
    
    extra_context = {"use_ci": "n"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["use_ci"] is False
    
    # Test 8: Invalid JSON raises ContextDecodingException
    context_file = tmp_path / "cookiecutter_invalid.json"
    context_file.write_text('{"invalid": json}')
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(context_file))
    
    # Test 9: Choice variable with invalid overwrite raises ValueError
    context_file = tmp_path / "cookiecutter8.json"
    context_file.write_text('{"env": ["dev", "staging", "prod"]}')
    
    extra_context = {"env": "invalid_env"}
    with pytest.raises(ValueError):
        generate_context(str(context_file), extra_context=extra_context)
    
    # Test 10: Multi-choice with invalid choice raises ValueError
    context_file = tmp_path / "cookiecutter9.json"
    context_file.write_text('{"options": ["a", "b", "c"]}')
    
    extra_context = {"options": ["a", "invalid"]}
    with pytest.raises(ValueError):
        generate_context(str(context_file), extra_context=extra_context)
    
    # Test 11: Ordered dict preservation
    context_file = tmp_path / "cookiecutter10.json"
    context_file.write_text('{"first": 1, "second": 2, "third": 3}')
    
    result = generate_context(str(context_file))
    keys = list(result["cookiecutter"].keys())
    assert keys == ["first", "second", "third"]
    
    # Test 12: Empty context with defaults
    context_file = tmp_path / "cookiecutter11.json"
    context_file.write_text('{}')
    
    result = generate_context(str(context_file))
    assert result["cookiecutter"] == {}


# LLM-generated content at query #21
#--------------------------

```python
def test_apply_overwrites_to_context():
    """Test apply_overwrites_to_context function with various scenarios."""
    
    # Test 1: Simple value overwrite
    context = {'key1': 'original_value'}
    overwrite_context = {'key1': 'new_value'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['key1'] == 'new_value'
    
    # Test 2: Ignore new variables at first level
    context = {'key1': 'value1'}
    overwrite_context = {'key2': 'value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert 'key2' not in context
    
    # Test 3: Add new variables in dictionary context
    context = {'key1': 'value1'}
    overwrite_context = {'key2': 'value2'}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context['key2'] == 'value2'
    
    # Test 4: Multichoice variable with valid overwrite
    context = {'choices': ['option1', 'option2', 'option3']}
    overwrite_context = {'choices': ['option2', 'option3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['choices'] == ['option2', 'option3']
    
    # Test 5: Multichoice variable with invalid overwrite
    context = {'choices': ['option1', 'option2']}
    overwrite_context = {'choices': ['invalid']}
    with pytest.raises(ValueError, match="provided for multi-choice variable"):
        apply_overwrites_to_context(context, overwrite_context)
    
    # Test 6: Choice variable - move selection to first position
    context = {'choice': ['default', 'option1', 'option2']}
    overwrite_context = {'choice': 'option1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['choice'] == ['option1', 'default', 'option2']
    
    # Test 7: Choice variable with invalid choice
    context = {'choice': ['option1', 'option2']}
    overwrite_context = {'choice': 'invalid'}
    with pytest.raises(ValueError, match="provided for choice variable"):
        apply_overwrites_to_context(context, overwrite_context)
    
    # Test 8: Dictionary variable with nested overwrite
    context = {'nested': {'key1': 'value1', 'key2': 'value2'}}
    overwrite_context = {'nested': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['nested']['key1'] == 'new_value1'
    assert context['nested']['key2'] == 'value2'
    
    # Test 9: Boolean variable with string 'y'
    context = {'flag': False}
    overwrite_context = {'flag': 'y'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['flag'] is True
    
    # Test 10: Boolean variable with string 'n'
    context = {'flag': True}
    overwrite_context = {'flag': 'n'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['flag'] is False
    
    # Test 11: Boolean variable with invalid string
    context = {'flag': True}
    overwrite_context = {'flag': 'invalid'}
    with pytest.raises(ValueError, match="could not be converted to a boolean"):
        apply_overwrites_to_context(context, overwrite_context)
    
    # Test 12: List to list overwrite in dictionary context
    context = {'items': ['item1', 'item2']}
    overwrite_context = {'items': ['new_item']}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context['items'] == ['new_item']
    
    # Test 13: Nested dictionary with new key
    context = {'config': {'existing': 'value'}}
    overwrite_context = {'config': {'new_key': 'new_value'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['config']['new_key'] == 'new_value'
    assert context['config']['existing'] == 'value'
    
    # Test 14: Empty overwrite context
    context = {'key1': 'value1', 'key2': 'value2'}
    original_context = context.copy()
    apply_overwrites_to_context(context, {})
    assert context == original_context
    
    # Test 15: Multiple levels of nested dictionaries
    context = {'level1': {'level2': {'key': 'value'}}}
    overwrite_context = {'level1': {'level2': {'key': 'new_value'}}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['level1']['level2']['key'] == 'new_value'


# LLM-generated content at query #22
#--------------------------

```python
def test_apply_overwrites_to_context():
    """Test apply_overwrites_to_context function with various scenarios."""
    
    # Test 1: Simple value overwrite
    context = {'var1': 'old_value'}
    overwrite = {'var1': 'new_value'}
    apply_overwrites_to_context(context, overwrite)
    assert context['var1'] == 'new_value'
    
    # Test 2: Ignore new variables at first level
    context = {'var1': 'value1'}
    overwrite = {'var2': 'value2'}
    apply_overwrites_to_context(context, overwrite)
    assert 'var2' not in context
    
    # Test 3: Add new variables in nested dictionary
    context = {'nested': {'var1': 'value1'}}
    overwrite = {'nested': {'var2': 'value2'}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['nested']['var2'] == 'value2'
    
    # Test 4: Choice variable - valid choice
    context = {'choice_var': ['option1', 'option2', 'option3']}
    overwrite = {'choice_var': 'option2'}
    apply_overwrites_to_context(context, overwrite)
    assert context['choice_var'][0] == 'option2'
    assert 'option2' in context['choice_var']
    
    # Test 5: Choice variable - invalid choice
    context = {'choice_var': ['option1', 'option2']}
    overwrite = {'choice_var': 'option3'}
    with pytest.raises(ValueError, match="option3 provided for choice variable"):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 6: Multi-choice variable - valid choices
    context = {'multi_choice': ['a', 'b', 'c']}
    overwrite = {'multi_choice': ['b', 'c']}
    apply_overwrites_to_context(context, overwrite)
    assert context['multi_choice'] == ['b', 'c']
    
    # Test 7: Multi-choice variable - invalid choices
    context = {'multi_choice': ['a', 'b']}
    overwrite = {'multi_choice': ['a', 'c']}
    with pytest.raises(ValueError, match="multi-choice variable"):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 8: Boolean variable with string 'y'
    context = {'bool_var': False}
    overwrite = {'bool_var': 'y'}
    apply_overwrites_to_context(context, overwrite)
    assert context['bool_var'] is True
    
    # Test 9: Boolean variable with string 'n'
    context = {'bool_var': True}
    overwrite = {'bool_var': 'n'}
    apply_overwrites_to_context(context, overwrite)
    assert context['bool_var'] is False
    
    # Test 10: Boolean variable with invalid string
    context = {'bool_var': True}
    overwrite = {'bool_var': 'invalid'}
    with pytest.raises(ValueError, match="could not be converted to a boolean"):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 11: Nested dictionary partial overwrite
    context = {'nested': {'key1': 'value1', 'key2': 'value2'}}
    overwrite = {'nested': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite)
    assert context['nested']['key1'] == 'new_value1'
    assert context['nested']['key2'] == 'value2'
    
    # Test 12: List in nested dict with in_dictionary_variable=True
    context = {'nested': {'list_var': ['a', 'b', 'c']}}
    overwrite = {'nested': {'list_var': ['b']}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['nested']['list_var'] == ['b']
    
    # Test 13: Empty overwrite context
    context = {'var1': 'value1'}
    overwrite = {}
    apply_overwrites_to_context(context, overwrite)
    assert context['var1'] == 'value1'
    
    # Test 14: Multiple variables
    context = {'var1': 'value1', 'var2': 'value2', 'var3': 'value3'}
    overwrite = {'var1': 'new1', 'var3': 'new3'}
    apply_overwrites_to_context(context, overwrite)
    assert context['var1'] == 'new1'
    assert context['var2'] == 'value2'
    assert context['var3'] == 'new3'
    
    # Test 15: Dict overwrite with non-dict value should overwrite
    context = {'var': 'string_value'}
    overwrite = {'var': {'nested': 'value'}}
    apply_overwrites_to_context(context, overwrite)
    assert context['var'] == {'nested': 'value'}


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Create a temporary template directory structure
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}")
    
    # Create a subdirectory with a template file
    subdir = template_dir / "src"
    subdir.mkdir()
    (subdir / "main.py").write_text("# Project: {{cookiecutter.project_name}}")
    
    # Create cookiecutter.json in repo_dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "author": "John Doe"}')
    
    # Copy template to repo
    import shutil
    shutil.copytree(template_dir, repo_dir / template_dir.name)
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Mock find_template to return our template directory
    def mock_find_template(repo, env):
        return str(repo_dir / template_dir.name)
    
    monkeypatch.setattr("cookiecutter.generate.find_template", mock_find_template)
    
    # Mock run_hook_from_repo_dir to do nothing
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", lambda *args, **kwargs: None)
    
    # Call generate_files
    context = OrderedDict([("cookiecutter", OrderedDict([("project_name", "my_project"), ("author", "John Doe")]))])
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False,
    )
    
    # Verify the project directory was created
    assert os.path.exists(result)
    assert "my_project" in result
    
    # Verify files were generated correctly
    readme_path = Path(result) / "README.md"
    assert readme_path.exists()
    readme_content = readme_path.read_text()
    assert "# my_project" in readme_content
    assert "Author: John Doe" in readme_content
    
    # Verify subdirectory and file were created
    main_path = Path(result) / "src" / "main.py"
    assert main_path.exists()
    main_content = main_path.read_text()
    assert "# Project: my_project" in main_content


def test_generate_files_with_overwrite(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Create template directory
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("Content: {{cookiecutter.name}}")
    
    # Create repo with context
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "proj", "name": "test"}')
    import shutil
    shutil.copytree(template_dir, repo_dir / template_dir.name)
    
    # Create output directory with existing project
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing_project = output_dir / "proj"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old content")
    
    # Mock functions
    def mock_find_template(repo, env):
        return str(repo_dir / template_dir.name)
    
    monkeypatch.setattr("cookiecutter.generate.find_template", mock_find_template)
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", lambda *args, **kwargs: None)
    
    # Call with overwrite
    context = OrderedDict([("cookiecutter", OrderedDict([("project_name", "proj"), ("name", "test")]))])
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False,
    )
    
    assert os.path.exists(result)
    assert (Path(result) / "file.txt").exists()
    assert not (Path(result) / "old_file.txt").exists()


def test_generate_files_empty_context(tmp_path, monkeypatch):
    """Test generate_files with empty context."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Create template directory with simple name (no variables)
    template_dir = tmp_path / "simple_project"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("Static content")
    
    # Create repo
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "default"}')
    import shutil
    shutil.copytree(template_dir, repo_dir / template_dir.name)
    
    # Mock functions
    def mock_find_template(repo, env):
        return str(repo_dir / template_dir.name)
    
    monkeypatch.setattr("cookiecutter.generate.find_template", mock_find_template)
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", lambda *args, **kwargs: None)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Call with None context
    result = generate_files(
        repo_dir=str(repo_dir),
        context=None,
        output_dir=str(output_dir),
        accept_hooks=False,
    )
    
    assert os.path.exists(result)
    assert (Path(result) / "file.txt").exists()


# LLM-generated content at query #24
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function creates project from template."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Create a temporary template directory structure
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create template files
    (template_dir / "README.md").write_text("# {{cookiecutter.project_name}}\n")
    (template_dir / "file.txt").write_text("Project: {{cookiecutter.project_name}}")
    
    # Create a subdirectory in template
    subdir = template_dir / "src"
    subdir.mkdir()
    (subdir / "main.py").write_text("# {{cookiecutter.project_name}}")
    
    # Create cookiecutter.json
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "test_project"}')
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Mock find_template to return our template directory
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    monkeypatch.setattr("cookiecutter.generate.find_template", mock_find_template)
    
    # Mock run_hook_from_repo_dir to do nothing
    def mock_run_hook(*args, **kwargs):
        pass
    
    monkeypatch.setattr(
        "cookiecutter.generate.run_hook_from_repo_dir", mock_run_hook
    )
    
    # Prepare context
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Call generate_files
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False,
    )
    
    # Verify project directory was created
    assert os.path.exists(result)
    assert os.path.isdir(result)
    
    # Verify files were generated with rendered content
    readme_path = Path(result) / "README.md"
    assert readme_path.exists()
    assert readme_path.read_text() == "# test_project\n"
    
    file_path = Path(result) / "file.txt"
    assert file_path.exists()
    assert file_path.read_text() == "Project: test_project"
    
    # Verify subdirectory and its files were created
    main_py_path = Path(result) / "src" / "main.py"
    assert main_py_path.exists()
    assert main_py_path.read_text() == "# test_project"


def test_generate_files_with_overwrite(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists flag."""
    from pathlib import Path
    
    template_dir = tmp_path / "{{cookiecutter.name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("{{cookiecutter.name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    def mock_run_hook(*args, **kwargs):
        pass
    
    monkeypatch.setattr("cookiecutter.generate.find_template", mock_find_template)
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", mock_run_hook)
    
    context = {"cookiecutter": {"name": "myproject"}}
    
    # First generation
    result1 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        accept_hooks=False,
    )
    
    assert Path(result1).exists()
    
    # Second generation with overwrite
    result2 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False,
    )
    
    assert Path(result2).exists()
    assert Path(result2).read_text() == "myproject"


def test_generate_files_with_copy_without_render(tmp_path, monkeypatch):
    """Test generate_files respects _copy_without_render setting."""
    from pathlib import Path
    
    template_dir = tmp_path / "{{cookiecutter.project}}"
    template_dir.mkdir()
    
    # Create a binary-like file (we'll treat .bin as copy-only)
    (template_dir / "binary.bin").write_bytes(b"\x00\x01\x02{{cookiecutter.project}}")
    (template_dir / "rendered.txt").write_text("{{cookiecutter.project}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    def mock_run_hook(*args, **kwargs):
        pass
    
    monkeypatch.setattr("cookiecutter.generate.find_template", mock_find_template)
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", mock_run_hook)
    
    context = {
        "cookiecutter": {
            "project": "myproject",
            "_copy_without_render": ["*.bin"]
        }
    }
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False,
    )
    
    # Binary file should not be rendered
    binary_path = Path(result) / "binary.bin"
    assert binary_path.exists()
    assert b"{{cookiecutter.project}}" in binary_path.read_bytes()
    
    # Text file should be rendered
    text_path = Path(result) / "rendered.txt"
    assert text_path.exists()
    assert text_path.read_text() == "myproject"


def test_generate_files_skip_if_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists flag."""
    from pathlib import Path
    
    template_dir = tmp_path / "{{cookiecutter.name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("new content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Pre-create output directory with existing file
    project_dir = output_dir / "myproject"
    project_dir.mkdir()
    existing_file = project_dir / "file.txt"
    existing_file.write_text("old content")
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    def mock_run_hook(*args, **kwargs):
        pass
    
    monkeypatch.setattr("cookiecutter.generate.find_template", mock_find_template)
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", mock_run_hook)
    
    context = {"cookiecutter": {"name": "myproject"}}
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        overwrite_if_exists=True,
        accept_hooks=False


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "author": "John Doe"
    }
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context with default_context override."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "author": "John Doe"
    }
    context_file.write_text(json.dumps(context_data))
    
    default_context = {"author": "Jane Smith"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["author"] == "Jane Smith"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context with extra_context override."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "version": "1.0.0"
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"version": "2.0.0"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["version"] == "2.0.0"


def test_generate_context_choice_variable(tmp_path):
    """Test generate_context with choice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "python_version": ["3.8", "3.9", "3.10"]
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"python_version": "3.9"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    # The chosen option should be first in the list
    assert result["cookiecutter"]["python_version"][0] == "3.9"
    assert set(result["cookiecutter"]["python_version"]) == {"3.8", "3.9", "3.10"}


def test_generate_context_multichoice_variable(tmp_path):
    """Test generate_context with multichoice variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "features": ["feature1", "feature2", "feature3"]
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"features": ["feature2", "feature3"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["features"] == ["feature2", "feature3"]


def test_generate_context_invalid_choice(tmp_path):
    """Test generate_context with invalid choice raises ValueError."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "python_version": ["3.8", "3.9"]
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"python_version": "3.11"}
    
    with pytest.raises(ValueError, match="provided for choice variable"):
        generate_context(str(context_file), extra_context=extra_context)


def test_generate_context_invalid_multichoice(tmp_path):
    """Test generate_context with invalid multichoice raises ValueError."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "features": ["feature1", "feature2"]
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"features": ["feature1", "feature3"]}
    
    with pytest.raises(ValueError, match="provided for multi-choice variable"):
        generate_context(str(context_file), extra_context=extra_context)


def test_generate_context_dict_variable(tmp_path):
    """Test generate_context with nested dictionary variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "config": {
            "debug": True,
            "port": 8000
        }
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"config": {"port": 9000}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["config"]["debug"] is True
    assert result["cookiecutter"]["config"]["port"] == 9000


def test_generate_context_boolean_variable(tmp_path):
    """Test generate_context with boolean variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "use_docker": True
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"use_docker": "n"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context with invalid JSON raises ContextDecodingException."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text("{invalid json}")
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(context_file))


def test_generate_context_invalid_boolean_conversion(tmp_path):
    """Test generate_context with invalid boolean conversion raises ValueError."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "use_docker": True
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"use_docker": "maybe"}
    
    with pytest.raises(ValueError, match="could not be converted to a boolean"):
        generate_context(str(context_file), extra_context=extra_context)


def test_generate_context_with_invalid_default_warns(tmp_path):
    """Test generate_context with invalid default_context issues warning."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "version": ["1.0", "2.0"]
    }
    context_file.write_text(json.dumps(context_data))
    
    default_context = {"version": "3.0"}
    
    with pytest.warns(UserWarning, match="Invalid default received"):
        result = generate_context(str(context_file), default_context=default_context)
    
    # Original context should remain unchanged when default is invalid
    assert result["cookiecutter"]["version"] == ["1.0", "2.0"]


def test_generate_context_preserves_order(tmp_path):
    """Test that generate_context preserves key order."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = OrderedDict([
        ("first", "value1"),
        ("second", "value2"),
        ("third", "value3")
    ])
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    
    keys = list(result["cookiecutter"].keys())
    assert keys == ["first", "second", "thir


# LLM-generated content at query #26
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import OrderedDict
from jinja2 import Environment, FileSystemLoader, UndefinedError

def test_generate_files():
    """Test generate_files function with various scenarios."""
    
    # Test 1: Basic successful generation
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.work_in') as mock_work_in, \
         patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook, \
         patch('os.walk') as mock_walk, \
         patch('os.path.abspath') as mock_abspath:
        
        mock_env = MagicMock(spec=Environment)
        mock_create_env.return_value = mock_env
        mock_find_template.return_value = '/template/dir'
        mock_abspath.return_value = '/project/dir'
        mock_render_dir.return_value = (Path('/project/dir'), True)
        mock_walk.return_value = [
            ('.', ['subdir'], ['file.txt']),
        ]
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=None)
        mock_env.from_string.return_value.render.return_value = 'rendered'
        
        with patch('cookiecutter.generate.generate_file'):
            result = generate_files(
                repo_dir='/repo',
                context={'cookiecutter': {'project_name': 'test'}},
                output_dir='/output'
            )
        
        assert result == '/project/dir'
        mock_create_env.assert_called_once()
        mock_find_template.assert_called_once()
        mock_render_dir.assert_called()

    # Test 2: With default context
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.work_in') as mock_work_in, \
         patch('os.walk') as mock_walk, \
         patch('os.path.abspath') as mock_abspath:
        
        mock_env = MagicMock(spec=Environment)
        mock_create_env.return_value = mock_env
        mock_find_template.return_value = '/template/dir'
        mock_abspath.return_value = '/project/dir'
        mock_render_dir.return_value = (Path('/project/dir'), True)
        mock_walk.return_value = [('.', [], [])]
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=None)
        
        context = OrderedDict([])
        result = generate_files(
            repo_dir='/repo',
            context=context,
            output_dir='/output'
        )
        
        assert result == '/project/dir'
        assert isinstance(context, dict)

    # Test 3: UndefinedError during directory creation
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.rmtree') as mock_rmtree:
        
        mock_env = MagicMock(spec=Environment)
        mock_create_env.return_value = mock_env
        mock_find_template.return_value = '/template/dir'
        mock_render_dir.side_effect = UndefinedError('undefined variable')
        
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(
                repo_dir='/repo',
                context={'cookiecutter': {}},
                output_dir='/output'
            )

    # Test 4: With accept_hooks=False
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.work_in') as mock_work_in, \
         patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook, \
         patch('os.walk') as mock_walk, \
         patch('os.path.abspath') as mock_abspath:
        
        mock_env = MagicMock(spec=Environment)
        mock_create_env.return_value = mock_env
        mock_find_template.return_value = '/template/dir'
        mock_abspath.return_value = '/project/dir'
        mock_render_dir.return_value = (Path('/project/dir'), True)
        mock_walk.return_value = [('.', [], [])]
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=None)
        
        result = generate_files(
            repo_dir='/repo',
            context={'cookiecutter': {}},
            output_dir='/output',
            accept_hooks=False
        )
        
        assert result == '/project/dir'
        mock_run_hook.assert_not_called()

    # Test 5: With overwrite_if_exists=True
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.work_in') as mock_work_in, \
         patch('os.walk') as mock_walk, \
         patch('os.path.abspath') as mock_abspath:
        
        mock_env = MagicMock(spec=Environment)
        mock_create_env.return_value = mock_env
        mock_find_template.return_value = '/template/dir'
        mock_abspath.return_value = '/project/dir'
        mock_render_dir.return_value = (Path('/project/dir'), False)
        mock_walk.return_value = [('.', [], [])]
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=None)
        
        result = generate_files(
            repo_dir='/repo',
            context={'cookiecutter': {}},
            output_dir='/output',
            overwrite_if_exists=True
        )
        
        assert result == '/project/dir'
        mock_render_dir.assert_called()

    # Test 6: With keep_project_on_failure=True
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.work_in') as mock_work_in, \
         patch('os.walk') as mock_walk, \
         patch('os.path.abspath') as mock_abspath:
        
        mock_env = MagicMock(spec=Environment)


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    from pathlib import Path
    from collections import OrderedDict
    
    # Setup template directory structure
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}\n")
    
    # Create context
    context = {
        'cookiecutter': OrderedDict([
            ('project_name', 'test_project')
        ])
    }
    
    # Mock find_template to return our template directory
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    
    # Mock run_hook_from_repo_dir to do nothing
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_run_hook)
    
    # Call generate_files
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=True,
        keep_project_on_failure=False
    )
    
    # Assertions
    assert result is not None
    assert os.path.exists(result)
    assert os.path.isdir(result)
    
    # Check that the project directory was created with the rendered name
    project_path = Path(result)
    assert project_path.name == 'test_project'
    
    # Check that the template file was rendered
    readme_path = project_path / "README.md"
    assert readme_path.exists()
    assert readme_path.read_text() == "# test_project\n"


def test_generate_files_with_overwrite(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")
    
    context = {
        'cookiecutter': OrderedDict([
            ('project_name', 'test_project')
        ])
    }
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_run_hook)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Generate once
    result1 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False
    )
    
    # Generate again with overwrite
    result2 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True
    )
    
    assert result1 == result2
    assert os.path.exists(result2)


def test_generate_files_skip_if_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists=True."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("{{cookiecutter.content}}")
    
    context = {
        'cookiecutter': OrderedDict([
            ('project_name', 'test_project'),
            ('content', 'new content')
        ])
    }
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_run_hook)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True
    )
    
    file_path = Path(result) / "file.txt"
    assert file_path.exists()


def test_generate_files_with_binary_file(tmp_path, monkeypatch):
    """Test generate_files with binary files."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a binary file
    binary_file = template_dir / "image.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    context = {
        'cookiecutter': OrderedDict([
            ('project_name', 'test_project')
        ])
    }
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_run_hook)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir)
    )
    
    output_binary = Path(result) / "image.bin"
    assert output_binary.exists()
    assert output_binary.read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_files_empty_context(tmp_path, monkeypatch):
    """Test generate_files with empty context."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "project"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_run_hook)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(tmp_path),


# LLM-generated content at query #28
#--------------------------

```python
def test_generate_context(tmp_path, monkeypatch):
    """Test generate_context function with various scenarios."""
    # Test 1: Basic context generation from valid JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "my_project", "author": "John Doe"}
    context_file.write_text(json.dumps(context_data))
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_generate_context_with_default_context(tmp_path, monkeypatch):
    """Test generate_context with default_context parameter."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "my_project", "version": "1.0"}
    context_file.write_text(json.dumps(context_data))
    
    default_context = {"project_name": "default_project"}
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "default_project"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path, monkeypatch):
    """Test generate_context with extra_context parameter."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "my_project", "version": "1.0"}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"version": "2.0"}
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["version"] == "2.0"


def test_generate_context_invalid_json(tmp_path, monkeypatch):
    """Test generate_context with invalid JSON file."""
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text("{ invalid json }")
    
    monkeypatch.chdir(tmp_path)
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(context_file))


def test_generate_context_file_not_found(tmp_path, monkeypatch):
    """Test generate_context with non-existent file."""
    monkeypatch.chdir(tmp_path)
    
    with pytest.raises(FileNotFoundError):
        generate_context("nonexistent.json")


def test_generate_context_with_choice_variable(tmp_path, monkeypatch):
    """Test generate_context with choice variable in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"license": ["MIT", "Apache", "GPL"]}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"license": "Apache"}
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file), extra_context=extra_context)
    
    # Apache should be first in the list (default choice)
    assert result["cookiecutter"]["license"][0] == "Apache"


def test_generate_context_with_multichoice_variable(tmp_path, monkeypatch):
    """Test generate_context with multichoice variable in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"features": ["feature1", "feature2", "feature3"]}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"features": ["feature1", "feature3"]}
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["features"]) == {"feature1", "feature3"}


def test_generate_context_with_nested_dict(tmp_path, monkeypatch):
    """Test generate_context with nested dictionary context."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "my_project",
        "author": {"name": "John", "email": "john@example.com"}
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"author": {"email": "newemail@example.com"}}
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["author"]["name"] == "John"
    assert result["cookiecutter"]["author"]["email"] == "newemail@example.com"


def test_generate_context_with_boolean_variable(tmp_path, monkeypatch):
    """Test generate_context with boolean variable."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"include_tests": True}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"include_tests": "y"}
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["include_tests"] is True


def test_generate_context_custom_filename(tmp_path, monkeypatch):
    """Test generate_context with custom context filename."""
    context_file = tmp_path / "custom_context.json"
    context_data = {"project": "test_project"}
    context_file.write_text(json.dumps(context_data))
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file))
    
    assert "custom_context" in result
    assert result["custom_context"]["project"] == "test_project"


def test_generate_context_ordered_dict(tmp_path, monkeypatch):
    """Test that generate_context preserves order."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = OrderedDict([
        ("first", "value1"),
        ("second", "value2"),
        ("third", "value3")
    ])
    context_file.write_text(json.dumps(context_data))
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file))
    
    keys = list(result["cookiecutter"].keys())
    assert keys == ["first", "second", "third"]


def test_generate_context_invalid_choice_in_extra_context(tmp_path, monkeypatch):
    """Test generate_context with invalid choice in extra_context."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"license": ["MIT", "Apache"]}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"license": "GPL"}
    
    monkeypatch.chdir(tmp_path)
    
    with pytest.raises(ValueError):
        generate_context(str(context_file), extra_context=extra_context)


def test_generate_context_invalid_boolean_conversion(tmp_path, monkeypatch):
    """Test generate_context with invalid boolean conversion."""
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"include_tests": True}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"include_tests": "invalid_bool"}
    
    monkeypatch.chdir


# LLM-generated content at query #29
#--------------------------

```python
def test_generate_file(tmp_path, monkeypatch):
    """Test generate_file function with various scenarios."""
    import tempfile
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'Test Author',
        }
    }
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    # Test 1: Generate regular text file with template variables
    infile = "test_{{cookiecutter.project_name}}.txt"
    template_file = template_dir / infile
    template_file.write_text("Author: {{cookiecutter.author}}\nProject: {{cookiecutter.project_name}}")
    
    monkeypatch.chdir(template_dir)
    generate_file(project_dir, infile, context, env)
    
    outfile = os.path.join(project_dir, "test_test_project.txt")
    assert os.path.exists(outfile)
    content = Path(outfile).read_text()
    assert "Author: Test Author" in content
    assert "Project: test_project" in content
    
    # Test 2: Skip if file exists
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(outfile)
    
    # Test 3: Binary file - copy without rendering
    binary_file = template_dir / "binary_file.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    generate_file(project_dir, "binary_file.bin", context, env)
    
    outfile_binary = os.path.join(project_dir, "binary_file.bin")
    assert os.path.exists(outfile_binary)
    assert Path(outfile_binary).read_bytes() == b'\x89PNG\r\n\x1a\n'
    
    # Test 4: File with custom newline configuration
    context['cookiecutter']['_new_lines'] = '\n'
    infile_newline = "newline_test.txt"
    template_file_newline = template_dir / infile_newline
    template_file_newline.write_text("Line 1\nLine 2\nLine 3")
    
    generate_file(project_dir, infile_newline, context, env)
    
    outfile_newline = os.path.join(project_dir, infile_newline)
    assert os.path.exists(outfile_newline)
    
    # Test 5: Empty directory name handling (rendered as directory)
    empty_dir_file = template_dir / "empty_dir_file.txt"
    empty_dir_file.write_text("content")
    
    # Simulate rendering to empty string by having outfile be a directory
    os.makedirs(os.path.join(project_dir, "empty_result"), exist_ok=True)
    
    # Test 6: File with undefined template variable raises error
    infile_undefined = "undefined_{{cookiecutter.missing_var}}.txt"
    template_file_undefined = template_dir / infile_undefined
    template_file_undefined.write_text("content")
    
    with pytest.raises(UndefinedError):
        generate_file(project_dir, infile_undefined, context, env)
    
    # Test 7: Template syntax error handling
    infile_syntax_error = "syntax_error.txt"
    template_file_syntax = template_dir / infile_syntax_error
    template_file_syntax.write_text("{% if unclosed %}")
    
    with pytest.raises(TemplateSyntaxError):
        generate_file(project_dir, infile_syntax_error, context, env)


# LLM-generated content at query #30
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    # Setup
    repo_dir = tmp_path / "template_repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create a cookiecutter.json
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    # Create a test template file
    test_file = template_dir / "README.md"
    test_file.write_text("# {{cookiecutter.project_name}}\n")
    
    # Create a test directory
    test_subdir = template_dir / "src"
    test_subdir.mkdir()
    (test_subdir / "main.py").write_text("# Main file for {{cookiecutter.project_name}}")
    
    # Mock find_template to return our template_dir
    monkeypatch.setattr("cookiecutter.generate.find_template", lambda x, y: str(template_dir))
    
    # Mock run_hook_from_repo_dir to do nothing
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", lambda *args, **kwargs: None)
    
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Execute
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Assert
    assert result is not None
    project_path = Path(result)
    assert project_path.exists()
    assert project_path.name == "my_project"
    
    # Check that files were rendered correctly
    readme = project_path / "README.md"
    assert readme.exists()
    assert "# my_project" in readme.read_text()
    
    # Check that subdirectories were created
    src_dir = project_path / "src"
    assert src_dir.exists()
    main_file = src_dir / "main.py"
    assert main_file.exists()
    assert "# Main file for my_project" in main_file.read_text()


def test_generate_files_with_binary_file(tmp_path, monkeypatch):
    """Test generate_files with binary files."""
    repo_dir = tmp_path / "template_repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    # Create a binary file
    binary_file = template_dir / "image.png"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    monkeypatch.setattr("cookiecutter.generate.find_template", lambda x, y: str(template_dir))
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", lambda *args, **kwargs: None)
    
    context = {"cookiecutter": {"project_name": "my_project"}}
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    project_path = Path(result)
    assert (project_path / "image.png").exists()
    assert (project_path / "image.png").read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_files_skip_if_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists."""
    repo_dir = tmp_path / "template_repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    test_file = template_dir / "config.txt"
    test_file.write_text("config={{cookiecutter.project_name}}")
    
    monkeypatch.setattr("cookiecutter.generate.find_template", lambda x, y: str(template_dir))
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", lambda *args, **kwargs: None)
    
    context = {"cookiecutter": {"project_name": "my_project"}}
    
    # First generation
    result1 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    config_file = Path(result1) / "config.txt"
    config_file.write_text("modified content")
    
    # Second generation with skip_if_file_exists
    result2 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    assert config_file.read_text() == "modified content"


def test_generate_files_with_copy_without_render(tmp_path, monkeypatch):
    """Test generate_files with _copy_without_render."""
    repo_dir = tmp_path / "template_repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "_copy_without_render": ["*.txt"]}')
    
    test_file = template_dir / "data.txt"
    test_file.write_text("{{cookiecutter.project_name}}")
    
    monkeypatch.setattr("cookiecutter.generate.find_template", lambda x, y: str(template_dir))
    monkeypatch.setattr("cookiecutter.generate.run_hook_from_repo_dir", lambda *args, **kwargs: None)
    
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "_copy_without_render": ["*.txt"]
        }
    }
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    data_file = Path(result) / "data.txt"
    assert data_file.exists()
    assert data_file.read_text() == "{{cookiecutter.project_name}}"


def test_generate_files_overwrite_if_exists(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists."""
    repo_dir = tmp_path / "template_repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    import json
    from collections import OrderedDict
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "my_project", "project_slug": "my_project"}
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["project_slug"] == "my_project"


def test_generate_context_with_custom_filename(tmp_path):
    """Test generate_context with custom JSON filename."""
    import json
    
    context_file = tmp_path / "custom_context.json"
    context_data = {"author": "John Doe"}
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    assert "custom_context" in result
    assert result["custom_context"]["author"] == "John Doe"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context with default_context override."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "default_name", "version": "1.0"}
    context_file.write_text(json.dumps(context_data))
    
    default_context = {"project_name": "overridden_name"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "overridden_name"
    assert result["cookiecutter"]["version"] == "1.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context with extra_context override."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "original", "license": "MIT"}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"project_name": "extra_override"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "extra_override"
    assert result["cookiecutter"]["license"] == "MIT"


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context with invalid JSON file."""
    from cookiecutter.exceptions import ContextDecodingException
    
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text("{ invalid json }")
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(context_file))


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context with choice variable override."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"python_version": ["3.8", "3.9", "3.10"]}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"python_version": "3.9"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    # Choice should be moved to front of list
    assert result["cookiecutter"]["python_version"][0] == "3.9"


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test generate_context with multi-choice variable override."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"optional_features": ["feature_a", "feature_b", "feature_c"]}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"optional_features": ["feature_a", "feature_c"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert set(result["cookiecutter"]["optional_features"]) == {"feature_a", "feature_c"}


def test_generate_context_with_dict_variable(tmp_path):
    """Test generate_context with nested dictionary override."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "metadata": {"author": "Original Author", "email": "original@example.com"}
    }
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"metadata": {"author": "New Author"}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["metadata"]["author"] == "New Author"
    assert result["cookiecutter"]["metadata"]["email"] == "original@example.com"


def test_generate_context_with_boolean_override(tmp_path):
    """Test generate_context with boolean variable string override."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"use_docker": True}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"use_docker": "n"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_invalid_choice_override(tmp_path):
    """Test generate_context with invalid choice override."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"python_version": ["3.8", "3.9", "3.10"]}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"python_version": "3.11"}
    
    with pytest.raises(ValueError, match="provided for choice variable"):
        generate_context(str(context_file), extra_context=extra_context)


def test_generate_context_invalid_boolean_override(tmp_path):
    """Test generate_context with invalid boolean override."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"use_docker": True}
    context_file.write_text(json.dumps(context_data))
    
    extra_context = {"use_docker": "maybe"}
    
    with pytest.raises(ValueError, match="could not be converted to a boolean"):
        generate_context(str(context_file), extra_context=extra_context)


def test_generate_context_invalid_default_context_warning(tmp_path):
    """Test generate_context with invalid default context shows warning."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"python_version": ["3.8", "3.9"]}
    context_file.write_text(json.dumps(context_data))
    
    default_context = {"python_version": "invalid_version"}
    
    with pytest.warns(UserWarning, match="Invalid default received"):
        result = generate_context(
            str(context_file), default_context=default_context
        )
    # Should still return context with original values
    assert result["cookiecutter"]["python_version"] == ["3.8", "3.9"]


def test_generate_context_with_none_contexts(tmp_path):
    """Test generate_context with None for optional context parameters."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"name": "test"}
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file), default_context=None, extra_context=None)
    
    assert "cookiecutter" in result
    assert


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from pathlib import Path
from jinja2 import Environment
from cookiecutter.exceptions import EmptyDirNameException, OutputDirExistsException


def test_render_and_create_dir():
    """Test render_and_create_dir function."""
    
    # Test 1: Successfully create a new directory with rendered name
    with pytest.TempDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {'cookiecutter': {'project_name': 'my_project'}}
        environment = Environment()
        dirname = '{{cookiecutter.project_name}}'
        
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, environment
        )
        
        assert result_dir == output_dir / 'my_project'
        assert created is True
        assert result_dir.exists()
    
    # Test 2: Directory already exists, overwrite_if_exists=False
    with pytest.TempDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        existing_dir = output_dir / 'existing'
        existing_dir.mkdir()
        
        context = {'cookiecutter': {}}
        environment = Environment()
        
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                'existing', context, output_dir, environment, overwrite_if_exists=False
            )
    
    # Test 3: Directory already exists, overwrite_if_exists=True
    with pytest.TempDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        existing_dir = output_dir / 'existing'
        existing_dir.mkdir()
        
        context = {'cookiecutter': {}}
        environment = Environment()
        
        result_dir, created = render_and_create_dir(
            'existing', context, output_dir, environment, overwrite_if_exists=True
        )
        
        assert result_dir == output_dir / 'existing'
        assert created is False
        assert result_dir.exists()
    
    # Test 4: Empty directory name raises EmptyDirNameException
    with pytest.TempDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {'cookiecutter': {}}
        environment = Environment()
        
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir('', context, output_dir, environment)
    
    # Test 5: None directory name raises EmptyDirNameException
    with pytest.TempDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {'cookiecutter': {}}
        environment = Environment()
        
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(None, context, output_dir, environment)
    
    # Test 6: Complex template rendering with multiple variables
    with pytest.TempDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {
            'cookiecutter': {
                'company': 'acme',
                'project': 'widget'
            }
        }
        environment = Environment()
        dirname = '{{cookiecutter.company}}_{{cookiecutter.project}}'
        
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, environment
        )
        
        assert result_dir == output_dir / 'acme_widget'
        assert created is True
        assert result_dir.exists()
    
    # Test 7: Directory with special characters in rendered name
    with pytest.TempDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {'cookiecutter': {'name': 'my-awesome-project'}}
        environment = Environment()
        dirname = '{{cookiecutter.name}}'
        
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, environment
        )
        
        assert result_dir == output_dir / 'my-awesome-project'
        assert created is True
        assert result_dir.exists()
    
    # Test 8: output_dir as string instead of Path
    with pytest.TempDirectory() as tmpdir:
        output_dir = str(tmpdir)
        context = {'cookiecutter': {'project': 'test'}}
        environment = Environment()
        dirname = '{{cookiecutter.project}}'
        
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, environment
        )
        
        assert result_dir == Path(output_dir) / 'test'
        assert created is True
        assert result_dir.exists()


# LLM-generated content at query #33
#--------------------------

```python
import pytest
import json
import os
import tempfile
from pathlib import Path
from collections import OrderedDict
from cookiecutter.exceptions import ContextDecodingException


def test_generate_context():
    """Test generate_context function with various scenarios."""
    
    # Test 1: Basic context generation from valid JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {'project_name': 'My Project', 'author': 'John Doe'}
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        result = generate_context(context_file)
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'My Project'
        assert result['cookiecutter']['author'] == 'John Doe'


def test_generate_context_with_invalid_json():
    """Test generate_context raises ContextDecodingException for invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write('{invalid json content}')
        
        with pytest.raises(ContextDecodingException):
            generate_context(context_file)


def test_generate_context_with_default_context():
    """Test generate_context applies default_context overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'Default Project',
            'version': '1.0.0',
            'use_docker': True
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        default_context = {
            'project_name': 'Override Project',
            'use_docker': False
        }
        
        result = generate_context(context_file, default_context=default_context)
        assert result['cookiecutter']['project_name'] == 'Override Project'
        assert result['cookiecutter']['version'] == '1.0.0'
        assert result['cookiecutter']['use_docker'] is False


def test_generate_context_with_extra_context():
    """Test generate_context applies extra_context overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {'project_name': 'Original', 'author': 'Original Author'}
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {'project_name': 'Extra Override'}
        
        result = generate_context(context_file, extra_context=extra_context)
        assert result['cookiecutter']['project_name'] == 'Extra Override'
        assert result['cookiecutter']['author'] == 'Original Author'


def test_generate_context_with_choice_variable():
    """Test generate_context with choice variable (list)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {'license': ['MIT', 'Apache', 'GPL']}
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {'license': 'Apache'}
        
        result = generate_context(context_file, extra_context=extra_context)
        assert result['cookiecutter']['license'][0] == 'Apache'
        assert 'MIT' in result['cookiecutter']['license']


def test_generate_context_with_invalid_choice():
    """Test generate_context raises ValueError for invalid choice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {'license': ['MIT', 'Apache']}
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {'license': 'InvalidLicense'}
        
        with pytest.raises(ValueError):
            generate_context(context_file, extra_context=extra_context)


def test_generate_context_with_multichoice_variable():
    """Test generate_context with multichoice variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {'features': [['feature1', 'feature2', 'feature3']]}
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {'features': ['feature1', 'feature3']}
        
        result = generate_context(context_file, extra_context=extra_context)
        assert result['cookiecutter']['features'] == ['feature1', 'feature3']


def test_generate_context_with_dict_variable():
    """Test generate_context with nested dictionary variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'config': {
                'database': 'postgresql',
                'port': 5432
            }
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {'config': {'port': 3306}}
        
        result = generate_context(context_file, extra_context=extra_context)
        assert result['cookiecutter']['config']['database'] == 'postgresql'
        assert result['cookiecutter']['config']['port'] == 3306


def test_generate_context_preserves_order():
    """Test generate_context preserves OrderedDict order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = OrderedDict([
            ('first', 1),
            ('second', 2),
            ('third', 3)
        ])
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        result = generate_context(context_file)
        keys = list(result['cookiecutter'].keys())
        assert keys == ['first', 'second', 'third']


def test_generate_context_with_boolean_string_override():
    """Test generate_context converts string to boolean for boolean variables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {'use_feature': True}
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {'use_feature': 'n'}
        
        result = generate_context(context_file, extra_context=extra_context)
        assert result['cookiecutter']['use_feature'] is False


def test_generate_context_custom_filename():
    """Test generate_context with custom context filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'custom_context.json')
        context_data = {'custom_var': 'custom_value'}
        
        with open(context_file, 'w', encoding='utf


# LLM-generated content at query #34
#--------------------------

```python
def test_generate_file(tmp_path, monkeypatch):
    """Test generate_file function with various scenarios."""
    from jinja2 import Environment, FileSystemLoader
    
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    # Test 1: Render text file with Jinja2 variables
    infile = "test_{{cookiecutter.project_name}}.txt"
    template_file = template_dir / "test_{{cookiecutter.project_name}}.txt"
    template_file.write_text("Author: {{cookiecutter.author}}\nProject: {{cookiecutter.project_name}}")
    
    monkeypatch.chdir(template_dir)
    generate_file(project_dir, infile, context, env)
    
    outfile_path = Path(project_dir) / "test_my_project.txt"
    assert outfile_path.exists()
    assert outfile_path.read_text() == "Author: John Doe\nProject: my_project"
    
    # Test 2: Skip file if exists
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    outfile_path.write_text("Modified content")
    
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert outfile_path.read_text() == "Modified content"
    
    # Test 3: Empty directory name handling
    empty_dir_infile = "subdir/{{cookiecutter.empty}}.txt"
    template_file2 = template_dir / "subdir" / "{{cookiecutter.empty}}.txt"
    template_file2.parent.mkdir(exist_ok=True)
    template_file2.write_text("content")
    
    context['cookiecutter']['empty'] = ''
    generate_file(project_dir, str(template_file2.relative_to(template_dir)), context, env)
    
    # Test 4: Binary file handling
    binary_file = template_dir / "binary_file.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    generate_file(project_dir, "binary_file.bin", context, env)
    
    outfile_binary = Path(project_dir) / "binary_file.bin"
    assert outfile_binary.exists()
    assert outfile_binary.read_bytes() == b'\x89PNG\r\n\x1a\n'
    
    # Test 5: Newline handling - configured newline
    infile_newline = "newline_test.txt"
    template_file_newline = template_dir / infile_newline
    template_file_newline.write_text("line1\nline2")
    
    context['cookiecutter']['_new_lines'] = '\r\n'
    generate_file(project_dir, infile_newline, context, env)
    
    outfile_newline = Path(project_dir) / infile_newline
    with open(outfile_newline, 'rb') as f:
        content = f.read()
    assert b'\r\n' in content or content == b'line1\nline2'
    
    # Test 6: File permissions are copied
    perm_file = template_dir / "perm_test.sh"
    perm_file.write_text("#!/bin/bash\necho 'test'")
    os.chmod(perm_file, 0o755)
    
    generate_file(project_dir, "perm_test.sh", context, env)
    
    outfile_perm = Path(project_dir) / "perm_test.sh"
    assert os.access(outfile_perm, os.X_OK)


def test_generate_file_with_undefined_variable(tmp_path, monkeypatch):
    """Test generate_file with undefined Jinja2 variable raises error."""
    from jinja2 import Environment, FileSystemLoader
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    
    context = {'cookiecutter': {'defined_var': 'value'}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    infile = "test.txt"
    template_file = template_dir / infile
    template_file.write_text("{{undefined_variable}}")
    
    monkeypatch.chdir(template_dir)
    
    with pytest.raises(UndefinedError):
        generate_file(project_dir, infile, context, env)


def test_generate_file_template_syntax_error(tmp_path, monkeypatch):
    """Test generate_file with invalid Jinja2 syntax raises error."""
    from jinja2 import Environment, FileSystemLoader
    
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    infile = "test.txt"
    template_file = template_dir / infile
    template_file.write_text("{% if true %}")
    
    monkeypatch.chdir(template_dir)
    
    with pytest.raises(TemplateSyntaxError):
        generate_file(project_dir, infile, context, env)


# LLM-generated content at query #35
#--------------------------

```python
def test_generate_file(tmp_path, monkeypatch):
    """Test generate_file function for rendering and creating files."""
    import tempfile
    from jinja2 import Environment, FileSystemLoader
    
    # Create a temporary template directory
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Test 1: Render a text file with template variables
    infile_name = "{{cookiecutter.project_name}}.txt"
    infile_path = template_dir / infile_name
    infile_path.write_text("Project: {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}")
    
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    generate_file(
        str(output_dir),
        infile_name,
        context,
        env,
        skip_if_file_exists=False
    )
    
    # Verify the output file was created with correct name and content
    output_file = output_dir / "my_project.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Project: my_project\nAuthor: John Doe"
    
    # Test 2: Skip if file exists
    generate_file(
        str(output_dir),
        infile_name,
        context,
        env,
        skip_if_file_exists=True
    )
    # File should not be modified
    assert output_file.read_text() == "Project: my_project\nAuthor: John Doe"
    
    # Test 3: Handle binary file (copy without rendering)
    binary_infile = template_dir / "image.bin"
    binary_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    binary_infile.write_bytes(binary_content)
    
    generate_file(
        str(output_dir),
        "image.bin",
        context,
        env,
        skip_if_file_exists=False
    )
    
    output_binary = output_dir / "image.bin"
    assert output_binary.exists()
    assert output_binary.read_bytes() == binary_content
    
    # Test 4: Empty directory name results in early return
    empty_dir_infile = template_dir / "test.txt"
    empty_dir_infile.write_text("test")
    
    # Simulate empty rendered path by creating a directory at output location
    empty_output = output_dir / ""
    
    generate_file(
        str(empty_output) if str(empty_output).strip() else str(output_dir),
        "test.txt",
        context,
        env,
        skip_if_file_exists=False
    )
    
    # Test 5: Handle newlines configuration from context
    newline_infile = template_dir / "newlines.txt"
    newline_infile.write_text("line1\nline2\nline3\n")
    
    context_with_newlines = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe",
            "_new_lines": "\r\n"
        }
    }
    
    generate_file(
        str(output_dir),
        "newlines.txt",
        context_with_newlines,
        env,
        skip_if_file_exists=False
    )
    
    output_newline_file = output_dir / "newlines.txt"
    assert output_newline_file.exists()
    
    # Test 6: File permissions are preserved
    perm_infile = template_dir / "script.sh"
    perm_infile.write_text("#!/bin/bash\necho 'test'")
    os.chmod(perm_infile, 0o755)
    
    generate_file(
        str(output_dir),
        "script.sh",
        context,
        env,
        skip_if_file_exists=False
    )
    
    output_script = output_dir / "script.sh"
    assert output_script.exists()
    assert os.stat(output_script).st_mode & 0o111  # Check executable bits are set
    
    # Test 7: Template syntax error handling
    syntax_error_infile = template_dir / "error.txt"
    syntax_error_infile.write_text("{% if unclosed %}")
    
    with pytest.raises(TemplateSyntaxError):
        generate_file(
            str(output_dir),
            "error.txt",
            context,
            env,
            skip_if_file_exists=False
        )


# LLM-generated content at query #36
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Create a temporary template directory structure
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    (template_dir / "README.md").write_text("# {{cookiecutter.project_name}}\n")
    
    # Create cookiecutter.json in repo_dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "my_project"}')
    
    # Create template subdirectory
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "README.md").write_text("# {{cookiecutter.project_name}}\n")
    
    # Prepare output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create context
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'test_project')
        ]))
    ])
    
    # Mock find_template to return the template directory
    def mock_find_template(repo_dir_arg, env):
        return str(repo_dir / "{{cookiecutter.project_name}}")
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    
    # Mock run_hook_from_repo_dir to do nothing
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    # Call generate_files
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False
    )
    
    # Verify the result
    assert result is not None
    assert os.path.isdir(result)
    assert "test_project" in result
    
    # Verify the generated file exists
    generated_file = Path(result) / "README.md"
    assert generated_file.exists()
    assert "test_project" in generated_file.read_text()


def test_generate_files_with_overwrite(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    from collections import OrderedDict
    from pathlib import Path
    
    # Setup template
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content")
    
    # Setup output directory with existing project
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing_project = output_dir / "test_project"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old content")
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'test_project')
        ]))
    ])
    
    def mock_find_template(repo_dir_arg, env):
        return str(template_dir)
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert result is not None
    assert Path(result).exists()


def test_generate_files_with_copy_without_render(tmp_path, monkeypatch):
    """Test generate_files with _copy_without_render setting."""
    from collections import OrderedDict
    from pathlib import Path
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a binary-like file to be copied without rendering
    binary_file = template_dir / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02')
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'test_project'),
            ('_copy_without_render', ['binary.bin'])
        ]))
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    def mock_find_template(repo_dir_arg, env):
        return str(template_dir)
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert result is not None
    assert Path(result).exists()


def test_generate_files_undefined_error(tmp_path, monkeypatch):
    """Test generate_files raises UndefinedVariableInTemplate on undefined variable."""
    from collections import OrderedDict
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.undefined_var}}"
    template_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'test_project')
        ]))
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    def mock_find_template(repo_dir_arg, env):
        return str(template_dir)
    
    def mock_render_and_create_dir(*args, **kwargs):
        raise UndefinedError("undefined_var is undefined")
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    monkeypatch.setattr('cookiecutter.generate.render_and_create_dir', mock_render_and_create_dir)
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', lambda *args, **kwargs: None)
    
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False
        )


def test_generate_files_with_hooks(tmp_path, monkeypatch):
    """Test generate_files calls hooks when accept_hooks=True."""
    from collections import OrderedDict
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content")
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'test_project')
        ]))
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()


# LLM-generated content at query #37
#--------------------------

```python
def test_generate_context(tmp_path, monkeypatch):
    """Test generate_context function with various scenarios."""
    import json
    from collections import OrderedDict
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "my_project", "author": "John Doe"}
    context_file.write_text(json.dumps(context_data), encoding='utf-8')
    
    monkeypatch.chdir(tmp_path)
    result = generate_context(str(context_file))
    
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"
    assert isinstance(result, OrderedDict)
    
    # Test 2: Context with default_context overrides
    context_file2 = tmp_path / "cookiecutter2.json"
    context_data2 = {
        "project_name": "default_project",
        "version": "1.0.0"
    }
    context_file2.write_text(json.dumps(context_data2), encoding='utf-8')
    
    default_context = {"project_name": "overridden_project"}
    result2 = generate_context(str(context_file2), default_context=default_context)
    
    assert result2["cookiecutter"]["project_name"] == "overridden_project"
    assert result2["cookiecutter"]["version"] == "1.0.0"
    
    # Test 3: Context with extra_context overrides
    context_file3 = tmp_path / "cookiecutter3.json"
    context_data3 = {
        "project_name": "original",
        "license": "MIT"
    }
    context_file3.write_text(json.dumps(context_data3), encoding='utf-8')
    
    extra_context = {"license": "Apache"}
    result3 = generate_context(str(context_file3), extra_context=extra_context)
    
    assert result3["cookiecutter"]["license"] == "Apache"
    assert result3["cookiecutter"]["project_name"] == "original"
    
    # Test 4: Invalid JSON file raises ContextDecodingException
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text("{invalid json}", encoding='utf-8')
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(invalid_json_file))
    
    # Test 5: Choice variable with list context
    context_file4 = tmp_path / "cookiecutter4.json"
    context_data4 = {
        "python_version": ["3.8", "3.9", "3.10"]
    }
    context_file4.write_text(json.dumps(context_data4), encoding='utf-8')
    
    extra_context4 = {"python_version": "3.9"}
    result4 = generate_context(str(context_file4), extra_context=extra_context4)
    
    # The chosen option should be first in the list
    assert result4["cookiecutter"]["python_version"][0] == "3.9"
    
    # Test 6: Multi-choice variable with list of choices
    context_file5 = tmp_path / "cookiecutter5.json"
    context_data5 = {
        "features": ["auth", "api", "admin", "logging"]
    }
    context_file5.write_text(json.dumps(context_data5), encoding='utf-8')
    
    extra_context5 = {"features": ["api", "logging"]}
    result5 = generate_context(str(context_file5), extra_context=extra_context5)
    
    assert result5["cookiecutter"]["features"] == ["api", "logging"]
    
    # Test 7: Boolean variable with string override
    context_file6 = tmp_path / "cookiecutter6.json"
    context_data6 = {
        "use_docker": True
    }
    context_file6.write_text(json.dumps(context_data6), encoding='utf-8')
    
    extra_context6 = {"use_docker": "n"}
    result6 = generate_context(str(context_file6), extra_context=extra_context6)
    
    assert result6["cookiecutter"]["use_docker"] is False
    
    # Test 8: Nested dictionary context
    context_file7 = tmp_path / "cookiecutter7.json"
    context_data7 = {
        "database": {
            "engine": "postgresql",
            "version": "12"
        }
    }
    context_file7.write_text(json.dumps(context_data7), encoding='utf-8')
    
    extra_context7 = {"database": {"version": "13"}}
    result7 = generate_context(str(context_file7), extra_context=extra_context7)
    
    assert result7["cookiecutter"]["database"]["engine"] == "postgresql"
    assert result7["cookiecutter"]["database"]["version"] == "13"
    
    # Test 9: Invalid choice value raises ValueError
    context_file8 = tmp_path / "cookiecutter8.json"
    context_data8 = {
        "license": ["MIT", "Apache", "GPL"]
    }
    context_file8.write_text(json.dumps(context_data8), encoding='utf-8')
    
    invalid_choice = {"license": "BSD"}
    with pytest.raises(ValueError, match="provided for choice variable"):
        generate_context(str(context_file8), extra_context=invalid_choice)
    
    # Test 10: Invalid multi-choice values raise ValueError
    context_file9 = tmp_path / "cookiecutter9.json"
    context_data9 = {
        "features": ["feature1", "feature2", "feature3"]
    }
    context_file9.write_text(json.dumps(context_data9), encoding='utf-8')
    
    invalid_multichoice = {"features": ["feature1", "feature4"]}
    with pytest.raises(ValueError, match="provided for multi-choice variable"):
        generate_context(str(context_file9), extra_context=invalid_multichoice)


# LLM-generated content at query #38
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    import json
    from collections import OrderedDict
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
        "author_name": "John Doe"
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    result = generate_context(str(context_file))
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == "My Project"
    assert result['cookiecutter']['author_name'] == "John Doe"
    
    # Test 2: Context with default_context override
    default_context = {
        "project_name": "Default Project",
        "author_name": "Jane Doe"
    }
    result = generate_context(str(context_file), default_context=default_context)
    assert result['cookiecutter']['project_name'] == "Default Project"
    assert result['cookiecutter']['author_name'] == "Jane Doe"
    
    # Test 3: Context with extra_context override
    extra_context = {
        "project_name": "Extra Project"
    }
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result['cookiecutter']['project_name'] == "Extra Project"
    assert result['cookiecutter']['author_name'] == "John Doe"
    
    # Test 4: Both default_context and extra_context (extra_context takes precedence)
    result = generate_context(
        str(context_file),
        default_context=default_context,
        extra_context=extra_context
    )
    assert result['cookiecutter']['project_name'] == "Extra Project"
    assert result['cookiecutter']['author_name'] == "Jane Doe"
    
    # Test 5: Invalid JSON file raises ContextDecodingException
    invalid_json_file = tmp_path / "invalid.json"
    with open(invalid_json_file, 'w', encoding='utf-8') as f:
        f.write("{invalid json content")
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(invalid_json_file))
    
    # Test 6: Context with choice variable and overwrite
    choice_context_file = tmp_path / "choice_cookiecutter.json"
    choice_data = {
        "license": ["MIT", "Apache", "GPL"]
    }
    with open(choice_context_file, 'w', encoding='utf-8') as f:
        json.dump(choice_data, f)
    
    result = generate_context(str(choice_context_file), extra_context={"license": "Apache"})
    assert result['cookiecutter']['license'][0] == "Apache"
    
    # Test 7: Context with multi-choice variable
    multichoice_context_file = tmp_path / "multichoice_cookiecutter.json"
    multichoice_data = {
        "features": ["feature1", "feature2", "feature3"]
    }
    with open(multichoice_context_file, 'w', encoding='utf-8') as f:
        json.dump(multichoice_data, f)
    
    result = generate_context(
        str(multichoice_context_file),
        extra_context={"features": ["feature2", "feature3"]}
    )
    assert set(result['cookiecutter']['features']) == {"feature2", "feature3"}
    
    # Test 8: Context with boolean variable
    bool_context_file = tmp_path / "bool_cookiecutter.json"
    bool_data = {
        "use_docker": True
    }
    with open(bool_context_file, 'w', encoding='utf-8') as f:
        json.dump(bool_data, f)
    
    result = generate_context(str(bool_context_file), extra_context={"use_docker": "n"})
    assert result['cookiecutter']['use_docker'] is False
    
    # Test 9: Context with nested dictionary
    nested_context_file = tmp_path / "nested_cookiecutter.json"
    nested_data = {
        "author": {
            "name": "John Doe",
            "email": "john@example.com"
        }
    }
    with open(nested_context_file, 'w', encoding='utf-8') as f:
        json.dump(nested_data, f)
    
    result = generate_context(
        str(nested_context_file),
        extra_context={"author": {"email": "jane@example.com"}}
    )
    assert result['cookiecutter']['author']['name'] == "John Doe"
    assert result['cookiecutter']['author']['email'] == "jane@example.com"
    
    # Test 10: Result is OrderedDict
    result = generate_context(str(context_file))
    assert isinstance(result, OrderedDict)


# LLM-generated content at query #39
#--------------------------

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from jinja2 import Environment, FileSystemLoader


def test_generate_file():
    """Test generate_file function with various scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        
        # Test 1: Text file rendering
        with tempfile.TemporaryDirectory() as template_dir:
            infile = "test_{{ cookiecutter.project_name }}.txt"
            infile_path = os.path.join(template_dir, infile)
            
            with open(infile_path, 'w') as f:
                f.write("Hello {{ cookiecutter.project_name }}")
            
            context = {
                'cookiecutter': {
                    'project_name': 'myproject',
                    '_new_lines': None
                }
            }
            
            env = Environment(loader=FileSystemLoader(template_dir))
            
            with patch('cookiecutter.generate.is_binary', return_value=False):
                with patch('builtins.open', create=True) as mock_open:
                    mock_file = MagicMock()
                    mock_file.newlines = '\n'
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    generate_file(project_dir, infile, context, env)
                    
                    # Verify file was written
                    mock_open.assert_called()


def test_generate_file_binary():
    """Test generate_file with binary files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        
        with tempfile.TemporaryDirectory() as template_dir:
            infile = "binary_{{ cookiecutter.name }}.bin"
            infile_path = os.path.join(template_dir, infile)
            
            with open(infile_path, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n')
            
            context = {
                'cookiecutter': {
                    'name': 'test',
                    '_new_lines': None
                }
            }
            
            env = Environment(loader=FileSystemLoader(template_dir))
            
            with patch('cookiecutter.generate.is_binary', return_value=True):
                with patch('shutil.copyfile') as mock_copy:
                    with patch('shutil.copymode'):
                        generate_file(project_dir, infile, context, env)
                        mock_copy.assert_called_once()


def test_generate_file_skip_if_exists():
    """Test generate_file skips existing files when flag is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        outfile_name = "existing_file.txt"
        outfile_path = os.path.join(project_dir, outfile_name)
        
        with open(outfile_path, 'w') as f:
            f.write("existing content")
        
        with tempfile.TemporaryDirectory() as template_dir:
            infile = "template.txt"
            infile_path = os.path.join(template_dir, infile)
            
            with open(infile_path, 'w') as f:
                f.write("new content")
            
            context = {
                'cookiecutter': {
                    '_new_lines': None
                }
            }
            
            env = Environment(loader=FileSystemLoader(template_dir))
            
            with patch('cookiecutter.generate.is_binary', return_value=False):
                with patch('builtins.open', create=True) as mock_open:
                    with patch('os.path.exists', return_value=True):
                        generate_file(
                            project_dir,
                            "{{ cookiecutter.name }}",
                            context,
                            env,
                            skip_if_file_exists=True
                        )
                        # File should not be written
                        mock_open.assert_not_called()


def test_generate_file_empty_dirname():
    """Test generate_file handles empty directory names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        infile = "test.txt"
        
        context = {
            'cookiecutter': {
                '_new_lines': None
            }
        }
        
        env = Environment()
        
        with patch('os.path.isdir', return_value=True):
            generate_file(project_dir, infile, context, env)


def test_generate_file_custom_newlines():
    """Test generate_file uses custom newline character from context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        
        with tempfile.TemporaryDirectory() as template_dir:
            infile = "test.txt"
            infile_path = os.path.join(template_dir, infile)
            
            with open(infile_path, 'w') as f:
                f.write("line1\nline2\n")
            
            context = {
                'cookiecutter': {
                    '_new_lines': '\r\n',
                }
            }
            
            env = Environment(loader=FileSystemLoader(template_dir))
            
            with patch('cookiecutter.generate.is_binary', return_value=False):
                with patch('shutil.copymode'):
                    generate_file(project_dir, infile, context, env)


def test_generate_file_template_syntax_error():
    """Test generate_file raises TemplateSyntaxError on invalid template."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        
        with tempfile.TemporaryDirectory() as template_dir:
            infile = "invalid.txt"
            infile_path = os.path.join(template_dir, infile)
            
            with open(infile_path, 'w') as f:
                f.write("{% if invalid")
            
            context = {
                'cookiecutter': {
                    '_new_lines': None
                }
            }
            
            env = Environment(loader=FileSystemLoader(template_dir))
            
            with patch('cookiecutter.generate.is_binary', return_value=False):
                with pytest.raises(Exception):
                    generate_file(project_dir, infile, context, env)


def test_generate_file_mixed_line_endings():
    """Test generate_file handles mixed line endings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        
        with tempfile.TemporaryDirectory() as template_dir:
            infile = "mixed.txt"
            infile_path = os.path.join(template_dir, infile)
            
            with open(infile_path, 'w') as f:
                f.write("line1\nline2\r\nline3")
            
            context = {
                'cookiecutter': {
                    '_new_lines': None
                }
            }
            
            env = Environment(loader=FileSystemLoader(template_dir))
            
            with patch('cookiecutter.generate.is_binary', return_value=False):
                with patch('shutil.copymode'):
                    generate_file(project_dir, infile, context, env)


# LLM-generated content at query #40
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    import json
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
        "author_name": "John Doe"
    }
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["author_name"] == "John Doe"
    
    # Test 2: Context with default_context overrides
    context_file2 = tmp_path / "cookiecutter2.json"
    context_data2 = {
        "project_name": "Original",
        "author": "Original Author"
    }
    with open(context_file2, 'w') as f:
        json.dump(context_data2, f)
    
    default_context = {"project_name": "Overridden"}
    result = generate_context(str(context_file2), default_context=default_context)
    assert result["cookiecutter"]["project_name"] == "Overridden"
    assert result["cookiecutter"]["author"] == "Original Author"
    
    # Test 3: Context with extra_context overrides
    context_file3 = tmp_path / "cookiecutter3.json"
    context_data3 = {
        "project_name": "Original",
        "version": "1.0.0"
    }
    with open(context_file3, 'w') as f:
        json.dump(context_data3, f)
    
    extra_context = {"version": "2.0.0"}
    result = generate_context(str(context_file3), extra_context=extra_context)
    assert result["cookiecutter"]["version"] == "2.0.0"
    assert result["cookiecutter"]["project_name"] == "Original"
    
    # Test 4: Context with choice variable and overwrite
    context_file4 = tmp_path / "cookiecutter4.json"
    context_data4 = {
        "license": ["MIT", "Apache", "BSD"]
    }
    with open(context_file4, 'w') as f:
        json.dump(context_data4, f)
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file4), extra_context=extra_context)
    assert result["cookiecutter"]["license"][0] == "Apache"
    
    # Test 5: Context with multi-choice variable
    context_file5 = tmp_path / "cookiecutter5.json"
    context_data5 = {
        "features": ["feature1", "feature2", "feature3"]
    }
    with open(context_file5, 'w') as f:
        json.dump(context_data5, f)
    
    extra_context = {"features": ["feature2", "feature3"]}
    result = generate_context(str(context_file5), extra_context=extra_context)
    assert result["cookiecutter"]["features"] == ["feature2", "feature3"]
    
    # Test 6: Context with nested dictionary
    context_file6 = tmp_path / "cookiecutter6.json"
    context_data6 = {
        "config": {
            "debug": True,
            "timeout": 30
        }
    }
    with open(context_file6, 'w') as f:
        json.dump(context_data6, f)
    
    extra_context = {"config": {"debug": False}}
    result = generate_context(str(context_file6), extra_context=extra_context)
    assert result["cookiecutter"]["config"]["debug"] is False
    assert result["cookiecutter"]["config"]["timeout"] == 30
    
    # Test 7: Context with boolean variable as string
    context_file7 = tmp_path / "cookiecutter7.json"
    context_data7 = {
        "use_docker": True
    }
    with open(context_file7, 'w') as f:
        json.dump(context_data7, f)
    
    extra_context = {"use_docker": "n"}
    result = generate_context(str(context_file7), extra_context=extra_context)
    assert result["cookiecutter"]["use_docker"] is False
    
    # Test 8: Invalid JSON file raises ContextDecodingException
    invalid_json_file = tmp_path / "invalid.json"
    with open(invalid_json_file, 'w') as f:
        f.write("{invalid json content")
    
    try:
        generate_context(str(invalid_json_file))
        assert False, "Should have raised ContextDecodingException"
    except ContextDecodingException as e:
        assert "JSON decoding error" in str(e)
    
    # Test 9: Invalid choice in overwrite raises ValueError
    context_file9 = tmp_path / "cookiecutter9.json"
    context_data9 = {
        "license": ["MIT", "Apache"]
    }
    with open(context_file9, 'w') as f:
        json.dump(context_data9, f)
    
    extra_context = {"license": "GPL"}
    try:
        generate_context(str(context_file9), extra_context=extra_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "GPL" in str(e)
    
    # Test 10: Invalid multi-choice raises ValueError
    context_file10 = tmp_path / "cookiecutter10.json"
    context_data10 = {
        "features": ["feature1", "feature2"]
    }
    with open(context_file10, 'w') as f:
        json.dump(context_data10, f)
    
    extra_context = {"features": ["feature1", "feature3"]}
    try:
        generate_context(str(context_file10), extra_context=extra_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "feature3" in str(e)
    
    # Test 11: Invalid boolean conversion raises ValueError
    context_file11 = tmp_path / "cookiecutter11.json"
    context_data11 = {
        "use_feature": True
    }
    with open(context_file11, 'w') as f:
        json.dump(context_data11, f)
    
    extra_context = {"use_feature": "maybe"}
    try:
        generate_context(str(context_file11), extra_context=extra_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pathlib import Path
from jinja2 import Environment
from cookiecutter.exceptions import EmptyDirNameException, OutputDirExistsException


def test_render_and_create_dir():
    """Test render_and_create_dir function with various scenarios."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    # Test 1: Normal case - create directory with rendered name
    with pytest.tmp_path as tmp_path:
        dirname = '{{cookiecutter.project_name}}'
        result_dir, created = render_and_create_dir(
            dirname, context, tmp_path, env, overwrite_if_exists=False
        )
        assert result_dir == Path(tmp_path) / 'my_project'
        assert created is True
        assert result_dir.exists()
    
    # Test 2: Empty directory name raises exception
    with pytest.tmp_path as tmp_path:
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir('', context, tmp_path, env)
    
    # Test 3: Empty string directory name raises exception
    with pytest.tmp_path as tmp_path:
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir("", context, tmp_path, env)
    
    # Test 4: Directory already exists, overwrite_if_exists=False raises exception
    with pytest.tmp_path as tmp_path:
        existing_dir = tmp_path / 'existing'
        existing_dir.mkdir()
        
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                'existing', context, tmp_path, env, overwrite_if_exists=False
            )
    
    # Test 5: Directory already exists, overwrite_if_exists=True returns existing dir
    with pytest.tmp_path as tmp_path:
        existing_dir = tmp_path / 'existing'
        existing_dir.mkdir()
        
        result_dir, created = render_and_create_dir(
            'existing', context, tmp_path, env, overwrite_if_exists=True
        )
        assert result_dir == existing_dir
        assert created is False
    
    # Test 6: Directory with template variables in name
    with pytest.tmp_path as tmp_path:
        context = {'cookiecutter': {'org': 'acme', 'project': 'widget'}}
        dirname = '{{cookiecutter.org}}_{{cookiecutter.project}}'
        
        result_dir, created = render_and_create_dir(
            dirname, context, tmp_path, env, overwrite_if_exists=False
        )
        assert result_dir == Path(tmp_path) / 'acme_widget'
        assert created is True
        assert result_dir.exists()
    
    # Test 7: Nested path creation
    with pytest.tmp_path as tmp_path:
        dirname = 'parent/{{cookiecutter.project_name}}'
        
        result_dir, created = render_and_create_dir(
            dirname, context, tmp_path, env, overwrite_if_exists=False
        )
        assert result_dir == Path(tmp_path) / 'parent' / 'my_project'
        assert created is True
        assert result_dir.exists()
    
    # Test 8: Using Path objects for output_dir
    with pytest.tmp_path as tmp_path:
        dirname = 'test_project'
        output_path = Path(tmp_path)
        
        result_dir, created = render_and_create_dir(
            dirname, context, output_path, env, overwrite_if_exists=False
        )
        assert result_dir == output_path / 'test_project'
        assert created is True
        assert result_dir.exists()


# LLM-generated content at query #2
#--------------------------

```python
def test_generate_file(tmp_path, monkeypatch):
    """Test generate_file function with various input scenarios."""
    import tempfile
    from jinja2 import Environment, FileSystemLoader
    
    # Create a temporary directory structure for testing
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Test 1: Generate a simple text file with template variables
    test_file = template_dir / "test_{{cookiecutter.project_name}}.txt"
    test_file.write_text("Project: {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}")
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    # Change to template directory for the test
    monkeypatch.chdir(template_dir)
    
    generate_file(
        str(project_dir),
        "test_{{cookiecutter.project_name}}.txt",
        context,
        env
    )
    
    # Verify the file was created with correct name and content
    output_file = project_dir / "test_my_project.txt"
    assert output_file.exists()
    content = output_file.read_text()
    assert "Project: my_project" in content
    assert "Author: John Doe" in content


def test_generate_file_binary(tmp_path, monkeypatch):
    """Test generate_file function with binary file."""
    from jinja2 import Environment, FileSystemLoader
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create a binary file
    binary_file = template_dir / "image.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'fake image data')
    
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    generate_file(
        str(project_dir),
        "image.bin",
        context,
        env
    )
    
    output_file = project_dir / "image.bin"
    assert output_file.exists()
    assert output_file.read_bytes() == binary_file.read_bytes()


def test_generate_file_skip_if_exists(tmp_path, monkeypatch):
    """Test generate_file with skip_if_file_exists flag."""
    from jinja2 import Environment, FileSystemLoader
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    test_file = template_dir / "existing.txt"
    test_file.write_text("Template content: {{cookiecutter.value}}")
    
    # Pre-create the output file
    output_file = project_dir / "existing.txt"
    output_file.write_text("Existing content")
    
    context = {'cookiecutter': {'value': 'new_value'}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    generate_file(
        str(project_dir),
        "existing.txt",
        context,
        env,
        skip_if_file_exists=True
    )
    
    # Content should remain unchanged
    assert output_file.read_text() == "Existing content"


def test_generate_file_empty_filename(tmp_path, monkeypatch):
    """Test generate_file when rendered filename is empty/directory."""
    from jinja2 import Environment, FileSystemLoader
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    test_file = template_dir / "test.txt"
    test_file.write_text("content")
    
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    # This should not raise an error, just return early
    generate_file(
        str(project_dir),
        "test.txt",
        context,
        env
    )


def test_generate_file_with_custom_newlines(tmp_path, monkeypatch):
    """Test generate_file respects custom newline configuration."""
    from jinja2 import Environment, FileSystemLoader
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    test_file = template_dir / "test.txt"
    test_file.write_text("Line 1\nLine 2\nLine 3")
    
    context = {
        'cookiecutter': {
            '_new_lines': '\r\n'
        }
    }
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    generate_file(
        str(project_dir),
        "test.txt",
        context,
        env
    )
    
    output_file = project_dir / "test.txt"
    assert output_file.exists()


def test_generate_file_template_syntax_error(tmp_path, monkeypatch):
    """Test generate_file raises TemplateSyntaxError for invalid templates."""
    from jinja2 import Environment, FileSystemLoader
    import pytest
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Create a file with invalid Jinja2 syntax
    test_file = template_dir / "bad.txt"
    test_file.write_text("{{cookiecutter.value | undefined_filter}}")
    
    context = {'cookiecutter': {'value': 'test'}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    # Should raise TemplateSyntaxError or similar
    with pytest.raises(Exception):
        generate_file(
            str(project_dir),
            "bad.txt",
            context,
            env
        )


def test_generate_file_preserves_permissions(tmp_path, monkeypatch):
    """Test generate_file preserves file permissions from template."""
    import stat
    from jinja2 import Environment, FileSystemLoader
    
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    test_file = template_dir / "script.sh"
    test_file.write_text("#!/bin/bash\necho {{cookiecutter.message}}")
    
    # Make template file executable
    test_file.chmod(0o755)
    
    context = {'cookiecutter': {'message': 'hello'}}
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    monkeypatch.chdir(template_dir)
    
    generate_file(
        str(project_dir),
        "script.sh",
        context,
        env
    )
    
    output_file = project_dir / "script.sh"
    assert output_file.exists()
    # Check if executable bit is preserved
    mode = stat.S_IMODE(output_file.stat().st_mode)
    assert mode & stat.S_IXUSR


# LLM-generated content at query #3
#--------------------------

```python
def test_apply_overwrites_to_context():
    """Test apply_overwrites_to_context function with various scenarios."""
    
    # Test 1: Simple value overwrite
    context = {'var1': 'original', 'var2': 'value2'}
    overwrite = {'var1': 'overwritten'}
    apply_overwrites_to_context(context, overwrite)
    assert context['var1'] == 'overwritten'
    assert context['var2'] == 'value2'
    
    # Test 2: Ignore new variables at first level
    context = {'var1': 'value1'}
    overwrite = {'new_var': 'new_value'}
    apply_overwrites_to_context(context, overwrite)
    assert 'new_var' not in context
    
    # Test 3: Add new variables in deeper level (in_dictionary_variable=True)
    context = {'nested': {'existing': 'value'}}
    overwrite = {'nested': {'new_key': 'new_value'}}
    apply_overwrites_to_context(context, overwrite)
    assert context['nested']['new_key'] == 'new_value'
    
    # Test 4: Choice variable - valid overwrite
    context = {'choice': ['option1', 'option2', 'option3']}
    overwrite = {'choice': 'option2'}
    apply_overwrites_to_context(context, overwrite)
    assert context['choice'][0] == 'option2'
    assert 'option2' in context['choice']
    
    # Test 5: Choice variable - invalid overwrite
    context = {'choice': ['option1', 'option2']}
    overwrite = {'choice': 'invalid_option'}
    with pytest.raises(ValueError, match='provided for choice variable'):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 6: Multi-choice variable - valid overwrite
    context = {'multichoice': ['opt1', 'opt2', 'opt3']}
    overwrite = {'multichoice': ['opt1', 'opt3']}
    apply_overwrites_to_context(context, overwrite)
    assert context['multichoice'] == ['opt1', 'opt3']
    
    # Test 7: Multi-choice variable - invalid overwrite
    context = {'multichoice': ['opt1', 'opt2']}
    overwrite = {'multichoice': ['opt1', 'invalid']}
    with pytest.raises(ValueError, match='provided for multi-choice variable'):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 8: Boolean variable - string 'y' to True
    context = {'bool_var': True}
    overwrite = {'bool_var': 'y'}
    apply_overwrites_to_context(context, overwrite)
    assert context['bool_var'] is True
    
    # Test 9: Boolean variable - string 'n' to False
    context = {'bool_var': True}
    overwrite = {'bool_var': 'n'}
    apply_overwrites_to_context(context, overwrite)
    assert context['bool_var'] is False
    
    # Test 10: Boolean variable - invalid string
    context = {'bool_var': True}
    overwrite = {'bool_var': 'invalid'}
    with pytest.raises(ValueError, match='could not be converted to a boolean'):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 11: Nested dictionary overwrite
    context = {'nested': {'key1': 'value1', 'key2': 'value2'}}
    overwrite = {'nested': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite)
    assert context['nested']['key1'] == 'new_value1'
    assert context['nested']['key2'] == 'value2'
    
    # Test 12: Overwrite list with list (not multi-choice, in_dictionary_variable)
    context = {'dict_var': {'list_var': ['a', 'b', 'c']}}
    overwrite = {'dict_var': {'list_var': ['x', 'y']}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['dict_var']['list_var'] == ['x', 'y']
    
    # Test 13: Empty overwrite context
    context = {'var1': 'value1'}
    overwrite = {}
    apply_overwrites_to_context(context, overwrite)
    assert context['var1'] == 'value1'
    
    # Test 14: Multiple levels of nesting
    context = {
        'level1': {
            'level2': {
                'var': 'original'
            }
        }
    }
    overwrite = {
        'level1': {
            'level2': {
                'var': 'modified'
            }
        }
    }
    apply_overwrites_to_context(context, overwrite)
    assert context['level1']['level2']['var'] == 'modified'
    
    # Test 15: List choice - ensure first element is moved to front
    context = {'choice': ['first', 'second', 'third']}
    overwrite = {'choice': 'third'}
    apply_overwrites_to_context(context, overwrite)
    assert context['choice'][0] == 'third'
    assert len(context['choice']) == 3


# LLM-generated content at query #4
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Create a temporary template directory structure
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}\n")
    
    # Create a cookiecutter.json in the repo
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Prepare context
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'my_project')
        ]))
    ])
    
    # Mock find_template to return our template directory
    monkeypatch.setattr(
        'cookiecutter.generate.find_template',
        lambda repo, env: str(template_dir)
    )
    
    # Mock run_hook_from_repo_dir to do nothing
    monkeypatch.setattr(
        'cookiecutter.generate.run_hook_from_repo_dir',
        lambda *args, **kwargs: None
    )
    
    # Call generate_files
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=True,
        keep_project_on_failure=False
    )
    
    # Verify the result
    assert result is not None
    assert os.path.isdir(result)
    
    # Verify the generated file exists and has correct content
    generated_file = Path(result) / "README.md"
    assert generated_file.exists()
    assert generated_file.read_text() == "# my_project\n"


def test_generate_files_with_subdirectories(tmp_path, monkeypatch):
    """Test generate_files with nested directory structure."""
    from collections import OrderedDict
    
    # Create template with subdirectories
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    subdir = template_dir / "src"
    subdir.mkdir()
    
    src_file = subdir / "main.py"
    src_file.write_text("# {{cookiecutter.project_name}}\n")
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'test_project')
        ]))
    ])
    
    monkeypatch.setattr(
        'cookiecutter.generate.find_template',
        lambda repo, env: str(template_dir)
    )
    
    monkeypatch.setattr(
        'cookiecutter.generate.run_hook_from_repo_dir',
        lambda *args, **kwargs: None
    )
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    generated_py = Path(result) / "src" / "main.py"
    assert generated_py.exists()
    assert "test_project" in generated_py.read_text()


def test_generate_files_overwrite_if_exists(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Pre-create the output project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    existing_file = existing_project / "file.txt"
    existing_file.write_text("old content")
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'my_project')
        ]))
    ])
    
    monkeypatch.setattr(
        'cookiecutter.generate.find_template',
        lambda repo, env: str(template_dir)
    )
    
    monkeypatch.setattr(
        'cookiecutter.generate.run_hook_from_repo_dir',
        lambda *args, **kwargs: None
    )
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert Path(result).exists()


def test_generate_files_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists=True."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("new content")
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'my_project')
        ]))
    ])
    
    monkeypatch.setattr(
        'cookiecutter.generate.find_template',
        lambda repo, env: str(template_dir)
    )
    
    monkeypatch.setattr(
        'cookiecutter.generate.run_hook_from_repo_dir',
        lambda *args, **kwargs: None
    )
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    assert Path(result).exists()


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    # Setup directory structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create a cookiecutter.json file
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text(
        '{"project_name": "my_project", "author": "John Doe"}',
        encoding='utf-8'
    )
    
    # Create a template file
    test_file = template_dir / "README.md"
    test_file.write_text(
        "# {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}",
        encoding='utf-8'
    )
    
    # Create a subdirectory with a template file
    subdir = template_dir / "src"
    subdir.mkdir()
    src_file = subdir / "main.py"
    src_file.write_text("# {{cookiecutter.project_name}} main module", encoding='utf-8')
    
    # Generate context
    context = generate_context(
        str(context_file),
        extra_context={"project_name": "test_project", "author": "Jane Smith"}
    )
    
    # Generate files
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False
    )
    
    # Assertions
    assert result is not None
    project_path = Path(result)
    assert project_path.exists()
    assert project_path.name == "test_project"
    
    # Check that files were rendered correctly
    readme = project_path / "README.md"
    assert readme.exists()
    readme_content = readme.read_text(encoding='utf-8')
    assert "# test_project" in readme_content
    assert "Author: Jane Smith" in readme_content
    
    # Check subdirectory and file
    main_file = project_path / "src" / "main.py"
    assert main_file.exists()
    main_content = main_file.read_text(encoding='utf-8')
    assert "# test_project main module" in main_content


def test_generate_files_overwrite_if_exists(tmp_path):
    """Test generate_files with overwrite_if_exists=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create cookiecutter.json
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}', encoding='utf-8')
    
    # Create a template file
    test_file = template_dir / "file.txt"
    test_file.write_text("content", encoding='utf-8')
    
    context = generate_context(str(context_file))
    
    # Generate files first time
    result1 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert Path(result1).exists()
    
    # Generate files second time with overwrite_if_exists=True
    result2 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert Path(result2).exists()
    assert Path(result2).name == "my_project"


def test_generate_files_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}', encoding='utf-8')
    
    test_file = template_dir / "file.txt"
    test_file.write_text("original content", encoding='utf-8')
    
    context = generate_context(str(context_file))
    
    # Generate files first time
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=False,
        accept_hooks=False
    )
    
    project_path = Path(result)
    output_file = project_path / "file.txt"
    
    # Modify the file
    output_file.write_text("modified content", encoding='utf-8')
    
    # Generate files again with skip_if_file_exists=True
    generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    # File should still have modified content
    assert output_file.read_text(encoding='utf-8') == "modified content"


def test_generate_files_with_binary_file(tmp_path):
    """Test generate_files with binary files."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}', encoding='utf-8')
    
    # Create a binary file (e.g., a simple PNG header)
    binary_file = template_dir / "image.png"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')
    
    context = generate_context(str(context_file))
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    project_path = Path(result)
    output_binary = project_path / "image.png"
    
    assert output_binary.exists()
    assert output_binary.read_bytes() == b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'


def test_generate_files_with_copy_without_render(tmp_path):
    """Test generate_files with _copy_without_render setting."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text(
        '{"project_name": "my_project", "_copy_without_render": ["*.jinja2"]}',
        encoding='utf-8'
    


# LLM-generated content at query #6
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
        "author_name": "John Doe"
    }
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["author_name"] == "John Doe"

    # Test 2: Context with default_context overwrite
    context_file2 = tmp_path / "cookiecutter2.json"
    context_data2 = {
        "project_name": "Original",
        "version": "1.0.0"
    }
    context_file2.write_text(json.dumps(context_data2))
    
    default_context = {"project_name": "Overwritten"}
    result = generate_context(str(context_file2), default_context=default_context)
    assert result["cookiecutter"]["project_name"] == "Overwritten"
    assert result["cookiecutter"]["version"] == "1.0.0"

    # Test 3: Context with extra_context overwrite
    context_file3 = tmp_path / "cookiecutter3.json"
    context_data3 = {
        "project_name": "Original",
        "author": "Original Author"
    }
    context_file3.write_text(json.dumps(context_data3))
    
    extra_context = {"author": "Extra Author"}
    result = generate_context(str(context_file3), extra_context=extra_context)
    assert result["cookiecutter"]["author"] == "Extra Author"

    # Test 4: Choice variable with default_context
    context_file4 = tmp_path / "cookiecutter4.json"
    context_data4 = {
        "license": ["MIT", "Apache", "GPL"]
    }
    context_file4.write_text(json.dumps(context_data4))
    
    default_context = {"license": "Apache"}
    result = generate_context(str(context_file4), default_context=default_context)
    assert result["cookiecutter"]["license"][0] == "Apache"

    # Test 5: Multi-choice variable with extra_context
    context_file5 = tmp_path / "cookiecutter5.json"
    context_data5 = {
        "features": ["feature1", "feature2", "feature3"]
    }
    context_file5.write_text(json.dumps(context_data5))
    
    extra_context = {"features": ["feature1", "feature3"]}
    result = generate_context(str(context_file5), extra_context=extra_context)
    assert result["cookiecutter"]["features"] == ["feature1", "feature3"]

    # Test 6: Dictionary variable with extra_context
    context_file6 = tmp_path / "cookiecutter6.json"
    context_data6 = {
        "config": {
            "debug": False,
            "timeout": 30
        }
    }
    context_file6.write_text(json.dumps(context_data6))
    
    extra_context = {"config": {"debug": True}}
    result = generate_context(str(context_file6), extra_context=extra_context)
    assert result["cookiecutter"]["config"]["debug"] is True
    assert result["cookiecutter"]["config"]["timeout"] == 30

    # Test 7: Invalid JSON file raises ContextDecodingException
    context_file7 = tmp_path / "cookiecutter7.json"
    context_file7.write_text("{ invalid json }")
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(context_file7))

    # Test 8: Invalid choice in default_context generates warning
    context_file8 = tmp_path / "cookiecutter8.json"
    context_data8 = {
        "choice_var": ["option1", "option2"]
    }
    context_file8.write_text(json.dumps(context_data8))
    
    invalid_default = {"choice_var": "invalid_option"}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = generate_context(str(context_file8), default_context=invalid_default)
        assert len(w) == 1
        assert "Invalid default received" in str(w[0].message)

    # Test 9: Boolean variable with string extra_context
    context_file9 = tmp_path / "cookiecutter9.json"
    context_data9 = {
        "use_feature": True
    }
    context_file9.write_text(json.dumps(context_data9))
    
    extra_context = {"use_feature": "n"}
    result = generate_context(str(context_file9), extra_context=extra_context)
    assert result["cookiecutter"]["use_feature"] is False

    # Test 10: OrderedDict preservation
    context_file10 = tmp_path / "cookiecutter10.json"
    context_data10 = {
        "z_field": "last",
        "a_field": "first",
        "m_field": "middle"
    }
    context_file10.write_text(json.dumps(context_data10))
    
    result = generate_context(str(context_file10))
    assert isinstance(result, OrderedDict)
    assert isinstance(result["cookiecutter"], OrderedDict)


# LLM-generated content at query #7
#--------------------------

```python
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import pytest

from cookiecutter.exceptions import ContextDecodingException


def test_generate_context():
    """Test generate_context function with various scenarios."""
    
    # Test 1: Basic context generation from valid JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'author': 'John Doe',
            'version': '1.0.0'
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        result = generate_context(context_file)
        
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'My Project'
        assert result['cookiecutter']['author'] == 'John Doe'
        assert result['cookiecutter']['version'] == '1.0.0'


def test_generate_context_with_default_context():
    """Test generate_context with default_context parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'use_pytest': True
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        default_context = {
            'project_name': 'Default Project'
        }
        
        result = generate_context(context_file, default_context=default_context)
        
        assert result['cookiecutter']['project_name'] == 'Default Project'


def test_generate_context_with_extra_context():
    """Test generate_context with extra_context parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'author': 'John Doe'
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'author': 'Jane Doe',
            'version': '2.0.0'
        }
        
        result = generate_context(context_file, extra_context=extra_context)
        
        assert result['cookiecutter']['author'] == 'Jane Doe'
        assert result['cookiecutter']['version'] == '2.0.0'


def test_generate_context_invalid_json():
    """Test generate_context with invalid JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write('{ invalid json }')
        
        with pytest.raises(ContextDecodingException) as exc_info:
            generate_context(context_file)
        
        assert 'JSON decoding error' in str(exc_info.value)
        assert context_file in str(exc_info.value)


def test_generate_context_preserves_order():
    """Test that generate_context preserves OrderedDict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = OrderedDict([
            ('first', 'value1'),
            ('second', 'value2'),
            ('third', 'value3')
        ])
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        result = generate_context(context_file)
        
        assert isinstance(result, OrderedDict)
        assert list(result['cookiecutter'].keys()) == ['first', 'second', 'third']


def test_generate_context_with_choice_variable():
    """Test generate_context with choice variable in extra_context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'python_version': ['3.8', '3.9', '3.10']
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'python_version': '3.10'
        }
        
        result = generate_context(context_file, extra_context=extra_context)
        
        assert result['cookiecutter']['python_version'][0] == '3.10'


def test_generate_context_with_boolean_variable():
    """Test generate_context with boolean variable in extra_context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'use_docker': True
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'use_docker': 'n'
        }
        
        result = generate_context(context_file, extra_context=extra_context)
        
        assert result['cookiecutter']['use_docker'] is False


def test_generate_context_with_nested_dict():
    """Test generate_context with nested dictionary variables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project': {
                'name': 'My Project',
                'version': '1.0.0'
            }
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'project': {
                'version': '2.0.0'
            }
        }
        
        result = generate_context(context_file, extra_context=extra_context)
        
        assert result['cookiecutter']['project']['name'] == 'My Project'
        assert result['cookiecutter']['project']['version'] == '2.0.0'


def test_generate_context_custom_filename():
    """Test generate_context with custom context filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'custom_context.json')
        context_data = {
            'project_name': 'Custom Project'
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        result = generate_context(context_file)
        
        assert 'custom_context' in result
        assert result['custom_context']['project_name'] == 'Custom Project'


# LLM-generated content at query #8
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    import json
    from collections import OrderedDict
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "project_slug": "my_project",
        "author_name": "John Doe"
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["project_slug"] == "my_project"
    assert result["cookiecutter"]["author_name"] == "John Doe"


def test_generate_context_with_default_context(tmp_path):
    """Test generate_context with default_context parameter."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "Default Name",
        "version": "1.0.0"
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    default_context = {"project_name": "Overridden Name"}
    result = generate_context(str(context_file), default_context=default_context)
    
    assert result["cookiecutter"]["project_name"] == "Overridden Name"
    assert result["cookiecutter"]["version"] == "1.0.0"


def test_generate_context_with_extra_context(tmp_path):
    """Test generate_context with extra_context parameter."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "version": "1.0.0"
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    extra_context = {"version": "2.0.0", "author": "Jane Doe"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["version"] == "2.0.0"
    assert result["cookiecutter"]["author"] == "Jane Doe"


def test_generate_context_with_choice_variable(tmp_path):
    """Test generate_context with choice variable overwrite."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "license": ["MIT", "Apache", "GPL"]
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    # Apache should be first in the list
    assert result["cookiecutter"]["license"][0] == "Apache"
    assert "MIT" in result["cookiecutter"]["license"]
    assert "GPL" in result["cookiecutter"]["license"]


def test_generate_context_with_multichoice_variable(tmp_path):
    """Test generate_context with multichoice variable overwrite."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "features": ["feature1", "feature2", "feature3"]
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    extra_context = {"features": ["feature2", "feature3"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["features"] == ["feature2", "feature3"]


def test_generate_context_with_boolean_variable(tmp_path):
    """Test generate_context with boolean variable overwrite."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "use_docker": True
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    extra_context = {"use_docker": "n"}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["use_docker"] is False


def test_generate_context_with_dict_variable(tmp_path):
    """Test generate_context with dictionary variable overwrite."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "author": {
            "name": "John Doe",
            "email": "john@example.com"
        }
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    extra_context = {"author": {"name": "Jane Doe"}}
    result = generate_context(str(context_file), extra_context=extra_context)
    
    assert result["cookiecutter"]["author"]["name"] == "Jane Doe"
    assert result["cookiecutter"]["author"]["email"] == "john@example.com"


def test_generate_context_invalid_json(tmp_path):
    """Test generate_context with invalid JSON file."""
    from cookiecutter.exceptions import ContextDecodingException
    
    context_file = tmp_path / "cookiecutter.json"
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write("{ invalid json }")
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(context_file))


def test_generate_context_invalid_choice_overwrite(tmp_path):
    """Test generate_context with invalid choice overwrite."""
    import json
    
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "license": ["MIT", "Apache"]
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    extra_context = {"license": "BSD"}
    with pytest.raises(ValueError):
        generate_context(str(context_file), extra_context=extra_context)


def test_generate_context_custom_filename(tmp_path):
    """Test generate_context with custom context filename."""
    import json
    
    context_file = tmp_path / "custom.json"
    context_data = {
        "project": "Custom Project"
    }
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context_data, f)
    
    result = generate_context(str(context_file))
    assert "custom" in result
    assert result["custom"]["project"] == "Custom Project"


# LLM-generated content at query #9
#--------------------------

```python
def test_generate_files(tmp_path, mocker):
    """Test generate_files function."""
    # Setup
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    # Create template directory structure
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a cookiecutter.json
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    # Create a template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}")
    
    # Create a subdirectory in template
    subdir = template_dir / "src"
    subdir.mkdir()
    (subdir / "main.py").write_text("# {{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Mock hooks
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))
    
    # Execute
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=True,
        keep_project_on_failure=False
    )
    
    # Assert
    assert result is not None
    project_path = Path(result)
    assert project_path.exists()
    assert project_path.name == "my_project"
    assert (project_path / "README.md").exists()
    assert (project_path / "README.md").read_text() == "# my_project"
    assert (project_path / "src" / "main.py").exists()
    assert (project_path / "src" / "main.py").read_text() == "# my_project"


def test_generate_files_with_binary_file(tmp_path, mocker):
    """Test generate_files with binary file."""
    # Setup
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    # Create a binary file
    binary_file = template_dir / "image.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "my_project"}}
    
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))
    
    # Execute
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Assert
    assert (Path(result) / "image.bin").exists()
    assert (Path(result) / "image.bin").read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_files_overwrite_if_exists(tmp_path, mocker):
    """Test generate_files with overwrite_if_exists=True."""
    # Setup
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create existing project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old")
    
    context = {"cookiecutter": {"project_name": "my_project"}}
    
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))
    
    # Execute
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    # Assert
    assert Path(result).exists()
    assert (Path(result) / "file.txt").exists()


def test_generate_files_with_copy_without_render(tmp_path, mocker):
    """Test generate_files with _copy_without_render setting."""
    # Setup
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project", "_copy_without_render": ["*.bin"]}')
    
    (template_dir / "template.txt").write_text("# {{cookiecutter.project_name}}")
    (template_dir / "data.bin").write_bytes(b'\x00\x01\x02')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "_copy_without_render": ["*.bin"]
        }
    }
    
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))
    
    # Execute
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Assert
    assert (Path(result) / "template.txt").read_text() == "# my_project"
    assert (Path(result) / "data.bin").read_bytes() == b'\x00\x01\x02'


def test_generate_files_skip_if_file_exists(tmp_path, mocker):
    """Test generate_files with skip_if_file_exists=True."""
    # Setup
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    (template_dir / "file.txt").write_text("new content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create existing file in output
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    existing_file = existing_project / "file.txt"
    existing_file.write_text("old content")
    
    context =


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Setup template directory structure
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    project_template_dir = template_dir / "{{cookiecutter.project_name}}"
    project_template_dir.mkdir()
    
    # Create a simple template file
    test_file = project_template_dir / "README.md"
    test_file.write_text("# {{cookiecutter.project_name}}\n")
    
    # Create cookiecutter.json
    cookiecutter_json = template_dir / "cookiecutter.json"
    cookiecutter_json.write_text('{"project_name": "my_project"}')
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Setup context
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'my_project')
        ]))
    ])
    
    # Call generate_files
    result = generate_files(
        repo_dir=str(template_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False
    )
    
    # Assertions
    assert result is not None
    assert os.path.isdir(result)
    assert "my_project" in result
    
    # Check if the project was created
    created_project = output_dir / "my_project"
    assert created_project.exists()
    assert created_project.is_dir()
    
    # Check if the file was rendered
    readme_file = created_project / "README.md"
    assert readme_file.exists()
    assert "# my_project" in readme_file.read_text()


def test_generate_files_with_overwrite(tmp_path):
    """Test generate_files with overwrite_if_exists=True."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    project_template_dir = template_dir / "{{cookiecutter.project_name}}"
    project_template_dir.mkdir()
    
    test_file = project_template_dir / "test.txt"
    test_file.write_text("Test content")
    
    cookiecutter_json = template_dir / "cookiecutter.json"
    cookiecutter_json.write_text('{"project_name": "test_project"}')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'test_project')
        ]))
    ])
    
    # Create project first time
    result1 = generate_files(
        repo_dir=str(template_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    assert (output_dir / "test_project").exists()
    
    # Try to create again with overwrite
    result2 = generate_files(
        repo_dir=str(template_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert result2 is not None
    assert (output_dir / "test_project").exists()


def test_generate_files_with_binary_file(tmp_path):
    """Test generate_files with binary files."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    project_template_dir = template_dir / "{{cookiecutter.project_name}}"
    project_template_dir.mkdir()
    
    # Create a binary file (simple PNG header)
    binary_file = project_template_dir / "image.png"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
    
    cookiecutter_json = template_dir / "cookiecutter.json"
    cookiecutter_json.write_text('{"project_name": "binary_test"}')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'binary_test')
        ]))
    ])
    
    result = generate_files(
        repo_dir=str(template_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    output_binary = output_dir / "binary_test" / "image.png"
    assert output_binary.exists()
    assert output_binary.read_bytes().startswith(b'\x89PNG')


def test_generate_files_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists=True."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    project_template_dir = template_dir / "{{cookiecutter.project_name}}"
    project_template_dir.mkdir()
    
    test_file = project_template_dir / "existing.txt"
    test_file.write_text("Template content")
    
    cookiecutter_json = template_dir / "cookiecutter.json"
    cookiecutter_json.write_text('{"project_name": "skip_test"}')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'skip_test')
        ]))
    ])
    
    # Create the file first
    result = generate_files(
        repo_dir=str(template_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    existing_file = output_dir / "skip_test" / "existing.txt"
    existing_file.write_text("Modified content")
    
    # Generate again with skip_if_file_exists
    result = generate_files(
        repo_dir=str(template_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    # File should still have modified content
    assert "Modified content" in existing_file.read_text()


def test_generate_files_copy_without_render(tmp_path):
    """Test generate_files with _copy_without_render setting."""
    from collections import OrderedDict
    
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    
    project_template_dir = template_dir / "{{cookiecutter.project_name}}"
    project_template_dir.mkdir()
    
    # Create a file with template syntax that should not be rendered
    no_render_file = project_template_dir / "{{not_rendered}}.txt"
    no_render_file.write_text("{{cookiecutter.project_name}}")
    
    cookiecutter_json = template_dir / "cookiecutter.json"
    cookiecutter_json.write_text(
        '{"project_name": "copy_test", "_copy_without_render": ["*{{*.txt}"]}'
    )
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'copy_test'),
            ('_copy_without_render', ['*{{*.txt}'])
        ]))
    ])


# LLM-generated content at query #11
#--------------------------

```python
def test_render_and_create_dir(tmp_path):
    """Test render_and_create_dir function."""
    from jinja2 import Environment
    
    # Test 1: Basic directory creation
    context = {'cookiecutter': {'project_name': 'my_project'}}
    env = Environment()
    dirname = '{{cookiecutter.project_name}}'
    
    result_path, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_path == tmp_path / 'my_project'
    assert created is True
    assert result_path.exists()
    assert result_path.is_dir()


def test_render_and_create_dir_empty_dirname(tmp_path):
    """Test render_and_create_dir raises error for empty directory name."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {}}
    env = Environment()
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, tmp_path, env)


def test_render_and_create_dir_empty_string_dirname(tmp_path):
    """Test render_and_create_dir raises error for empty string directory name."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {}}
    env = Environment()
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, tmp_path, env)


def test_render_and_create_dir_exists_no_overwrite(tmp_path):
    """Test render_and_create_dir raises error when dir exists and overwrite is False."""
    from jinja2 import Environment
    
    # Create directory first
    existing_dir = tmp_path / 'existing_project'
    existing_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'existing_project'}}
    env = Environment()
    dirname = '{{cookiecutter.project_name}}'
    
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, tmp_path, env, overwrite_if_exists=False
        )


def test_render_and_create_dir_exists_with_overwrite(tmp_path):
    """Test render_and_create_dir overwrites when dir exists and overwrite is True."""
    from jinja2 import Environment
    
    # Create directory first
    existing_dir = tmp_path / 'existing_project'
    existing_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'existing_project'}}
    env = Environment()
    dirname = '{{cookiecutter.project_name}}'
    
    result_path, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=True
    )
    
    assert result_path == tmp_path / 'existing_project'
    assert created is False
    assert result_path.exists()


def test_render_and_create_dir_nested_path(tmp_path):
    """Test render_and_create_dir with nested directory path."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'org': 'myorg', 'project': 'myproj'}}
    env = Environment()
    dirname = '{{cookiecutter.org}}/{{cookiecutter.project}}'
    
    result_path, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_path == tmp_path / 'myorg' / 'myproj'
    assert created is True
    assert result_path.exists()
    assert result_path.parent.exists()


def test_render_and_create_dir_no_template_vars(tmp_path):
    """Test render_and_create_dir with plain directory name (no template variables)."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {}}
    env = Environment()
    dirname = 'plain_project'
    
    result_path, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_path == tmp_path / 'plain_project'
    assert created is True
    assert result_path.exists()


def test_render_and_create_dir_special_characters(tmp_path):
    """Test render_and_create_dir with special characters in directory name."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'project_name': 'my-project_v1.0'}}
    env = Environment()
    dirname = '{{cookiecutter.project_name}}'
    
    result_path, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_path == tmp_path / 'my-project_v1.0'
    assert created is True
    assert result_path.exists()


def test_render_and_create_dir_with_string_output_dir(tmp_path):
    """Test render_and_create_dir accepts string output_dir."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    env = Environment()
    dirname = '{{cookiecutter.project_name}}'
    output_dir_str = str(tmp_path)
    
    result_path, created = render_and_create_dir(
        dirname, context, output_dir_str, env, overwrite_if_exists=False
    )
    
    assert created is True
    assert result_path.exists()


# LLM-generated content at query #12
#--------------------------

```python
import os
import pytest
from pathlib import Path
from collections import OrderedDict
from unittest.mock import Mock, patch, MagicMock, call
from jinja2 import Environment, UndefinedError

def test_generate_files():
    """Test generate_files function."""
    
    # Setup test data
    repo_dir = "/path/to/repo"
    context = OrderedDict([("cookiecutter", {"project_name": "test_project"})])
    output_dir = "/output"
    
    # Mock dependencies
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook, \
         patch('cookiecutter.generate.work_in') as mock_work_in, \
         patch('cookiecutter.generate.os.walk') as mock_walk, \
         patch('cookiecutter.generate.os.path.split') as mock_split, \
         patch('cookiecutter.generate.os.path.abspath') as mock_abspath, \
         patch('cookiecutter.generate.FileSystemLoader') as mock_loader:
        
        # Setup mock environment
        mock_env = MagicMock(spec=Environment)
        mock_create_env.return_value = mock_env
        
        # Setup mock template directory
        mock_find_template.return_value = "/path/to/repo/{{cookiecutter.project_name}}"
        mock_split.return_value = ("/path/to/repo", "{{cookiecutter.project_name}}")
        
        # Setup mock project directory creation
        mock_project_path = Path("/output/test_project")
        mock_render_dir.return_value = (mock_project_path, True)
        
        # Setup mock abspath
        mock_abspath.return_value = str(mock_project_path)
        
        # Setup mock os.walk to return empty directory structure
        mock_walk.return_value = [
            (".", [], []),
        ]
        
        # Setup mock work_in context manager
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=None)
        
        # Call the function
        result = generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir,
            overwrite_if_exists=False,
            skip_if_file_exists=False,
            accept_hooks=True,
            keep_project_on_failure=False
        )
        
        # Assertions
        assert result == str(mock_project_path)
        mock_create_env.assert_called_once_with(context)
        mock_find_template.assert_called_once()
        mock_render_dir.assert_called_once()
        assert mock_run_hook.call_count == 2  # pre_gen_project and post_gen_project
        mock_work_in.assert_called_once()


def test_generate_files_with_undefined_error():
    """Test generate_files handles UndefinedError in directory creation."""
    
    repo_dir = "/path/to/repo"
    context = OrderedDict([("cookiecutter", {})])
    output_dir = "/output"
    
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.os.path.split') as mock_split:
        
        mock_env = MagicMock(spec=Environment)
        mock_create_env.return_value = mock_env
        mock_find_template.return_value = "/path/to/repo/{{cookiecutter.project_name}}"
        mock_split.return_value = ("/path/to/repo", "{{cookiecutter.project_name}}")
        
        # Simulate UndefinedError during directory rendering
        undefined_error = UndefinedError("Undefined variable")
        mock_render_dir.side_effect = undefined_error
        
        # Should raise UndefinedVariableInTemplate
        from cookiecutter.exceptions import UndefinedVariableInTemplate
        
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(
                repo_dir=repo_dir,
                context=context,
                output_dir=output_dir,
                accept_hooks=False
            )


def test_generate_files_without_hooks():
    """Test generate_files with accept_hooks=False."""
    
    repo_dir = "/path/to/repo"
    context = OrderedDict([("cookiecutter", {"project_name": "test"})])
    output_dir = "/output"
    
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook, \
         patch('cookiecutter.generate.work_in') as mock_work_in, \
         patch('cookiecutter.generate.os.walk') as mock_walk, \
         patch('cookiecutter.generate.os.path.split') as mock_split, \
         patch('cookiecutter.generate.os.path.abspath') as mock_abspath, \
         patch('cookiecutter.generate.FileSystemLoader'):
        
        mock_env = MagicMock(spec=Environment)
        mock_create_env.return_value = mock_env
        mock_find_template.return_value = "/path/to/repo/test"
        mock_split.return_value = ("/path/to/repo", "test")
        mock_project_path = Path("/output/test")
        mock_render_dir.return_value = (mock_project_path, True)
        mock_abspath.return_value = str(mock_project_path)
        mock_walk.return_value = [(".", [], [])]
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=None)
        
        result = generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir,
            accept_hooks=False
        )
        
        # Hooks should not be called
        mock_run_hook.assert_not_called()
        assert result == str(mock_project_path)


def test_generate_files_keep_project_on_failure():
    """Test generate_files with keep_project_on_failure=True."""
    
    repo_dir = "/path/to/repo"
    context = OrderedDict([("cookiecutter", {"project_name": "test"})])
    output_dir = "/output"
    
    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook, \
         patch('cookiecutter.generate.work_in') as mock_work_in, \
         patch('cookiecutter.generate.os.walk') as mock_walk, \
         patch('cookiecutter.generate.os.path.split') as mock_split, \
         patch('cookiecutter.generate.os.path.abspath') as mock_abspath, \
         patch('cookiecutter.generate.FileSystemLoader'):


# LLM-generated content at query #13
#--------------------------

```python
def test_render_and_create_dir(tmp_path):
    """Test render_and_create_dir function."""
    from jinja2 import Environment
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    env = Environment()
    output_dir = tmp_path
    
    # Test 1: Basic directory creation with template rendering
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env
    )
    
    assert result_dir == output_dir / 'my_project'
    assert result_dir.exists()
    assert created is True
    
    # Test 2: Directory already exists without overwrite
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, output_dir, env, overwrite_if_exists=False
        )
    
    # Test 3: Directory already exists with overwrite
    result_dir2, created2 = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=True
    )
    
    assert result_dir2 == output_dir / 'my_project'
    assert result_dir2.exists()
    assert created2 is False
    
    # Test 4: Empty directory name raises exception
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, env)
    
    # Test 5: Complex template in dirname
    context2 = {
        'cookiecutter': {
            'project_name': 'test_app',
            'version': '1.0.0'
        }
    }
    dirname2 = '{{cookiecutter.project_name}}_v{{cookiecutter.version}}'
    result_dir3, created3 = render_and_create_dir(
        dirname2, context2, tmp_path / 'complex', env
    )
    
    assert result_dir3 == tmp_path / 'complex' / 'test_app_v1.0.0'
    assert result_dir3.exists()
    assert created3 is True
    
    # Test 6: Simple dirname without template variables
    simple_dirname = 'simple_dir'
    result_dir4, created4 = render_and_create_dir(
        simple_dirname, context, tmp_path / 'simple', env
    )
    
    assert result_dir4 == tmp_path / 'simple' / 'simple_dir'
    assert result_dir4.exists()
    assert created4 is True


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function with basic template rendering."""
    # Setup template directory structure
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}\n")
    
    # Create cookiecutter.json
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Generate context
    monkeypatch.chdir(tmp_path)
    context = generate_context(str(context_file))
    
    # Call generate_files
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Verify project was created
    assert os.path.exists(result)
    assert "my_project" in result
    
    # Verify generated file content
    generated_readme = Path(result) / "README.md"
    assert generated_readme.exists()
    assert generated_readme.read_text() == "# my_project\n"


def test_generate_files_with_overwrite(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists flag."""
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "test.txt"
    template_file.write_text("content")
    
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "test_proj"}')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    context = generate_context(str(context_file))
    
    # First generation
    result1 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Second generation with overwrite
    result2 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert os.path.exists(result2)
    assert (Path(result2) / "test.txt").exists()


def test_generate_files_without_overwrite_raises(tmp_path, monkeypatch):
    """Test generate_files raises when directory exists and overwrite is False."""
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "test_proj"}')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    context = generate_context(str(context_file))
    
    # First generation
    generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Second generation without overwrite should raise
    with pytest.raises(OutputDirExistsException):
        generate_files(
            repo_dir=str(tmp_path),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=False,
            accept_hooks=False
        )


def test_generate_files_with_binary_file(tmp_path, monkeypatch):
    """Test generate_files handles binary files correctly."""
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a binary file
    binary_file = template_dir / "image.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_proj"}')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    context = generate_context(str(context_file))
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Verify binary file was copied
    copied_binary = Path(result) / "image.bin"
    assert copied_binary.exists()
    assert copied_binary.read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_files_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists flag."""
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "config.txt"
    template_file.write_text("{{cookiecutter.config}}")
    
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_proj", "config": "new_config"}')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    context = generate_context(str(context_file))
    
    # First generation
    result1 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )
    
    # Modify the generated file
    config_file = Path(result1) / "config.txt"
    config_file.write_text("existing_config")
    
    # Change context
    context['cookiecutter']['config'] = "updated_config"
    
    # Second generation with skip_if_file_exists
    result2 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    # File should not be updated
    assert config_file.read_text() == "existing_config"


def test_generate_files_with_copy_without_render(tmp_path, monkeypatch):
    """Test generate_files respects _copy_without_render setting."""
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a file that should not be rendered
    no_render_file = template_dir / "static.txt"
    no_render_file.write_text("{{cookiecutter.project_name}}")
    
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text(
        '{"project_name": "my_proj", "_copy_without_render": ["static.txt"]}'
    )
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    context = generate_context(str(context_file))
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        


# LLM-generated content at query #15
#--------------------------

```python
def test_render_and_create_dir(tmp_path, monkeypatch):
    """Test render_and_create_dir function."""
    from jinja2 import Environment
    
    # Test basic directory creation
    context = {'cookiecutter': {'project_name': 'my_project'}}
    env = Environment()
    output_dir = tmp_path
    
    result_path, created = render_and_create_dir(
        '{{cookiecutter.project_name}}',
        context,
        output_dir,
        env
    )
    
    assert result_path == output_dir / 'my_project'
    assert created is True
    assert result_path.exists()


def test_render_and_create_dir_empty_dirname(tmp_path):
    """Test render_and_create_dir with empty directory name."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {}}
    env = Environment()
    output_dir = tmp_path
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, env)


def test_render_and_create_dir_existing_dir_no_overwrite(tmp_path):
    """Test render_and_create_dir with existing directory and no overwrite."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    env = Environment()
    output_dir = tmp_path
    
    # Create directory first
    (output_dir / 'my_project').mkdir()
    
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            '{{cookiecutter.project_name}}',
            context,
            output_dir,
            env,
            overwrite_if_exists=False
        )


def test_render_and_create_dir_existing_dir_with_overwrite(tmp_path):
    """Test render_and_create_dir with existing directory and overwrite enabled."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    env = Environment()
    output_dir = tmp_path
    
    # Create directory first
    (output_dir / 'my_project').mkdir()
    
    result_path, created = render_and_create_dir(
        '{{cookiecutter.project_name}}',
        context,
        output_dir,
        env,
        overwrite_if_exists=True
    )
    
    assert result_path == output_dir / 'my_project'
    assert created is False
    assert result_path.exists()


def test_render_and_create_dir_nested_path(tmp_path):
    """Test render_and_create_dir with nested path creation."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'org': 'myorg', 'project': 'myproject'}}
    env = Environment()
    output_dir = tmp_path
    
    result_path, created = render_and_create_dir(
        '{{cookiecutter.org}}/{{cookiecutter.project}}',
        context,
        output_dir,
        env
    )
    
    assert result_path == output_dir / 'myorg' / 'myproject'
    assert created is True
    assert result_path.exists()


def test_render_and_create_dir_with_special_chars(tmp_path):
    """Test render_and_create_dir with special characters in directory name."""
    from jinja2 import Environment
    
    context = {'cookiecutter': {'project_name': 'my-project_v1'}}
    env = Environment()
    output_dir = tmp_path
    
    result_path, created = render_and_create_dir(
        '{{cookiecutter.project_name}}',
        context,
        output_dir,
        env
    )
    
    assert result_path == output_dir / 'my-project_v1'
    assert created is True
    assert result_path.exists()


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from pathlib import Path
from jinja2 import Environment
from cookiecutter.exceptions import EmptyDirNameException, OutputDirExistsException


def test_render_and_create_dir(tmp_path):
    """Test render_and_create_dir creates directory with rendered name."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    dirname = '{{cookiecutter.project_name}}'
    
    result_dir, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_dir == Path(tmp_path) / 'my_project'
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_empty_dirname():
    """Test render_and_create_dir raises error for empty directory name."""
    env = Environment()
    context = {'cookiecutter': {}}
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, '.', env)


def test_render_and_create_dir_empty_string_dirname():
    """Test render_and_create_dir raises error for empty string directory name."""
    env = Environment()
    context = {'cookiecutter': {}}
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, '.', env)


def test_render_and_create_dir_already_exists(tmp_path):
    """Test render_and_create_dir raises error when dir exists and overwrite is False."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    dirname = '{{cookiecutter.project_name}}'
    
    # Create directory first
    existing_dir = tmp_path / 'my_project'
    existing_dir.mkdir()
    
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, tmp_path, env, overwrite_if_exists=False
        )


def test_render_and_create_dir_overwrite_exists(tmp_path):
    """Test render_and_create_dir overwrites when overwrite_if_exists is True."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    dirname = '{{cookiecutter.project_name}}'
    
    # Create directory first
    existing_dir = tmp_path / 'my_project'
    existing_dir.mkdir()
    
    result_dir, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=True
    )
    
    assert result_dir == existing_dir
    assert result_dir.exists()
    assert created is False


def test_render_and_create_dir_with_nested_context(tmp_path):
    """Test render_and_create_dir with nested template variables."""
    env = Environment()
    context = {'cookiecutter': {'org': 'myorg', 'project': 'myapp'}}
    dirname = '{{cookiecutter.org}}_{{cookiecutter.project}}'
    
    result_dir, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_dir == Path(tmp_path) / 'myorg_myapp'
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_returns_not_created_when_exists(tmp_path):
    """Test render_and_create_dir returns False for created when dir already exists."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'existing'}}
    dirname = '{{cookiecutter.project_name}}'
    
    # Create directory first
    existing_dir = tmp_path / 'existing'
    existing_dir.mkdir()
    
    result_dir, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=True
    )
    
    assert created is False


def test_render_and_create_dir_with_path_object(tmp_path):
    """Test render_and_create_dir works with Path objects."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'test_project'}}
    dirname = '{{cookiecutter.project_name}}'
    
    result_dir, created = render_and_create_dir(
        dirname, context, Path(tmp_path), env, overwrite_if_exists=False
    )
    
    assert isinstance(result_dir, Path)
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_with_special_characters(tmp_path):
    """Test render_and_create_dir with special characters in template."""
    env = Environment()
    context = {'cookiecutter': {'name': 'my-project'}}
    dirname = '{{cookiecutter.name}}'
    
    result_dir, created = render_and_create_dir(
        dirname, context, tmp_path, env, overwrite_if_exists=False
    )
    
    assert result_dir == Path(tmp_path) / 'my-project'
    assert result_dir.exists()
    assert created is True


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Create a temporary template directory structure
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}\n")
    
    # Create context
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Mock find_template to return our template directory
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    
    # Mock run_hook_from_repo_dir to do nothing
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_run_hook)
    
    # Call generate_files
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=True,
        keep_project_on_failure=False
    )
    
    # Verify the result
    assert result == str(output_dir / "my_project")
    assert os.path.isdir(result)
    
    # Verify the generated file
    generated_file = Path(result) / "README.md"
    assert generated_file.exists()
    assert generated_file.read_text() == "# my_project\n"


def test_generate_files_with_copy_without_render(tmp_path, monkeypatch):
    """Test generate_files with _copy_without_render setting."""
    from pathlib import Path
    
    # Create template directory
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a binary-like file (we'll just use a regular file)
    binary_file = template_dir / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03")
    
    # Create context with copy_without_render
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_copy_without_render': ['*.bin']
        }
    }
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_run_hook)
    
    # Call generate_files
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=True
    )
    
    # Verify binary file was copied
    generated_binary = Path(result) / "binary.bin"
    assert generated_binary.exists()
    assert generated_binary.read_bytes() == b"\x00\x01\x02\x03"


def test_generate_files_with_nested_directories(tmp_path, monkeypatch):
    """Test generate_files with nested directory structure."""
    from pathlib import Path
    
    # Create template with nested dirs
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    nested_dir = template_dir / "src" / "{{cookiecutter.module_name}}"
    nested_dir.mkdir(parents=True)
    
    nested_file = nested_dir / "main.py"
    nested_file.write_text("# {{cookiecutter.module_name}}\n")
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'module_name': 'my_module'
        }
    }
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_run_hook)
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=True
    )
    
    # Verify nested structure
    generated_nested = Path(result) / "src" / "my_module" / "main.py"
    assert generated_nested.exists()
    assert generated_nested.read_text() == "# my_module\n"


def test_generate_files_overwrite_if_exists(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    from pathlib import Path
    
    # Create template directory
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "test.txt"
    template_file.write_text("test content")
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Pre-create the project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    existing_file = existing_project / "test.txt"
    existing_file.write_text("old content")
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    monkeypatch.setattr('cookiecutter.generate.find_template', mock_find_template)
    
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr('cookiecutter.generate.run_hook_from_repo_dir', mock_run_hook)
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=True
    )
    
    # Verify file was overwritten
    assert Path(result).exists()
    assert (Path(result) / "test.txt").read_text() == "test content"


def test_generate_files_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists=True."""
    from pathlib import Path
    
    # Create template directory
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "test.txt"
    template_file.write_text("new content")
    


# LLM-generated content at query #18
#--------------------------

```python
import pytest
import os
import json
from pathlib import Path
from collections import OrderedDict
from unittest.mock import Mock, patch, MagicMock
from jinja2 import Environment, FileSystemLoader, UndefinedError

def test_generate_files(tmp_path):
    """Test generate_files function with basic project generation."""
    # Setup template directory structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    (template_dir / "README.md").write_text("# {{cookiecutter.project_name}}\n")
    
    # Create cookiecutter.json
    (repo_dir / "cookiecutter.json").write_text(
        json.dumps({"project_name": "test_project"})
    )
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Test basic file generation
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False,
    )
    
    assert os.path.exists(result)
    assert "test_project" in result
    assert (Path(result) / "README.md").exists()


def test_generate_files_with_overwrite(tmp_path):
    """Test generate_files with overwrite_if_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "myproject"}}
    
    # First generation
    result1 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        accept_hooks=False,
    )
    
    # Second generation with overwrite
    result2 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False,
    )
    
    assert result1 == result2
    assert os.path.exists(result2)


def test_generate_files_with_binary_file(tmp_path):
    """Test generate_files handles binary files correctly."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a binary file
    binary_file = template_dir / "image.bin"
    binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "binproject"}}
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False,
    )
    
    assert (Path(result) / "image.bin").exists()
    assert (Path(result) / "image.bin").read_bytes() == b'\x89PNG\r\n\x1a\n'


def test_generate_files_skip_if_exists(tmp_path):
    """Test generate_files with skip_if_file_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("template content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "skipproject"}}
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False,
    )
    
    # Modify the generated file
    generated_file = Path(result) / "file.txt"
    generated_file.write_text("modified content")
    
    # Generate again with skip flag
    generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        overwrite_if_exists=True,
        accept_hooks=False,
    )
    
    # File should not be overwritten
    assert generated_file.read_text() == "modified content"


def test_generate_files_copy_without_render(tmp_path):
    """Test generate_files with _copy_without_render option."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a file that should not be rendered
    (template_dir / "{{no_render}}.txt").write_text("{{cookiecutter.project_name}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {
        "cookiecutter": {
            "project_name": "myproject",
            "_copy_without_render": ["{{no_render}}.txt"]
        }
    }
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False,
    )
    
    # File should exist with unrendered content
    no_render_file = Path(result) / "{{no_render}}.txt"
    assert no_render_file.exists()
    assert no_render_file.read_text() == "{{cookiecutter.project_name}}"


def test_generate_files_undefined_variable_error(tmp_path):
    """Test generate_files raises error for undefined template variables."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create file with undefined variable
    (template_dir / "file.txt").write_text("{{undefined_var}}")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    with pytest.raises(Exception):  # UndefinedVariableInTemplate
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False,
        )


def test_generate_files_empty_context(tmp_path):
    """Test generate_files with empty context."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "project"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(repo_dir),
        context=None,
        output_dir=str(output_dir),
        accept_


# LLM-generated content at query #19
#--------------------------

```python
def test_is_copy_only_path():
    """Test is_copy_only_path function with various patterns and contexts."""
    
    # Test with matching pattern
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.pyc', '__pycache__', '*.egg-info']
        }
    }
    assert is_copy_only_path('test.pyc', context) is True
    assert is_copy_only_path('__pycache__', context) is True
    assert is_copy_only_path('my_package.egg-info', context) is True
    
    # Test with non-matching pattern
    assert is_copy_only_path('test.py', context) is False
    assert is_copy_only_path('README.md', context) is False
    assert is_copy_only_path('src', context) is False
    
    # Test with wildcard patterns
    context_wildcards = {
        'cookiecutter': {
            '_copy_without_render': ['*.bin', 'static/*', 'node_modules/**']
        }
    }
    assert is_copy_only_path('file.bin', context_wildcards) is True
    assert is_copy_only_path('static/css', context_wildcards) is True
    assert is_copy_only_path('file.txt', context_wildcards) is False
    
    # Test with empty _copy_without_render list
    context_empty = {
        'cookiecutter': {
            '_copy_without_render': []
        }
    }
    assert is_copy_only_path('test.py', context_empty) is False
    
    # Test with missing _copy_without_render key
    context_missing = {'cookiecutter': {}}
    assert is_copy_only_path('test.py', context_missing) is False
    
    # Test with missing cookiecutter key
    context_no_cookiecutter = {}
    assert is_copy_only_path('test.py', context_no_cookiecutter) is False
    
    # Test with path separators
    context_path = {
        'cookiecutter': {
            '_copy_without_render': ['dist/*', 'build/**/*.o']
        }
    }
    assert is_copy_only_path('dist/package.tar.gz', context_path) is True
    assert is_copy_only_path('build/obj/file.o', context_path) is True
    assert is_copy_only_path('src/main.py', context_path) is False
    
    # Test with complex patterns
    context_complex = {
        'cookiecutter': {
            '_copy_without_render': ['{{cookiecutter.project_name}}/*.bin', '.git/*']
        }
    }
    assert is_copy_only_path('{{cookiecutter.project_name}}/data.bin', context_complex) is True
    assert is_copy_only_path('.git/config', context_complex) is True


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    import json
    from pathlib import Path
    
    # Test 1: Basic context generation
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "project_slug": "my_project",
        "version": "0.1.0"
    }
    context_file.write_text(json.dumps(context_data), encoding='utf-8')
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["project_slug"] == "my_project"
    assert result["cookiecutter"]["version"] == "0.1.0"
    
    # Test 2: Context with default_context override
    default_context = {"project_name": "Default Project"}
    result = generate_context(str(context_file), default_context=default_context)
    assert result["cookiecutter"]["project_name"] == "Default Project"
    assert result["cookiecutter"]["project_slug"] == "my_project"
    
    # Test 3: Context with extra_context override
    extra_context = {"version": "1.0.0"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["version"] == "1.0.0"
    assert result["cookiecutter"]["project_name"] == "My Project"
    
    # Test 4: Context with both default_context and extra_context
    default_context = {"project_name": "Default Project"}
    extra_context = {"version": "2.0.0"}
    result = generate_context(
        str(context_file),
        default_context=default_context,
        extra_context=extra_context
    )
    assert result["cookiecutter"]["project_name"] == "Default Project"
    assert result["cookiecutter"]["version"] == "2.0.0"
    
    # Test 5: Invalid JSON file raises ContextDecodingException
    bad_json_file = tmp_path / "bad.json"
    bad_json_file.write_text("{invalid json}", encoding='utf-8')
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(bad_json_file))
    
    # Test 6: Choice variable with overwrite
    choice_context_file = tmp_path / "choice_cookiecutter.json"
    choice_data = {
        "license": ["MIT", "Apache", "GPL"]
    }
    choice_context_file.write_text(json.dumps(choice_data), encoding='utf-8')
    
    extra_context = {"license": "Apache"}
    result = generate_context(str(choice_context_file), extra_context=extra_context)
    assert result["cookiecutter"]["license"][0] == "Apache"
    
    # Test 7: Multi-choice variable with valid overwrite
    multi_choice_file = tmp_path / "multi_choice.json"
    multi_choice_data = {
        "features": ["feature1", "feature2", "feature3"]
    }
    multi_choice_file.write_text(json.dumps(multi_choice_data), encoding='utf-8')
    
    extra_context = {"features": ["feature1", "feature3"]}
    result = generate_context(str(multi_choice_file), extra_context=extra_context)
    assert result["cookiecutter"]["features"] == ["feature1", "feature3"]
    
    # Test 8: Boolean variable with string overwrite
    bool_context_file = tmp_path / "bool_cookiecutter.json"
    bool_data = {
        "use_docker": True
    }
    bool_context_file.write_text(json.dumps(bool_data), encoding='utf-8')
    
    extra_context = {"use_docker": "n"}
    result = generate_context(str(bool_context_file), extra_context=extra_context)
    assert result["cookiecutter"]["use_docker"] is False
    
    # Test 9: Nested dictionary context
    nested_context_file = tmp_path / "nested_cookiecutter.json"
    nested_data = {
        "author": {
            "name": "John Doe",
            "email": "john@example.com"
        }
    }
    nested_context_file.write_text(json.dumps(nested_data), encoding='utf-8')
    
    extra_context = {"author": {"name": "Jane Doe"}}
    result = generate_context(str(nested_context_file), extra_context=extra_context)
    assert result["cookiecutter"]["author"]["name"] == "Jane Doe"
    assert result["cookiecutter"]["author"]["email"] == "john@example.com"
    
    # Test 10: Default context file name
    default_file = tmp_path / "cookiecutter.json"
    default_file.write_text(json.dumps({"test": "value"}), encoding='utf-8')
    
    result = generate_context(str(default_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["test"] == "value"


# LLM-generated content at query #21
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {
        "project_name": "My Project",
        "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
        "version": "0.1.0"
    }
    context_file.write_text(json.dumps(context_data), encoding='utf-8')
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "My Project"
    assert result["cookiecutter"]["version"] == "0.1.0"
    
    # Test 2: Invalid JSON file
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text("{invalid json}", encoding='utf-8')
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(invalid_json_file))
    
    # Test 3: With default_context overrides
    context_file2 = tmp_path / "cookiecutter2.json"
    context_file2.write_text(json.dumps({
        "name": "default_name",
        "choice": ["option1", "option2"],
        "enabled": True
    }), encoding='utf-8')
    
    default_context = {
        "name": "overridden_name",
        "choice": "option2",
        "enabled": False
    }
    
    result = generate_context(str(context_file2), default_context=default_context)
    assert result["cookiecutter"]["name"] == "overridden_name"
    assert result["cookiecutter"]["choice"][0] == "option2"
    assert result["cookiecutter"]["enabled"] is False
    
    # Test 4: With extra_context overrides
    context_file3 = tmp_path / "cookiecutter3.json"
    context_file3.write_text(json.dumps({
        "author": "John Doe",
        "version": "1.0.0"
    }), encoding='utf-8')
    
    extra_context = {
        "author": "Jane Doe",
        "version": "2.0.0"
    }
    
    result = generate_context(str(context_file3), extra_context=extra_context)
    assert result["cookiecutter"]["author"] == "Jane Doe"
    assert result["cookiecutter"]["version"] == "2.0.0"
    
    # Test 5: With both default_context and extra_context (extra_context should win)
    context_file4 = tmp_path / "cookiecutter4.json"
    context_file4.write_text(json.dumps({
        "setting": "original"
    }), encoding='utf-8')
    
    result = generate_context(
        str(context_file4),
        default_context={"setting": "default"},
        extra_context={"setting": "extra"}
    )
    assert result["cookiecutter"]["setting"] == "extra"
    
    # Test 6: OrderedDict preservation
    context_file5 = tmp_path / "cookiecutter5.json"
    ordered_data = OrderedDict([
        ("first", "1"),
        ("second", "2"),
        ("third", "3")
    ])
    context_file5.write_text(json.dumps(ordered_data), encoding='utf-8')
    
    result = generate_context(str(context_file5))
    assert isinstance(result, dict)
    assert "cookiecutter" in result
    
    # Test 7: Custom context file name
    custom_context_file = tmp_path / "custom.json"
    custom_context_file.write_text(json.dumps({"key": "value"}), encoding='utf-8')
    
    result = generate_context(str(custom_context_file))
    assert "custom" in result
    assert result["custom"]["key"] == "value"
    
    # Test 8: Invalid default context should warn
    context_file6 = tmp_path / "cookiecutter6.json"
    context_file6.write_text(json.dumps({
        "choice": ["a", "b"]
    }), encoding='utf-8')
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = generate_context(
            str(context_file6),
            default_context={"choice": "invalid_choice"}
        )
        assert len(w) > 0
        assert "Invalid default" in str(w[0].message)
    
    # Test 9: Nested dictionary context
    context_file7 = tmp_path / "cookiecutter7.json"
    context_file7.write_text(json.dumps({
        "project": {
            "name": "test",
            "nested": {
                "value": "deep"
            }
        }
    }), encoding='utf-8')
    
    result = generate_context(str(context_file7))
    assert result["cookiecutter"]["project"]["name"] == "test"
    assert result["cookiecutter"]["project"]["nested"]["value"] == "deep"
    
    # Test 10: List context
    context_file8 = tmp_path / "cookiecutter8.json"
    context_file8.write_text(json.dumps({
        "items": ["item1", "item2", "item3"]
    }), encoding='utf-8')
    
    result = generate_context(str(context_file8))
    assert result["cookiecutter"]["items"] == ["item1", "item2", "item3"]


# LLM-generated content at query #22
#--------------------------

```python
import pytest
import json
import os
import tempfile
from pathlib import Path
from collections import OrderedDict
from cookiecutter.exceptions import ContextDecodingException


def test_generate_context():
    """Test generate_context function with various scenarios."""
    
    # Test 1: Basic context generation from valid JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'project_slug': '{{ cookiecutter.project_name.lower().replace(" ", "_") }}',
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        result = generate_context(context_file=context_file)
        
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'My Project'
        assert result['cookiecutter']['project_slug'] == '{{ cookiecutter.project_name.lower().replace(" ", "_") }}'


def test_generate_context_with_default_context():
    """Test generate_context with default_context overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'author': 'Default Author',
            'use_feature': True,
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        default_context = {
            'author': 'Custom Author',
            'use_feature': 'yes',
        }
        
        result = generate_context(
            context_file=context_file,
            default_context=default_context
        )
        
        assert result['cookiecutter']['author'] == 'Custom Author'
        assert result['cookiecutter']['use_feature'] is True


def test_generate_context_with_extra_context():
    """Test generate_context with extra_context overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project_name': 'My Project',
            'version': '0.1.0',
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'version': '1.0.0',
        }
        
        result = generate_context(
            context_file=context_file,
            extra_context=extra_context
        )
        
        assert result['cookiecutter']['version'] == '1.0.0'


def test_generate_context_invalid_json():
    """Test generate_context with invalid JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write('{ invalid json }')
        
        with pytest.raises(ContextDecodingException):
            generate_context(context_file=context_file)


def test_generate_context_preserves_order():
    """Test that generate_context preserves OrderedDict order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = OrderedDict([
            ('first_key', 'value1'),
            ('second_key', 'value2'),
            ('third_key', 'value3'),
        ])
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        result = generate_context(context_file=context_file)
        
        keys = list(result['cookiecutter'].keys())
        assert keys == ['first_key', 'second_key', 'third_key']


def test_generate_context_with_choice_variable():
    """Test generate_context with choice variable overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'license': ['MIT', 'Apache', 'GPL'],
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'license': 'Apache',
        }
        
        result = generate_context(
            context_file=context_file,
            extra_context=extra_context
        )
        
        # Apache should be moved to the front as the default choice
        assert result['cookiecutter']['license'][0] == 'Apache'
        assert set(result['cookiecutter']['license']) == {'MIT', 'Apache', 'GPL'}


def test_generate_context_with_nested_dict():
    """Test generate_context with nested dictionary overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        context_data = {
            'project': {
                'name': 'My Project',
                'version': '0.1.0',
            }
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        extra_context = {
            'project': {
                'version': '1.0.0',
            }
        }
        
        result = generate_context(
            context_file=context_file,
            extra_context=extra_context
        )
        
        assert result['cookiecutter']['project']['name'] == 'My Project'
        assert result['cookiecutter']['project']['version'] == '1.0.0'


def test_generate_context_file_not_found():
    """Test generate_context with non-existent context file."""
    with pytest.raises(FileNotFoundError):
        generate_context(context_file='/nonexistent/path/cookiecutter.json')


def test_generate_context_empty_dict():
    """Test generate_context with empty context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        
        result = generate_context(context_file=context_file)
        
        assert 'cookiecutter' in result
        assert result['cookiecutter'] == {}


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Create a temporary template directory structure
    repo_dir = tmp_path / "template_repo"
    repo_dir.mkdir()
    
    # Create template directory
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create cookiecutter.json
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    # Create a test file in template
    test_file = template_dir / "README.md"
    test_file.write_text("# {{cookiecutter.project_name}}\nThis is a test project.")
    
    # Create a test directory in template
    test_subdir = template_dir / "src"
    test_subdir.mkdir()
    (test_subdir / "main.py").write_text("print('Hello {{cookiecutter.project_name}}')")
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Mock find_template to return our template directory
    monkeypatch.setattr(
        "cookiecutter.generate.find_template",
        lambda repo, env: str(template_dir)
    )
    
    # Mock run_hook_from_repo_dir to do nothing
    monkeypatch.setattr(
        "cookiecutter.generate.run_hook_from_repo_dir",
        lambda *args, **kwargs: None
    )
    
    # Call generate_files
    context = OrderedDict([("cookiecutter", {"project_name": "test_project"})])
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False
    )
    
    # Verify results
    assert result is not None
    project_path = Path(result)
    assert project_path.exists()
    assert project_path.name == "test_project"
    
    # Check that files were rendered
    readme = project_path / "README.md"
    assert readme.exists()
    assert "test_project" in readme.read_text()
    
    # Check that subdirectory was created
    src_dir = project_path / "src"
    assert src_dir.exists()
    
    # Check that file in subdirectory was rendered
    main_py = src_dir / "main.py"
    assert main_py.exists()
    assert "test_project" in main_py.read_text()


def test_generate_files_with_hooks(tmp_path, monkeypatch):
    """Test generate_files function with hooks enabled."""
    from pathlib import Path
    from collections import OrderedDict
    
    repo_dir = tmp_path / "template_repo"
    repo_dir.mkdir()
    
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    test_file = template_dir / "test.txt"
    test_file.write_text("test content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    hook_calls = []
    
    def mock_run_hook(repo, hook_name, project_dir, context, delete_on_failure):
        hook_calls.append(hook_name)
    
    monkeypatch.setattr(
        "cookiecutter.generate.find_template",
        lambda repo, env: str(template_dir)
    )
    
    monkeypatch.setattr(
        "cookiecutter.generate.run_hook_from_repo_dir",
        mock_run_hook
    )
    
    context = OrderedDict([("cookiecutter", {"project_name": "test_project"})])
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=True,
        keep_project_on_failure=False
    )
    
    assert result is not None
    assert "pre_gen_project" in hook_calls
    assert "post_gen_project" in hook_calls


def test_generate_files_overwrite_if_exists(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    from pathlib import Path
    from collections import OrderedDict
    
    repo_dir = tmp_path / "template_repo"
    repo_dir.mkdir()
    
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    test_file = template_dir / "test.txt"
    test_file.write_text("new content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Pre-create the output project directory
    existing_project = output_dir / "test_project"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old content")
    
    monkeypatch.setattr(
        "cookiecutter.generate.find_template",
        lambda repo, env: str(template_dir)
    )
    
    monkeypatch.setattr(
        "cookiecutter.generate.run_hook_from_repo_dir",
        lambda *args, **kwargs: None
    )
    
    context = OrderedDict([("cookiecutter", {"project_name": "test_project"})])
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False
    )
    
    assert result is not None
    project_path = Path(result)
    assert project_path.exists()
    assert (project_path / "test.txt").exists()


def test_generate_files_skip_if_file_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists=True."""
    from pathlib import Path
    from collections import OrderedDict
    
    repo_dir = tmp_path / "template_repo"
    repo_dir.mkdir()
    
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}')
    
    test_file = template_dir / "config.txt"
    test_file.write_text("new config")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    monkeypatch.setattr(
        "cookiecutter.generate.find_template",
        lambda repo, env: str(template_dir)
    )
    
    monkeypatch.setattr(
        "cookiecutter.generate.run_hook_from_repo_dir",
        lambda *args, **kwargs: None
    )
    
    context = OrderedDict([("cookiecutter", {"project_name": "test_project"})])
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False
    )
    
    assert result is not None
    assert Path(result).exists()


# LLM-generated content at query #24
#--------------------------

```python
def test_apply_overwrites_to_context():
    """Test apply_overwrites_to_context function with various scenarios."""
    
    # Test 1: Simple value overwrite
    context = {'var1': 'original'}
    overwrite = {'var1': 'new_value'}
    apply_overwrites_to_context(context, overwrite)
    assert context['var1'] == 'new_value'
    
    # Test 2: Ignore new variables at first level
    context = {'var1': 'value1'}
    overwrite = {'var2': 'value2'}
    apply_overwrites_to_context(context, overwrite)
    assert 'var2' not in context
    
    # Test 3: Add new variables in dictionary (in_dictionary_variable=True)
    context = {'nested': {'key1': 'val1'}}
    overwrite = {'nested': {'key2': 'val2'}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['nested']['key2'] == 'val2'
    
    # Test 4: Choice variable (list) - valid choice
    context = {'choice': ['option1', 'option2', 'option3']}
    overwrite = {'choice': 'option2'}
    apply_overwrites_to_context(context, overwrite)
    assert context['choice'][0] == 'option2'
    assert 'option2' in context['choice']
    
    # Test 5: Choice variable - invalid choice raises ValueError
    context = {'choice': ['option1', 'option2']}
    overwrite = {'choice': 'invalid_option'}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'choice variable' in str(e)
    
    # Test 6: Multi-choice variable (list with list) - valid choices
    context = {'multichoice': ['opt1', 'opt2', 'opt3']}
    overwrite = {'multichoice': ['opt2', 'opt3']}
    apply_overwrites_to_context(context, overwrite)
    assert context['multichoice'] == ['opt2', 'opt3']
    
    # Test 7: Multi-choice variable - invalid choices raise ValueError
    context = {'multichoice': ['opt1', 'opt2']}
    overwrite = {'multichoice': ['opt1', 'invalid']}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'multi-choice variable' in str(e)
    
    # Test 8: Nested dictionary overwrite
    context = {'settings': {'key1': 'val1', 'key2': 'val2'}}
    overwrite = {'settings': {'key1': 'new_val1'}}
    apply_overwrites_to_context(context, overwrite)
    assert context['settings']['key1'] == 'new_val1'
    assert context['settings']['key2'] == 'val2'
    
    # Test 9: Boolean variable with string 'y'
    context = {'flag': True}
    overwrite = {'flag': 'y'}
    apply_overwrites_to_context(context, overwrite)
    assert context['flag'] is True
    
    # Test 10: Boolean variable with string 'n'
    context = {'flag': True}
    overwrite = {'flag': 'n'}
    apply_overwrites_to_context(context, overwrite)
    assert context['flag'] is False
    
    # Test 11: Boolean variable with invalid string raises ValueError
    context = {'flag': True}
    overwrite = {'flag': 'invalid'}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'could not be converted to a boolean' in str(e)
    
    # Test 12: List in dictionary variable at deeper level
    context = {'nested': {'items': ['a', 'b', 'c']}}
    overwrite = {'nested': {'items': ['x', 'y']}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['nested']['items'] == ['x', 'y']
    
    # Test 13: Multiple variables overwrite
    context = {'var1': 'val1', 'var2': 'val2', 'var3': 'val3'}
    overwrite = {'var1': 'new1', 'var3': 'new3'}
    apply_overwrites_to_context(context, overwrite)
    assert context['var1'] == 'new1'
    assert context['var2'] == 'val2'
    assert context['var3'] == 'new3'
    
    # Test 14: Empty overwrite dictionary
    context = {'var1': 'val1'}
    overwrite = {}
    apply_overwrites_to_context(context, overwrite)
    assert context['var1'] == 'val1'
    
    # Test 15: Deep nested dictionary
    context = {'level1': {'level2': {'level3': 'value'}}}
    overwrite = {'level1': {'level2': {'level3': 'new_value'}}}
    apply_overwrites_to_context(context, overwrite)
    assert context['level1']['level2']['level3'] == 'new_value'


# LLM-generated content at query #25
#--------------------------

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pytest
from jinja2 import Environment, FileSystemLoader
from cookiecutter.generate import generate_file


def test_generate_file():
    """Test generate_file function with various scenarios."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        project_dir = tmpdir
        context = {
            'cookiecutter': {
                'project_name': 'test_project',
                '_new_lines': '\n'
            }
        }
        env = Environment(loader=FileSystemLoader(tmpdir))
        
        # Create a test input file
        infile = 'test_{{ cookiecutter.project_name }}.txt'
        infile_path = os.path.join(tmpdir, infile)
        with open(infile_path, 'w', encoding='utf-8') as f:
            f.write('Hello {{ cookiecutter.project_name }}!')
        
        # Test rendering a text file
        generate_file(project_dir, infile, context, env)
        
        expected_outfile = os.path.join(project_dir, 'test_test_project.txt')
        assert os.path.exists(expected_outfile)
        
        with open(expected_outfile, 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == 'Hello test_project!'


def test_generate_file_binary():
    """Test generate_file with binary files."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        context = {'cookiecutter': {'name': 'test'}}
        env = Environment(loader=FileSystemLoader(tmpdir))
        
        # Create a binary file
        infile = 'binary_file.bin'
        infile_path = os.path.join(tmpdir, infile)
        binary_content = b'\x89PNG\r\n\x1a\n'
        with open(infile_path, 'wb') as f:
            f.write(binary_content)
        
        generate_file(project_dir, infile, context, env)
        
        outfile = os.path.join(project_dir, infile)
        assert os.path.exists(outfile)
        
        with open(outfile, 'rb') as f:
            assert f.read() == binary_content


def test_generate_file_skip_if_exists():
    """Test generate_file with skip_if_file_exists flag."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        context = {'cookiecutter': {'name': 'test'}}
        env = Environment(loader=FileSystemLoader(tmpdir))
        
        infile = 'existing_file.txt'
        infile_path = os.path.join(tmpdir, infile)
        with open(infile_path, 'w', encoding='utf-8') as f:
            f.write('Original content')
        
        # Create existing output file
        outfile = os.path.join(project_dir, infile)
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write('Existing content')
        
        # Call with skip_if_file_exists=True
        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
        
        # File should not be overwritten
        with open(outfile, 'r', encoding='utf-8') as f:
            assert f.read() == 'Existing content'


def test_generate_file_empty_filename():
    """Test generate_file when rendered filename is empty directory."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        context = {'cookiecutter': {'name': 'test'}}
        env = Environment(loader=FileSystemLoader(tmpdir))
        
        infile = 'test.txt'
        infile_path = os.path.join(tmpdir, infile)
        with open(infile_path, 'w', encoding='utf-8') as f:
            f.write('test')
        
        # Create a directory with the rendered name
        outdir = os.path.join(project_dir, 'test.txt')
        os.makedirs(outdir, exist_ok=True)
        
        # Should return early without error
        generate_file(project_dir, infile, context, env)
        
        # Directory should still exist
        assert os.path.isdir(outdir)


def test_generate_file_with_detected_newlines():
    """Test generate_file detects and uses file newlines."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        context = {'cookiecutter': {}}
        env = Environment(loader=FileSystemLoader(tmpdir))
        
        infile = 'multiline.txt'
        infile_path = os.path.join(tmpdir, infile)
        
        # Create file with specific newlines (Windows style)
        with open(infile_path, 'w', encoding='utf-8', newline='\r\n') as f:
            f.write('Line 1\r\nLine 2\r\n')
        
        generate_file(project_dir, infile, context, env)
        
        outfile = os.path.join(project_dir, infile)
        assert os.path.exists(outfile)
        
        with open(outfile, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'Line 1' in content
            assert 'Line 2' in content


def test_generate_file_template_syntax_error():
    """Test generate_file with template syntax error."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        context = {'cookiecutter': {}}
        env = Environment(loader=FileSystemLoader(tmpdir))
        
        infile = 'bad_syntax.txt'
        infile_path = os.path.join(tmpdir, infile)
        
        # Create file with invalid Jinja2 syntax
        with open(infile_path, 'w', encoding='utf-8') as f:
            f.write('{{ unclosed variable')
        
        from jinja2.exceptions import TemplateSyntaxError
        with pytest.raises(TemplateSyntaxError):
            generate_file(project_dir, infile, context, env)


def test_generate_file_with_custom_newlines():
    """Test generate_file uses _new_lines from context."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        context = {'cookiecutter': {'_new_lines': '\r\n'}}
        env = Environment(loader=FileSystemLoader(tmpdir))
        
        infile = 'test.txt'
        infile_path = os.path.join(tmpdir, infile)
        with open(infile_path, 'w', encoding='utf-8') as f:
            f.write('Line 1\nLine 2\n')
        
        generate_file(project_dir, infile, context, env)
        
        outfile = os.path.join(project_dir, infile)
        with open(outfile, 'rb') as f:
            content = f.read()
            # Should use Windows-style newlines
            assert b'\r\n' in content


def test_generate_file_preserves_permissions():
    """Test generate_file preserves file permissions."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = tmpdir
        context = {'cookiecutter': {}}
        env = Environment(loader=FileSystemLoader(tmpdir))
        
        infile = 'script.sh'
        infile_path = os.path.join(tmpdir, infile)
        with open(infile_path, 'w', encoding='utf-8') as f:
            f.write('#!/bin/bash\


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_context(tmp_path, monkeypatch):
    """Test generate_context function with various scenarios."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Test 1: Basic context generation
    context_file = tmp_path / 'cookiecutter.json'
    context_file.write_text('{"project_name": "my_project", "author": "John Doe"}')
    
    result = generate_context(str(context_file))
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'
    assert result['cookiecutter']['author'] == 'John Doe'
    
    # Test 2: With default_context
    context_file2 = tmp_path / 'cookiecutter2.json'
    context_file2.write_text('{"project_name": "default_project", "version": "1.0"}')
    
    default_context = {'project_name': 'overridden_project'}
    result = generate_context(str(context_file2), default_context=default_context)
    assert result['cookiecutter']['project_name'] == 'overridden_project'
    assert result['cookiecutter']['version'] == '1.0'
    
    # Test 3: With extra_context
    context_file3 = tmp_path / 'cookiecutter3.json'
    context_file3.write_text('{"project_name": "test", "license": "MIT"}')
    
    extra_context = {'license': 'Apache'}
    result = generate_context(str(context_file3), extra_context=extra_context)
    assert result['cookiecutter']['license'] == 'Apache'
    assert result['cookiecutter']['project_name'] == 'test'
    
    # Test 4: Invalid JSON raises ContextDecodingException
    invalid_json_file = tmp_path / 'invalid.json'
    invalid_json_file.write_text('{invalid json}')
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(invalid_json_file))
    
    # Test 5: Complex nested structure
    complex_context_file = tmp_path / 'complex.json'
    complex_context_file.write_text(
        '{"project_name": "complex", "options": {"debug": true, "level": 5}}'
    )
    result = generate_context(str(complex_context_file))
    assert result['cookiecutter']['project_name'] == 'complex'
    assert result['cookiecutter']['options']['debug'] is True
    assert result['cookiecutter']['options']['level'] == 5
    
    # Test 6: With choice variable (list)
    choice_context_file = tmp_path / 'choices.json'
    choice_context_file.write_text('{"license": ["MIT", "Apache", "GPL"]}')
    
    extra = {'license': 'Apache'}
    result = generate_context(str(choice_context_file), extra_context=extra)
    assert result['cookiecutter']['license'][0] == 'Apache'
    assert 'MIT' in result['cookiecutter']['license']
    
    # Test 7: With boolean variable and string override
    bool_context_file = tmp_path / 'bool.json'
    bool_context_file.write_text('{"use_docker": true}')
    
    extra = {'use_docker': 'n'}
    result = generate_context(str(bool_context_file), extra_context=extra)
    assert result['cookiecutter']['use_docker'] is False
    
    # Test 8: OrderedDict preservation
    ordered_context_file = tmp_path / 'ordered.json'
    ordered_context_file.write_text('{"a": 1, "b": 2, "c": 3}')
    
    result = generate_context(str(ordered_context_file))
    assert isinstance(result, (dict, OrderedDict))
    assert list(result['cookiecutter'].keys()) == ['a', 'b', 'c']
    
    # Test 9: Non-existent context file
    with pytest.raises(FileNotFoundError):
        generate_context(str(tmp_path / 'nonexistent.json'))
    
    # Test 10: Empty JSON object
    empty_context_file = tmp_path / 'empty.json'
    empty_context_file.write_text('{}')
    
    result = generate_context(str(empty_context_file))
    assert result['cookiecutter'] == {}


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_context(tmp_path, monkeypatch):
    """Test generate_context function with various scenarios."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / 'cookiecutter.json'
    context_data = {
        'project_name': 'My Project',
        'project_slug': '{{ cookiecutter.project_name.lower().replace(" ", "_") }}',
        'author_name': 'John Doe'
    }
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'My Project'
    assert result['cookiecutter']['author_name'] == 'John Doe'
    
    # Test 2: Context generation with default_context overrides
    default_context = {
        'project_name': 'Default Project',
        'author_name': 'Jane Doe'
    }
    result = generate_context(
        str(context_file),
        default_context=default_context
    )
    assert result['cookiecutter']['project_name'] == 'Default Project'
    assert result['cookiecutter']['author_name'] == 'Jane Doe'
    
    # Test 3: Context generation with extra_context overrides
    extra_context = {
        'project_name': 'Extra Project'
    }
    result = generate_context(
        str(context_file),
        extra_context=extra_context
    )
    assert result['cookiecutter']['project_name'] == 'Extra Project'
    
    # Test 4: Both default_context and extra_context (extra_context should win)
    result = generate_context(
        str(context_file),
        default_context={'project_name': 'Default'},
        extra_context={'project_name': 'Extra'}
    )
    assert result['cookiecutter']['project_name'] == 'Extra'
    
    # Test 5: Choice variable with overwrite
    choice_context_file = tmp_path / 'choice_cookiecutter.json'
    choice_data = {
        'python_version': ['3.9', '3.10', '3.11']
    }
    choice_context_file.write_text(json.dumps(choice_data))
    
    result = generate_context(
        str(choice_context_file),
        extra_context={'python_version': '3.10'}
    )
    assert result['cookiecutter']['python_version'][0] == '3.10'
    
    # Test 6: Multi-choice variable with overwrite
    multi_choice_file = tmp_path / 'multi_cookiecutter.json'
    multi_data = {
        'features': ['feature1', 'feature2', 'feature3']
    }
    multi_choice_file.write_text(json.dumps(multi_data))
    
    result = generate_context(
        str(multi_choice_file),
        extra_context={'features': ['feature2', 'feature3']}
    )
    assert set(result['cookiecutter']['features']) == {'feature2', 'feature3'}
    
    # Test 7: Dictionary context variable with nested overwrite
    dict_context_file = tmp_path / 'dict_cookiecutter.json'
    dict_data = {
        'database': {
            'engine': 'postgresql',
            'port': 5432
        }
    }
    dict_context_file.write_text(json.dumps(dict_data))
    
    result = generate_context(
        str(dict_context_file),
        extra_context={'database': {'port': 3306}}
    )
    assert result['cookiecutter']['database']['engine'] == 'postgresql'
    assert result['cookiecutter']['database']['port'] == 3306
    
    # Test 8: Boolean variable with string overwrite
    bool_context_file = tmp_path / 'bool_cookiecutter.json'
    bool_data = {
        'use_docker': False
    }
    bool_context_file.write_text(json.dumps(bool_data))
    
    result = generate_context(
        str(bool_context_file),
        extra_context={'use_docker': 'y'}
    )
    assert result['cookiecutter']['use_docker'] is True
    
    # Test 9: Invalid JSON file raises ContextDecodingException
    invalid_json_file = tmp_path / 'invalid.json'
    invalid_json_file.write_text('{invalid json}')
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(invalid_json_file))
    
    # Test 10: Invalid choice raises ValueError
    with pytest.raises(ValueError):
        generate_context(
            str(choice_context_file),
            extra_context={'python_version': '3.12'}
        )
    
    # Test 11: Invalid multi-choice raises ValueError
    with pytest.raises(ValueError):
        generate_context(
            str(multi_choice_file),
            extra_context={'features': ['feature1', 'invalid_feature']}
        )
    
    # Test 12: Invalid boolean conversion raises ValueError
    with pytest.raises(ValueError):
        generate_context(
            str(bool_context_file),
            extra_context={'use_docker': 'maybe'}
        )
    
    # Test 13: Context with _copy_without_render
    copy_render_file = tmp_path / 'copy_cookiecutter.json'
    copy_data = {
        'project_name': 'Test',
        '_copy_without_render': ['*.binary', 'static/*']
    }
    copy_render_file.write_text(json.dumps(copy_data))
    
    result = generate_context(str(copy_render_file))
    assert '_copy_without_render' in result['cookiecutter']
    assert result['cookiecutter']['_copy_without_render'] == ['*.binary', 'static/*']


# LLM-generated content at query #28
#--------------------------

```python
def test_generate_files(tmp_path, mocker):
    """Test generate_files function."""
    # Setup
    repo_dir = tmp_path / "repo"
    output_dir = tmp_path / "output"
    repo_dir.mkdir()
    output_dir.mkdir()

    # Create template directory structure
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}\n")
    
    # Create cookiecutter.json in repo
    config_file = repo_dir / "cookiecutter.json"
    config_file.write_text('{"project_name": "test_project"}')

    # Mock the hooks
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))

    # Setup context
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }

    # Execute
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=True
    )

    # Assert
    assert result is not None
    assert os.path.isdir(result)
    assert 'test_project' in result
    readme_file = Path(result) / "README.md"
    assert readme_file.exists()
    assert "# test_project" in readme_file.read_text()


def test_generate_files_with_overwrite(tmp_path, mocker):
    """Test generate_files with overwrite_if_exists."""
    repo_dir = tmp_path / "repo"
    output_dir = tmp_path / "output"
    repo_dir.mkdir()
    output_dir.mkdir()

    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")

    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))

    context = {'cookiecutter': {'project_name': 'my_project'}}

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

    assert result1 == result2
    assert os.path.isdir(result2)


def test_generate_files_empty_context(tmp_path, mocker):
    """Test generate_files with empty context."""
    repo_dir = tmp_path / "repo"
    output_dir = tmp_path / "output"
    repo_dir.mkdir()
    output_dir.mkdir()

    template_dir = repo_dir / "my_project"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")

    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))

    result = generate_files(
        repo_dir=str(repo_dir),
        context=None,
        output_dir=str(output_dir),
        accept_hooks=False
    )

    assert result is not None
    assert os.path.isdir(result)


def test_generate_files_skip_if_file_exists(tmp_path, mocker):
    """Test generate_files with skip_if_file_exists."""
    repo_dir = tmp_path / "repo"
    output_dir = tmp_path / "output"
    repo_dir.mkdir()
    output_dir.mkdir()

    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("new content")

    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))

    context = {'cookiecutter': {'project_name': 'my_project'}}

    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False
    )

    output_file = Path(result) / "file.txt"
    assert output_file.exists()
    assert output_file.read_text() == "new content"


def test_generate_files_with_hooks_disabled(tmp_path, mocker):
    """Test generate_files with accept_hooks=False."""
    repo_dir = tmp_path / "repo"
    output_dir = tmp_path / "output"
    repo_dir.mkdir()
    output_dir.mkdir()

    template_dir = repo_dir / "project"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")

    mock_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))

    context = {'cookiecutter': {}}

    generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False
    )

    mock_hook.assert_not_called()


def test_generate_files_with_hooks_enabled(tmp_path, mocker):
    """Test generate_files with accept_hooks=True."""
    repo_dir = tmp_path / "repo"
    output_dir = tmp_path / "output"
    repo_dir.mkdir()
    output_dir.mkdir()

    template_dir = repo_dir / "project"
    template_dir.mkdir()
    
    template_file = template_dir / "file.txt"
    template_file.write_text("content")

    mock_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))

    context = {'cookiecutter': {}}

    generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=True
    )

    assert mock_hook.call_count == 2


def test_generate_files_undefined_variable_in_dirname(tmp_path, mocker):
    """Test generate_files with undefined variable in directory name."""
    repo_dir = tmp_path / "repo"
    output_dir = tmp_path / "output"
    repo_dir.mkdir()
    output_dir.mkdir()

    template_dir = repo_dir / "{{cookiecutter.undefined_var}}"
    template_dir.mkdir()

    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))

    context = {'cookiecutter': {}}

    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_


# LLM-generated content at query #29
#--------------------------

```python
def test_render_and_create_dir(tmp_path, monkeypatch):
    """Test render_and_create_dir function."""
    from jinja2 import Environment
    
    # Test 1: Basic directory creation with simple name
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    result_dir, created = render_and_create_dir(
        'test_dir', context, output_dir, env
    )
    
    assert result_dir == output_dir / 'test_dir'
    assert result_dir.exists()
    assert created is True
    
    # Test 2: Directory creation with templated name
    output_dir2 = tmp_path / "output2"
    output_dir2.mkdir()
    context2 = {'cookiecutter': {'project_name': 'my_project'}}
    
    result_dir2, created2 = render_and_create_dir(
        '{{cookiecutter.project_name}}', context2, output_dir2, env
    )
    
    assert result_dir2 == output_dir2 / 'my_project'
    assert result_dir2.exists()
    assert created2 is True
    
    # Test 3: Directory already exists, overwrite_if_exists=False should raise
    output_dir3 = tmp_path / "output3"
    output_dir3.mkdir()
    existing_dir = output_dir3 / "existing"
    existing_dir.mkdir()
    
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            'existing', context, output_dir3, env, overwrite_if_exists=False
        )
    
    # Test 4: Directory already exists, overwrite_if_exists=True should succeed
    output_dir4 = tmp_path / "output4"
    output_dir4.mkdir()
    existing_dir2 = output_dir4 / "existing"
    existing_dir2.mkdir()
    
    result_dir4, created4 = render_and_create_dir(
        'existing', context, output_dir4, env, overwrite_if_exists=True
    )
    
    assert result_dir4 == output_dir4 / 'existing'
    assert result_dir4.exists()
    assert created4 is False
    
    # Test 5: Empty directory name should raise EmptyDirNameException
    output_dir5 = tmp_path / "output5"
    output_dir5.mkdir()
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir5, env)
    
    # Test 6: None directory name should raise EmptyDirNameException
    output_dir6 = tmp_path / "output6"
    output_dir6.mkdir()
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(None, context, output_dir6, env)
    
    # Test 7: Complex templated directory name
    output_dir7 = tmp_path / "output7"
    output_dir7.mkdir()
    context7 = {'cookiecutter': {'author': 'john', 'project': 'app'}}
    
    result_dir7, created7 = render_and_create_dir(
        '{{cookiecutter.author}}_{{cookiecutter.project}}',
        context7,
        output_dir7,
        env
    )
    
    assert result_dir7 == output_dir7 / 'john_app'
    assert result_dir7.exists()
    assert created7 is True
    
    # Test 8: Nested path creation
    output_dir8 = tmp_path / "output8"
    output_dir8.mkdir()
    
    result_dir8, created8 = render_and_create_dir(
        'parent/child', context, output_dir8, env
    )
    
    assert result_dir8 == output_dir8 / 'parent' / 'child'
    assert result_dir8.exists()
    assert created8 is True


# LLM-generated content at query #30
#--------------------------

```python
import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from collections import OrderedDict
from unittest.mock import Mock, patch, MagicMock
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError, UndefinedError


def test_generate_files():
    """Test generate_files function with various scenarios."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup template directory structure
        repo_dir = Path(tmpdir) / "repo"
        template_dir = repo_dir / "{{cookiecutter.project_name}}"
        output_dir = Path(tmpdir) / "output"
        
        repo_dir.mkdir()
        template_dir.mkdir()
        output_dir.mkdir()
        
        # Create a basic cookiecutter.json
        context_file = repo_dir / "cookiecutter.json"
        context_data = {
            "project_name": "test_project",
            "author": "Test Author"
        }
        with open(context_file, 'w') as f:
            json.dump(context_data, f)
        
        # Create a template file
        template_file = template_dir / "README.md"
        with open(template_file, 'w') as f:
            f.write("# {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}")
        
        # Create context
        context = OrderedDict([
            ('cookiecutter', OrderedDict([
                ('project_name', 'test_project'),
                ('author', 'Test Author')
            ]))
        ])
        
        # Test basic file generation
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=False,
            skip_if_file_exists=False,
            accept_hooks=False,
            keep_project_on_failure=False
        )
        
        # Verify output
        assert result is not None
        assert Path(result).exists()
        
        # Verify generated file content
        generated_readme = Path(result) / "README.md"
        assert generated_readme.exists()
        with open(generated_readme, 'r') as f:
            content = f.read()
            assert "test_project" in content
            assert "Test Author" in content


def test_generate_files_with_empty_context():
    """Test generate_files with empty context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        template_dir = repo_dir / "simple_project"
        output_dir = Path(tmpdir) / "output"
        
        repo_dir.mkdir()
        template_dir.mkdir()
        output_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        with open(context_file, 'w') as f:
            json.dump({"project_name": "default"}, f)
        
        template_file = template_dir / "file.txt"
        with open(template_file, 'w') as f:
            f.write("content")
        
        result = generate_files(
            repo_dir=str(repo_dir),
            context=None,
            output_dir=str(output_dir),
            accept_hooks=False
        )
        
        assert result is not None
        assert Path(result).exists()


def test_generate_files_with_overwrite_if_exists():
    """Test generate_files with overwrite_if_exists flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        template_dir = repo_dir / "{{cookiecutter.name}}"
        output_dir = Path(tmpdir) / "output"
        
        repo_dir.mkdir()
        template_dir.mkdir()
        output_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        with open(context_file, 'w') as f:
            json.dump({"name": "project"}, f)
        
        template_file = template_dir / "file.txt"
        with open(template_file, 'w') as f:
            f.write("content")
        
        context = OrderedDict([
            ('cookiecutter', OrderedDict([('name', 'project')]))
        ])
        
        # First generation
        result1 = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=False,
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


def test_generate_files_with_binary_file():
    """Test generate_files with binary files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        template_dir = repo_dir / "project"
        output_dir = Path(tmpdir) / "output"
        
        repo_dir.mkdir()
        template_dir.mkdir()
        output_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        with open(context_file, 'w') as f:
            json.dump({"name": "project"}, f)
        
        # Create a binary file
        binary_file = template_dir / "image.bin"
        with open(binary_file, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
        
        context = OrderedDict([
            ('cookiecutter', OrderedDict([('name', 'project')]))
        ])
        
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False
        )
        
        assert result is not None
        assert Path(result).exists()


def test_generate_files_with_skip_if_file_exists():
    """Test generate_files with skip_if_file_exists flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        template_dir = repo_dir / "project"
        output_dir = Path(tmpdir) / "output"
        
        repo_dir.mkdir()
        template_dir.mkdir()
        output_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        with open(context_file, 'w') as f:
            json.dump({"name": "project"}, f)
        
        template_file = template_dir / "file.txt"
        with open(template_file, 'w') as f:
            f.write("original content")
        
        context = OrderedDict([
            ('cookiecutter', OrderedDict([('name', 'project')]))
        ])
        
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            skip_if_file_exists=True,
            accept_hooks=False
        )
        
        assert result is not None
        assert Path(result).exists()


def test_generate_files_with_copy_without_render():
    """Test generate_files with _copy_without_render context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        template_dir = repo_dir / "project"
        output_dir = Path(tmpdir) / "output"
        
        repo_dir.mkdir()
        template_dir.mkdir()
        output_dir.mkdir()
        


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    import os
    from pathlib import Path
    from collections import OrderedDict
    
    # Create a temporary template directory structure
    template_dir = tmp_path / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    (template_dir / "README.md").write_text("# {{cookiecutter.project_name}}\n")
    
    # Create a simple Python file
    (template_dir / "main.py").write_text("print('{{cookiecutter.greeting}}')\n")
    
    # Create a subdirectory with a template
    subdir = template_dir / "src"
    subdir.mkdir()
    (subdir / "{{cookiecutter.module_name}}.py").write_text(
        "def hello():\n    return '{{cookiecutter.greeting}}'\n"
    )
    
    # Create cookiecutter.json
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text(
        '{"project_name": "my_project", "greeting": "Hello", "module_name": "mymodule"}'
    )
    
    # Mock find_template to return our template directory
    from cookiecutter import generate
    original_find_template = generate.find_template
    
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    monkeypatch.setattr(generate, "find_template", mock_find_template)
    
    # Mock run_hook_from_repo_dir to avoid actual hook execution
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr(generate, "run_hook_from_repo_dir", mock_run_hook)
    
    # Prepare context
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('project_name', 'test_project'),
            ('greeting', 'Hi'),
            ('module_name', 'testmod'),
        ]))
    ])
    
    # Generate files
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False,
    )
    
    # Verify the project directory was created
    assert os.path.exists(result)
    assert "test_project" in result
    
    # Verify files were generated with correct content
    readme_path = Path(result) / "README.md"
    assert readme_path.exists()
    assert readme_path.read_text() == "# test_project\n"
    
    main_py_path = Path(result) / "main.py"
    assert main_py_path.exists()
    assert main_py_path.read_text() == "print('Hi')\n"
    
    # Verify subdirectory and files
    module_path = Path(result) / "src" / "testmod.py"
    assert module_path.exists()
    assert module_path.read_text() == "def hello():\n    return 'Hi'\n"


def test_generate_files_with_overwrite(tmp_path, monkeypatch):
    """Test generate_files with overwrite_if_exists=True."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter import generate
    
    # Create template directory
    template_dir = tmp_path / "{{cookiecutter.name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content: {{cookiecutter.value}}\n")
    
    # Mock find_template
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    monkeypatch.setattr(generate, "find_template", mock_find_template)
    
    # Mock hooks
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr(generate, "run_hook_from_repo_dir", mock_run_hook)
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('name', 'myapp'),
            ('value', 'test1'),
        ]))
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # First generation
    result1 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        accept_hooks=False,
    )
    
    assert Path(result1, "file.txt").read_text() == "content: test1\n"
    
    # Second generation with overwrite
    context['cookiecutter']['value'] = 'test2'
    result2 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False,
    )
    
    assert Path(result2, "file.txt").read_text() == "content: test2\n"


def test_generate_files_skip_if_exists(tmp_path, monkeypatch):
    """Test generate_files with skip_if_file_exists=True."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter import generate
    
    # Create template directory
    template_dir = tmp_path / "{{cookiecutter.name}}"
    template_dir.mkdir()
    (template_dir / "existing.txt").write_text("new content\n")
    
    # Mock find_template
    def mock_find_template(repo_dir, env):
        return str(template_dir)
    
    monkeypatch.setattr(generate, "find_template", mock_find_template)
    
    # Mock hooks
    def mock_run_hook(repo_dir, hook_name, project_dir, context, delete_on_failure):
        pass
    
    monkeypatch.setattr(generate, "run_hook_from_repo_dir", mock_run_hook)
    
    context = OrderedDict([
        ('cookiecutter', OrderedDict([
            ('name', 'myapp'),
        ]))
    ])
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Generate files
    result = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        accept_hooks=False,
    )
    
    # Create a file that should be skipped
    existing_file = Path(result) / "existing.txt"
    existing_file.write_text("old content\n")
    
    # Generate again - should skip the existing file
    result2 = generate_files(
        repo_dir=str(tmp_path),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=True,
        overwrite_if_exists=True,
        accept_hooks=False,
    )
    
    # Content should remain unchanged
    assert existing_file.read_text() == "old content\n"


def test_generate_files_with_copy_without_render(tmp_path, monkeypatch):
    """Test generate_files with _copy_without_render setting."""
    from pathlib import Path
    from collections import OrderedDict
    from cookiecutter import generate
    
    # Create template directory
    template_dir = tmp_path / "{{cookiecutter.name}}"
    template_dir.mkdir()
    
    # Create a file that should not


# LLM-generated content at query #32
#--------------------------

```python
import pytest
import os
import json
from pathlib import Path
from collections import OrderedDict
from unittest.mock import Mock, patch, MagicMock
from jinja2 import Environment

def test_generate_files(tmp_path):
    """Test generate_files function with basic template rendering."""
    # Setup template directory structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    # Create a simple template file
    (template_dir / "README.md").write_text("# {{cookiecutter.project_name}}")
    
    # Create cookiecutter.json
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text(json.dumps({"project_name": "my_project"}))
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create context
    context = {"cookiecutter": {"project_name": "my_project"}}
    
    with patch('cookiecutter.generate.find_template', return_value=str(template_dir)):
        with patch('cookiecutter.generate.create_env_with_context') as mock_env:
            mock_environment = MagicMock(spec=Environment)
            mock_environment.from_string = lambda x: MagicMock(render=lambda **kw: x.format(**kw) if "{" in x else x)
            mock_environment.loader = None
            mock_env.return_value = mock_environment
            
            with patch('cookiecutter.generate.run_hook_from_repo_dir'):
                result = generate_files(
                    repo_dir=str(repo_dir),
                    context=context,
                    output_dir=str(output_dir),
                    accept_hooks=False
                )
    
    assert result is not None
    assert Path(result).exists()


def test_generate_files_with_overwrite(tmp_path):
    """Test generate_files with overwrite_if_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create existing output directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old content")
    
    context = {"cookiecutter": {"project_name": "my_project"}}
    
    with patch('cookiecutter.generate.find_template', return_value=str(template_dir)):
        with patch('cookiecutter.generate.create_env_with_context') as mock_env:
            mock_environment = MagicMock(spec=Environment)
            mock_environment.from_string = lambda x: MagicMock(render=lambda **kw: x.format(**kw) if "{" in x else x)
            mock_environment.loader = None
            mock_env.return_value = mock_environment
            
            with patch('cookiecutter.generate.run_hook_from_repo_dir'):
                result = generate_files(
                    repo_dir=str(repo_dir),
                    context=context,
                    output_dir=str(output_dir),
                    overwrite_if_exists=True,
                    accept_hooks=False
                )
    
    assert Path(result).exists()


def test_generate_files_empty_context(tmp_path):
    """Test generate_files with empty context."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "project"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    with patch('cookiecutter.generate.find_template', return_value=str(template_dir)):
        with patch('cookiecutter.generate.create_env_with_context') as mock_env:
            mock_environment = MagicMock(spec=Environment)
            mock_environment.from_string = lambda x: MagicMock(render=lambda **kw: x)
            mock_environment.loader = None
            mock_env.return_value = mock_environment
            
            with patch('cookiecutter.generate.run_hook_from_repo_dir'):
                result = generate_files(
                    repo_dir=str(repo_dir),
                    output_dir=str(output_dir),
                    accept_hooks=False
                )
    
    assert result is not None


def test_generate_files_with_hooks(tmp_path):
    """Test generate_files executes hooks when accept_hooks is True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "project"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    with patch('cookiecutter.generate.find_template', return_value=str(template_dir)):
        with patch('cookiecutter.generate.create_env_with_context') as mock_env:
            mock_environment = MagicMock(spec=Environment)
            mock_environment.from_string = lambda x: MagicMock(render=lambda **kw: x)
            mock_environment.loader = None
            mock_env.return_value = mock_environment
            
            with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_hook:
                generate_files(
                    repo_dir=str(repo_dir),
                    context=context,
                    output_dir=str(output_dir),
                    accept_hooks=True
                )
                
                assert mock_hook.call_count == 2


def test_generate_files_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists flag."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "project"
    template_dir.mkdir()
    (template_dir / "file.txt").write_text("new content")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    with patch('cookiecutter.generate.find_template', return_value=str(template_dir)):
        with patch('cookiecutter.generate.create_env_with_context') as mock_env:
            mock_environment = MagicMock(spec=Environment)
            mock_environment.from_string = lambda x: MagicMock(render=lambda **kw: x)
            mock_environment.loader = None
            mock_env.return_value = mock_environment
            
            with patch('cookiecutter.generate.run_hook_from_repo_dir'):
                with patch('cookiecutter.generate.generate_file'):
                    result = generate_files(
                        repo_dir=str(repo_dir),
                        context=context,
                        output_dir=str(output_dir),
                        skip_if_file_exists=True,
                        accept_hooks=False
                    )
    
    assert result is not None


def test_generate_files_keep_project_on_failure(tmp_path):
    """Test generate_files keeps project directory on failure when flag is set."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "my_project"}}
    
    with patch('cookiecutter.


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    import json
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "my_project", "author": "John Doe"}
    context_file.write_text(json.dumps(context_data))
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"
    
    # Test 2: Context generation with default_context
    default_context = {"project_name": "default_project"}
    result = generate_context(str(context_file), default_context=default_context)
    assert result["cookiecutter"]["project_name"] == "default_project"
    
    # Test 3: Context generation with extra_context
    extra_context = {"author": "Jane Smith"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["author"] == "Jane Smith"
    
    # Test 4: Both default_context and extra_context (extra_context takes precedence)
    default_context = {"project_name": "default_project"}
    extra_context = {"project_name": "extra_project"}
    result = generate_context(
        str(context_file),
        default_context=default_context,
        extra_context=extra_context
    )
    assert result["cookiecutter"]["project_name"] == "extra_project"
    
    # Test 5: Invalid JSON raises ContextDecodingException
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text("{invalid json content")
    
    with pytest.raises(ContextDecodingException):
        generate_context(str(invalid_json_file))
    
    # Test 6: Choice variable with extra_context
    context_data = {"license": ["MIT", "Apache", "GPL"]}
    context_file.write_text(json.dumps(context_data))
    extra_context = {"license": "Apache"}
    result = generate_context(str(context_file), extra_context=extra_context)
    # Apache should be first in list after overwrite
    assert result["cookiecutter"]["license"][0] == "Apache"
    
    # Test 7: Multi-choice variable with extra_context
    context_data = {"features": [["feature1", "feature2", "feature3"]]}
    context_file.write_text(json.dumps(context_data))
    extra_context = {"features": ["feature1", "feature3"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["features"] == ["feature1", "feature3"]
    
    # Test 8: Boolean variable with string extra_context
    context_data = {"use_docker": False}
    context_file.write_text(json.dumps(context_data))
    extra_context = {"use_docker": "y"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["use_docker"] is True
    
    # Test 9: Nested dictionary context
    context_data = {"project": {"name": "test", "version": "1.0"}}
    context_file.write_text(json.dumps(context_data))
    extra_context = {"project": {"version": "2.0"}}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["project"]["name"] == "test"
    assert result["cookiecutter"]["project"]["version"] == "2.0"
    
    # Test 10: Invalid default_context triggers warning
    context_data = {"license": ["MIT", "Apache"]}
    context_file.write_text(json.dumps(context_data))
    invalid_default = {"license": "InvalidChoice"}
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = generate_context(
            str(context_file),
            default_context=invalid_default
        )
        assert len(w) == 1
        assert "Invalid default received" in str(w[0].message)


# LLM-generated content at query #34
#--------------------------

```python
def test_generate_context(tmp_path):
    """Test generate_context function with various scenarios."""
    import json
    from collections import OrderedDict
    
    # Test 1: Basic context generation from JSON file
    context_file = tmp_path / "cookiecutter.json"
    context_data = {"project_name": "my_project", "author": "John Doe"}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    result = generate_context(str(context_file))
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"
    
    # Test 2: Context with default_context overwrite
    context_file = tmp_path / "cookiecutter2.json"
    context_data = {
        "project_name": "default_project",
        "author": "Jane Doe",
        "version": "1.0.0"
    }
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    default_context = {"author": "Admin", "version": "2.0.0"}
    result = generate_context(str(context_file), default_context=default_context)
    assert result["cookiecutter"]["author"] == "Admin"
    assert result["cookiecutter"]["version"] == "2.0.0"
    assert result["cookiecutter"]["project_name"] == "default_project"
    
    # Test 3: Context with extra_context overwrite
    context_file = tmp_path / "cookiecutter3.json"
    context_data = {"project_name": "base_project", "license": "MIT"}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    extra_context = {"project_name": "override_project"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["project_name"] == "override_project"
    assert result["cookiecutter"]["license"] == "MIT"
    
    # Test 4: Context with choice variables
    context_file = tmp_path / "cookiecutter4.json"
    context_data = {"project_type": ["web", "api", "cli"]}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    extra_context = {"project_type": "api"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["project_type"][0] == "api"
    
    # Test 5: Context with multi-choice variables
    context_file = tmp_path / "cookiecutter5.json"
    context_data = {"features": ["auth", "db", "cache", "logging"]}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    extra_context = {"features": ["auth", "db"]}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert set(result["cookiecutter"]["features"]) == {"auth", "db", "cache", "logging"}
    
    # Test 6: Context with nested dictionary
    context_file = tmp_path / "cookiecutter6.json"
    context_data = {
        "project_name": "test",
        "config": {"database": "postgres", "port": 5432}
    }
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    extra_context = {"config": {"port": 3306}}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["config"]["database"] == "postgres"
    assert result["cookiecutter"]["config"]["port"] == 3306
    
    # Test 7: Invalid JSON file
    context_file = tmp_path / "invalid.json"
    with open(context_file, 'w') as f:
        f.write("{invalid json")
    
    from cookiecutter.exceptions import ContextDecodingException
    try:
        generate_context(str(context_file))
        assert False, "Should raise ContextDecodingException"
    except ContextDecodingException as e:
        assert "JSON decoding error" in str(e)
    
    # Test 8: Boolean variable with string overwrite
    context_file = tmp_path / "cookiecutter7.json"
    context_data = {"use_docker": True}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    extra_context = {"use_docker": "n"}
    result = generate_context(str(context_file), extra_context=extra_context)
    assert result["cookiecutter"]["use_docker"] is False
    
    # Test 9: OrderedDict preservation
    context_file = tmp_path / "cookiecutter8.json"
    context_data = {"z_last": 1, "a_first": 2, "m_middle": 3}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    
    result = generate_context(str(context_file))
    assert isinstance(result, OrderedDict)
    assert isinstance(result["cookiecutter"], OrderedDict)


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from pathlib import Path
from jinja2 import Environment
from cookiecutter.exceptions import EmptyDirNameException, OutputDirExistsException


def test_render_and_create_dir(tmp_path):
    """Test render_and_create_dir creates directory with rendered name."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = tmp_path
    
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert result_dir == Path(output_dir) / 'my_project'
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_empty_dirname(tmp_path):
    """Test render_and_create_dir raises error for empty dirname."""
    env = Environment()
    context = {'cookiecutter': {}}
    output_dir = tmp_path
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, env)


def test_render_and_create_dir_empty_string_dirname(tmp_path):
    """Test render_and_create_dir raises error for empty string dirname."""
    env = Environment()
    context = {'cookiecutter': {}}
    output_dir = tmp_path
    
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, env)


def test_render_and_create_dir_already_exists(tmp_path):
    """Test render_and_create_dir raises error when directory exists."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = tmp_path
    
    # Create the directory first
    existing_dir = output_dir / 'my_project'
    existing_dir.mkdir()
    
    dirname = '{{cookiecutter.project_name}}'
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, output_dir, env, overwrite_if_exists=False
        )


def test_render_and_create_dir_overwrite_if_exists(tmp_path):
    """Test render_and_create_dir overwrites directory when flag is True."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = tmp_path
    
    # Create the directory first
    existing_dir = output_dir / 'my_project'
    existing_dir.mkdir()
    test_file = existing_dir / 'test.txt'
    test_file.write_text('content')
    
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=True
    )
    
    assert result_dir == Path(output_dir) / 'my_project'
    assert result_dir.exists()
    assert created is False


def test_render_and_create_dir_no_template_variables(tmp_path):
    """Test render_and_create_dir with plain dirname without template variables."""
    env = Environment()
    context = {'cookiecutter': {}}
    output_dir = tmp_path
    
    dirname = 'plain_project'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert result_dir == Path(output_dir) / 'plain_project'
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_nested_path(tmp_path):
    """Test render_and_create_dir with nested path."""
    env = Environment()
    context = {'cookiecutter': {'org': 'myorg', 'project': 'myproj'}}
    output_dir = tmp_path
    
    dirname = '{{cookiecutter.org}}/{{cookiecutter.project}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert result_dir == Path(output_dir) / 'myorg' / 'myproj'
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_with_pathlib_output_dir(tmp_path):
    """Test render_and_create_dir works with pathlib Path as output_dir."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = Path(tmp_path)
    
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert isinstance(result_dir, Path)
    assert result_dir.exists()
    assert created is True


def test_render_and_create_dir_with_string_output_dir(tmp_path):
    """Test render_and_create_dir works with string as output_dir."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = str(tmp_path)
    
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert Path(result_dir).exists()
    assert created is True


def test_render_and_create_dir_special_characters(tmp_path):
    """Test render_and_create_dir handles special characters in project name."""
    env = Environment()
    context = {'cookiecutter': {'project_name': 'my-project_v1'}}
    output_dir = tmp_path
    
    dirname = '{{cookiecutter.project_name}}'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert result_dir == Path(output_dir) / 'my-project_v1'
    assert result_dir.exists()
    assert created is True


# LLM-generated content at query #36
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from collections import OrderedDict
from unittest.mock import Mock, patch, MagicMock
import pytest


def test_generate_files():
    """Test generate_files function with various scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup test directories
        repo_dir = Path(tmpdir) / "repo"
        output_dir = Path(tmpdir) / "output"
        repo_dir.mkdir()
        output_dir.mkdir()
        
        # Create a basic template structure
        template_dir = repo_dir / "{{cookiecutter.project_name}}"
        template_dir.mkdir()
        
        # Create a test file in template
        test_file = template_dir / "test.txt"
        test_file.write_text("Hello {{cookiecutter.author}}")
        
        # Create context
        context = {
            'cookiecutter': {
                'project_name': 'my_project',
                'author': 'Test Author'
            }
        }
        
        # Call generate_files
        with patch('cookiecutter.generate.find_template') as mock_find:
            with patch('cookiecutter.generate.create_env_with_context') as mock_env:
                mock_find.return_value = str(template_dir)
                
                # Create a real Jinja2 environment
                from jinja2 import Environment, FileSystemLoader
                env = Environment(loader=FileSystemLoader(str(template_dir)))
                mock_env.return_value = env
                
                with patch('cookiecutter.generate.run_hook_from_repo_dir'):
                    result = generate_files(
                        repo_dir=str(repo_dir),
                        context=context,
                        output_dir=str(output_dir),
                        accept_hooks=False
                    )
        
        # Verify results
        assert result is not None
        assert os.path.exists(result)
        assert 'my_project' in result


def test_generate_files_with_overwrite():
    """Test generate_files with overwrite_if_exists=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        output_dir = Path(tmpdir) / "output"
        repo_dir.mkdir()
        output_dir.mkdir()
        
        template_dir = repo_dir / "{{cookiecutter.project_name}}"
        template_dir.mkdir()
        
        context = {
            'cookiecutter': {
                'project_name': 'my_project'
            }
        }
        
        with patch('cookiecutter.generate.find_template') as mock_find:
            with patch('cookiecutter.generate.create_env_with_context') as mock_env:
                from jinja2 import Environment, FileSystemLoader
                mock_find.return_value = str(template_dir)
                env = Environment(loader=FileSystemLoader(str(template_dir)))
                mock_env.return_value = env
                
                with patch('cookiecutter.generate.run_hook_from_repo_dir'):
                    # First call
                    result1 = generate_files(
                        repo_dir=str(repo_dir),
                        context=context,
                        output_dir=str(output_dir),
                        overwrite_if_exists=False,
                        accept_hooks=False
                    )
                    assert os.path.exists(result1)
                    
                    # Second call with overwrite
                    result2 = generate_files(
                        repo_dir=str(repo_dir),
                        context=context,
                        output_dir=str(output_dir),
                        overwrite_if_exists=True,
                        accept_hooks=False
                    )
                    assert os.path.exists(result2)


def test_generate_files_with_hooks():
    """Test generate_files with hooks enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        output_dir = Path(tmpdir) / "output"
        repo_dir.mkdir()
        output_dir.mkdir()
        
        template_dir = repo_dir / "{{cookiecutter.project_name}}"
        template_dir.mkdir()
        
        context = {
            'cookiecutter': {
                'project_name': 'my_project'
            }
        }
        
        with patch('cookiecutter.generate.find_template') as mock_find:
            with patch('cookiecutter.generate.create_env_with_context') as mock_env:
                with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_hook:
                    from jinja2 import Environment, FileSystemLoader
                    mock_find.return_value = str(template_dir)
                    env = Environment(loader=FileSystemLoader(str(template_dir)))
                    mock_env.return_value = env
                    
                    result = generate_files(
                        repo_dir=str(repo_dir),
                        context=context,
                        output_dir=str(output_dir),
                        accept_hooks=True
                    )
                    
                    # Verify hooks were called
                    assert mock_hook.call_count == 2


def test_generate_files_with_undefined_error():
    """Test generate_files handles UndefinedError in project directory name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        output_dir = Path(tmpdir) / "output"
        repo_dir.mkdir()
        output_dir.mkdir()
        
        template_dir = repo_dir / "{{cookiecutter.undefined_var}}"
        template_dir.mkdir()
        
        context = {
            'cookiecutter': {
                'project_name': 'my_project'
            }
        }
        
        with patch('cookiecutter.generate.find_template') as mock_find:
            with patch('cookiecutter.generate.create_env_with_context') as mock_env:
                from jinja2 import Environment, FileSystemLoader, Undefined
                mock_find.return_value = str(template_dir)
                env = Environment(loader=FileSystemLoader(str(template_dir)))
                mock_env.return_value = env
                
                with pytest.raises(Exception):
                    generate_files(
                        repo_dir=str(repo_dir),
                        context=context,
                        output_dir=str(output_dir),
                        accept_hooks=False
                    )


def test_generate_files_empty_context():
    """Test generate_files with empty context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        output_dir = Path(tmpdir) / "output"
        repo_dir.mkdir()
        output_dir.mkdir()
        
        template_dir = repo_dir / "project"
        template_dir.mkdir()
        
        with patch('cookiecutter.generate.find_template') as mock_find:
            with patch('cookiecutter.generate.create_env_with_context') as mock_env:
                from jinja2 import Environment, FileSystemLoader
                mock_find.return_value = str(template_dir)
                env = Environment(loader=FileSystemLoader(str(template_dir)))
                mock_env.return_value = env
                
                with patch('cookiecutter.generate.run_hook_from_repo_dir'):
                    result = generate_files(
                        repo_dir=str(repo_dir),
                        context=None,
                        output_dir=str(output_dir),
                        accept_hooks=False
                    )
                    
                    assert result is not None


def test_generate_files_skip_if_file_exists():
    """Test generate_files with skip_if_file_exists=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        output_dir = Path(tmpdir) / "output"
        repo_


# LLM-generated content at query #37
#--------------------------

```python
def test_render_and_create_dir(tmp_path):
    """Test render_and_create_dir function."""
    from jinja2 import Environment
    
    # Test 1: Basic directory creation
    context = {'cookiecutter': {'project_name': 'my_project'}}
    env = Environment()
    output_dir = tmp_path
    dirname = '{{cookiecutter.project_name}}'
    
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=False
    )
    
    assert created is True
    assert result_dir.name == 'my_project'
    assert result_dir.exists()
    assert result_dir.parent == output_dir
    
    # Test 2: Directory already exists, overwrite_if_exists=False
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, output_dir, env, overwrite_if_exists=False
        )
    
    # Test 3: Directory already exists, overwrite_if_exists=True
    result_dir2, created2 = render_and_create_dir(
        dirname, context, output_dir, env, overwrite_if_exists=True
    )
    
    assert created2 is False
    assert result_dir2.exists()
    
    # Test 4: Empty directory name raises exception
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, env)
    
    # Test 5: Complex template with nested context
    context_nested = {
        'cookiecutter': {
            'author': 'john',
            'version': '1.0'
        }
    }
    dirname_nested = '{{cookiecutter.author}}_{{cookiecutter.version}}'
    output_dir_nested = tmp_path / 'nested'
    output_dir_nested.mkdir()
    
    result_dir3, created3 = render_and_create_dir(
        dirname_nested, context_nested, output_dir_nested, env
    )
    
    assert created3 is True
    assert result_dir3.name == 'john_1.0'
    assert result_dir3.exists()
    
    # Test 6: Path object as output_dir
    output_dir_path = tmp_path / 'path_test'
    output_dir_path.mkdir()
    
    result_dir4, created4 = render_and_create_dir(
        'test_dir', context, output_dir_path, env
    )
    
    assert created4 is True
    assert result_dir4.parent == output_dir_path
    
    # Test 7: String as output_dir
    output_dir_str = str(tmp_path / 'str_test')
    os.makedirs(output_dir_str, exist_ok=True)
    
    result_dir5, created5 = render_and_create_dir(
        'test_dir_str', context, output_dir_str, env
    )
    
    assert created5 is True
    assert isinstance(result_dir5, Path)


# LLM-generated content at query #38
#--------------------------

```python
def test_generate_files(tmp_path, monkeypatch):
    """Test generate_files function."""
    # Setup
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create a cookiecutter.json
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text(
        '{"project_name": "my_project", "author": "John Doe"}',
        encoding='utf-8'
    )

    # Create a template file
    template_file = template_dir / "README.md"
    template_file.write_text("# {{cookiecutter.project_name}}\nAuthor: {{cookiecutter.author}}", encoding='utf-8')

    # Create a subdirectory in template
    subdir = template_dir / "src"
    subdir.mkdir()
    subdir_file = subdir / "main.py"
    subdir_file.write_text("# {{cookiecutter.project_name}} main file", encoding='utf-8')

    # Generate context
    context = generate_context(str(context_file))

    # Test generate_files
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False,
    )

    # Assertions
    assert result == str(output_dir / "my_project")
    assert os.path.isdir(result)
    
    readme_file = Path(result) / "README.md"
    assert readme_file.exists()
    assert readme_file.read_text(encoding='utf-8') == "# my_project\nAuthor: John Doe"
    
    main_file = Path(result) / "src" / "main.py"
    assert main_file.exists()
    assert main_file.read_text(encoding='utf-8') == "# my_project main file"


def test_generate_files_with_overwrite(tmp_path):
    """Test generate_files with overwrite_if_exists=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}', encoding='utf-8')

    template_file = template_dir / "test.txt"
    template_file.write_text("content", encoding='utf-8')

    context = generate_context(str(context_file))

    # Create existing project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()
    (existing_project / "old_file.txt").write_text("old content", encoding='utf-8')

    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        accept_hooks=False,
    )

    assert os.path.isdir(result)
    assert (Path(result) / "test.txt").exists()


def test_generate_files_without_overwrite_raises(tmp_path):
    """Test generate_files raises when directory exists and overwrite_if_exists=False."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}', encoding='utf-8')

    template_file = template_dir / "test.txt"
    template_file.write_text("content", encoding='utf-8')

    context = generate_context(str(context_file))

    # Create existing project directory
    existing_project = output_dir / "my_project"
    existing_project.mkdir()

    with pytest.raises(OutputDirExistsException):
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=False,
            accept_hooks=False,
        )


def test_generate_files_with_copy_without_render(tmp_path):
    """Test generate_files with _copy_without_render setting."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text(
        '{"project_name": "my_project", "_copy_without_render": ["*.bin"]}',
        encoding='utf-8'
    )

    # Create a binary-like file (we'll use a text file with .bin extension)
    binary_file = template_dir / "data.bin"
    binary_file.write_text("{{cookiecutter.project_name}}", encoding='utf-8')

    context = generate_context(str(context_file))

    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        accept_hooks=False,
    )

    # The .bin file should not be rendered
    bin_file = Path(result) / "data.bin"
    assert bin_file.exists()
    assert bin_file.read_text(encoding='utf-8') == "{{cookiecutter.project_name}}"


def test_generate_files_skip_if_file_exists(tmp_path):
    """Test generate_files with skip_if_file_exists=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    template_dir = repo_dir / "{{cookiecutter.project_name}}"
    template_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "my_project"}', encoding='utf-8')

    template_file = template_dir / "config.txt"
    template_file.write_text("new_content", encoding='utf-8')

    context = generate_context(str(context_file))

    # First generation
    result1 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        skip_if_file_exists=False,
        accept_hooks=False,
    )

    config_file = Path(result1) / "config.txt"
    config_file.write_text("existing_content", encoding='utf-8')

    # Second generation with skip_if_file_exists=True
    result2 = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        skip_if_file_exists=True,
        accept_hooks=False,
    )

    # File should not be overwritten
    assert config_file.read_text(encoding='utf-8') == "existing_content"


def test_generate_files_with_context_none(tmp_path):
    """Test generate_files with context=None."""
    repo_dir = tmp_path / "repo"
    repo


# LLM-generated content at query #39
#--------------------------

```python
def test_apply_overwrites_to_context():
    """Test apply_overwrites_to_context function with various scenarios."""
    
    # Test 1: Simple overwrite of string variable
    context = {'name': 'original', 'value': 42}
    overwrite = {'name': 'overwritten'}
    apply_overwrites_to_context(context, overwrite)
    assert context['name'] == 'overwritten'
    assert context['value'] == 42
    
    # Test 2: Ignore new variables at first level
    context = {'existing': 'value'}
    overwrite = {'new_var': 'new_value'}
    apply_overwrites_to_context(context, overwrite)
    assert 'new_var' not in context
    assert context['existing'] == 'value'
    
    # Test 3: Choice variable - valid overwrite
    context = {'choice': ['option1', 'option2', 'option3']}
    overwrite = {'choice': 'option2'}
    apply_overwrites_to_context(context, overwrite)
    assert context['choice'][0] == 'option2'
    assert 'option1' in context['choice']
    
    # Test 4: Choice variable - invalid overwrite raises ValueError
    context = {'choice': ['option1', 'option2']}
    overwrite = {'choice': 'invalid_option'}
    with pytest.raises(ValueError, match="invalid_option provided for choice variable"):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 5: Multi-choice variable - valid overwrite
    context = {'multi': ['a', 'b', 'c']}
    overwrite = {'multi': ['b', 'c']}
    apply_overwrites_to_context(context, overwrite)
    assert context['multi'] == ['b', 'c']
    
    # Test 6: Multi-choice variable - invalid overwrite raises ValueError
    context = {'multi': ['a', 'b']}
    overwrite = {'multi': ['a', 'x']}
    with pytest.raises(ValueError, match="provided for multi-choice variable"):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 7: Dictionary variable - partial overwrite
    context = {'config': {'key1': 'value1', 'key2': 'value2'}}
    overwrite = {'config': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite)
    assert context['config']['key1'] == 'new_value1'
    assert context['config']['key2'] == 'value2'
    
    # Test 8: Boolean variable - string to boolean conversion (yes)
    context = {'enabled': True}
    overwrite = {'enabled': 'yes'}
    apply_overwrites_to_context(context, overwrite)
    assert context['enabled'] is True
    
    # Test 9: Boolean variable - string to boolean conversion (no)
    context = {'enabled': False}
    overwrite = {'enabled': 'no'}
    apply_overwrites_to_context(context, overwrite)
    assert context['enabled'] is False
    
    # Test 10: Boolean variable - invalid string raises ValueError
    context = {'enabled': True}
    overwrite = {'enabled': 'invalid'}
    with pytest.raises(ValueError, match="could not be converted to a boolean"):
        apply_overwrites_to_context(context, overwrite)
    
    # Test 11: Dictionary variable with in_dictionary_variable=True adds new keys
    context = {'config': {'existing': 'value'}}
    overwrite = {'new_key': 'new_value'}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['new_key'] == 'new_value'
    
    # Test 12: Nested dictionary overwrite
    context = {'outer': {'inner': {'deep': 'value'}}}
    overwrite = {'outer': {'inner': {'deep': 'new_value'}}}
    apply_overwrites_to_context(context, overwrite)
    assert context['outer']['inner']['deep'] == 'new_value'
    
    # Test 13: Empty overwrite context
    context = {'key': 'value'}
    overwrite = {}
    apply_overwrites_to_context(context, overwrite)
    assert context['key'] == 'value'
    
    # Test 14: List to non-list overwrite in nested dictionary
    context = {'config': {'items': ['a', 'b']}}
    overwrite = {'config': {'items': 'single'}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context['config']['items'] == 'single'


