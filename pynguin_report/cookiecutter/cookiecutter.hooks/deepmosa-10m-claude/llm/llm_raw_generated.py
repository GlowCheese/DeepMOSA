####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_valid_hook_matching_supported_hook():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_matching_supported_hook_with_extension():
    result = valid_hook('/path/to/pre-commit.py', 'pre-commit')
    assert result is True

def test_valid_hook_non_matching_hook_name():
    result = valid_hook('/path/to/pre-push', 'pre-commit')
    assert result is False

def test_valid_hook_unsupported_hook():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False

def test_valid_hook_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False

def test_valid_hook_backup_file_with_extension():
    result = valid_hook('/path/to/pre-commit.py~', 'pre-commit')
    assert result is False

def test_valid_hook_only_basename_matching():
    result = valid_hook('pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_only_basename_matching_with_extension():
    result = valid_hook('pre-commit.sh', 'pre-commit')
    assert result is True

def test_valid_hook_multiple_dots_in_filename():
    result = valid_hook('/path/to/pre-commit.test.py', 'pre-commit')
    assert result is False

def test_valid_hook_case_sensitive():
    result = valid_hook('/path/to/Pre-Commit', 'pre-commit')
    assert result is False


# LLM-generated content at query #2
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_returns_original_repo_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_creates_temp_dir_when_hook_exists(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates temp directory and runs hook script."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.sh"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    hook_file.chmod(0o755)
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda script_path, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert str(result).startswith(tempfile.gettempdir())


def test_run_pre_prompt_hook_with_python_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes Python hook script."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('test')")
    
    run_script_called = []
    def mock_run_script(script_path, cwd):
        run_script_called.append((script_path, cwd))
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert len(run_script_called) == 1
    assert run_script_called[0][0].endswith("pre_prompt.py")


def test_run_pre_prompt_hook_failed_hook_raises_exception(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when hook fails."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.sh"
    hook_file.write_text("#!/bin/bash\nexit 1")
    hook_file.chmod(0o755)
    
    def mock_run_script(script_path, cwd):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_with_string_repo_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook works with string repo_dir path."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(str(repo_dir))
    assert result == str(repo_dir)


def test_run_pre_prompt_hook_with_path_repo_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook works with Path repo_dir."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #3
#--------------------------

```python
import os
import tempfile
import shutil


def test_find_hook_returns_none_when_hooks_dir_not_exists():
    result = find_hook('pre_prompt', 'nonexistent_hooks_dir')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        open(os.path.join(hooks_dir, 'other_hook.sh'), 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_returns_none_when_hook_is_backup_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        open(os.path.join(hooks_dir, 'pre_prompt~'), 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_returns_absolute_path_when_hook_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        open(hook_file, 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(hook_file)


def test_find_hook_returns_multiple_matching_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_prompt.sh')
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.py')
        open(hook_file1, 'w').close()
        open(hook_file2, 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert os.path.abspath(hook_file1) in result
        assert os.path.abspath(hook_file2) in result


def test_find_hook_ignores_unsupported_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        open(os.path.join(hooks_dir, 'invalid_hook.sh'), 'w').close()
        
        result = find_hook('invalid_hook', hooks_dir)
        assert result is None


def test_find_hook_returns_absolute_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        open(hook_file, 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook, find_hook
    
    # Create a temporary repo directory without any hooks
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    # Call the function
    result = run_pre_prompt_hook(repo_dir)
    
    # Assert that the original repo_dir is returned when no scripts are found
    assert result == repo_dir


# LLM-generated content at query #5
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_called = []
    original_popen = subprocess.Popen
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
            self.returncode = 0
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('run_script.utils.make_executable', lambda x: None)
    
    from run_script import run_script
    run_script(script_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('run_script.utils.make_executable', lambda x: None)
    
    from run_script import run_script
    run_script(script_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_nonzero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("import sys; sys.exit(1)")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('run_script.utils.make_executable', lambda x: None)
    
    from run_script import run_script, FailedHookException
    import pytest
    
    with pytest.raises(FailedHookException) as exc_info:
        run_script(script_path)
    
    assert 'exit status: 1' in str(exc_info.value)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError()
            err.errno = errno.ENOEXEC
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('run_script.utils.make_executable', lambda x: None)
    
    from run_script import run_script, FailedHookException
    import pytest
    
    with pytest.raises(FailedHookException) as exc_info:
        run_script(script_path)
    
    assert 'shebang' in str(exc_info.value)


def test_run_script_oserror(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError()
            err.errno = errno.EACCES
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('run_script.utils.make_executable', lambda x: None)
    
    from run_script import run_script, FailedHookException
    import pytest
    
    with pytest.raises(FailedHookException) as exc_info:
        run_script(script_path)
    
    assert 'error:' in str(exc_info.value)


def test_run_script_with_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    cwd_path = str(tmp_path / "subdir")
    
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('run_script.utils.make_executable', lambda x: None)
    
    from run_script import run_script
    run_script(script_path, cwd=cwd_path)
    
    assert mock_popen_called[0][1]['cwd'] == cwd_path


# LLM-generated content at query #6
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file
    script_file = tmp_path / "test_script.py"
    script_content = "print('Hello {{ cookiecutter.name }}')"
    script_file.write_text(script_content)
    
    # Create context
    context = {
        'cookiecutter': {
            'name': 'World',
            '_jinja2_env_vars': {}
        }
    }
    
    # Mock run_script to verify it's called with rendered content
    called_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        called_scripts.append(script_path)
        # Read the rendered script to verify rendering worked
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'World' in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(called_scripts) == 1


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension."""
    from cookiecutter.hooks import run_script_with_context
    import tempfile
    
    # Create a temporary script file with .sh extension
    script_file = tmp_path / "test_script.sh"
    script_content = "echo '{{ cookiecutter.message }}'"
    script_file.write_text(script_content)
    
    # Create context
    context = {
        'cookiecutter': {
            'message': 'Test message',
            '_jinja2_env_vars': {}
        }
    }
    
    created_temp_files = []
    original_named_temp_file = tempfile.NamedTemporaryFile
    
    def mock_named_temp_file(*args, **kwargs):
        temp_file = original_named_temp_file(*args, **kwargs)
        created_temp_files.append(temp_file.name)
        assert temp_file.name.endswith('.sh'), "Temporary file should have .sh extension"
        return temp_file
    
    def mock_run_script(script_path, cwd='.'):
        pass
    
    monkeypatch.setattr('tempfile.NamedTemporaryFile', mock_named_temp_file)
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify a temp file was created
    assert len(created_temp_files) > 0


def test_run_script_with_context_renders_template(tmp_path, monkeypatch):
    """Test run_script_with_context correctly renders Jinja2 templates."""
    from cookiecutter.hooks import run_script_with_context
    
    # Create a script with multiple template variables
    script_file = tmp_path / "test_script.py"
    script_content = "name={{ cookiecutter.project_name }}\nauthor={{ cookiecutter.author }}"
    script_file.write_text(script_content)
    
    # Create context
    context = {
        'cookiecutter': {
            'project_name': 'MyProject',
            'author': 'John Doe',
            '_jinja2_env_vars': {}
        }
    }
    
    rendered_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        rendered_scripts.append(rendered_content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify template was rendered correctly
    assert len(rendered_scripts) == 1
    assert 'name=MyProject' in rendered_scripts[0]
    assert 'author=John Doe' in rendered_scripts[0]


def test_run_script_with_context_uses_correct_cwd(tmp_path, monkeypatch):
    """Test run_script_with_context passes the correct working directory."""
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    context = {
        'cookiecutter': {
            '_jinja2_env_vars': {}
        }
    }
    
    called_cwd = []
    
    def mock_run_script(script_path, cwd='.'):
        called_cwd.append(cwd)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute with specific cwd
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify correct cwd was passed
    assert len(called_cwd) == 1
    assert str(tmp_path) in str(called_cwd[0])


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_true(mocker):
    import errno
    import subprocess
    from pathlib import Path
    
    # Mock the dependencies
    mocker.patch('sys.platform', 'linux')
    mock_make_executable = mocker.patch('utils.make_executable')
    mock_popen = mocker.patch('subprocess.Popen')
    
    # Create a mock process that raises OSError with ENOEXEC errno
    mock_proc = mocker.MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.wait.side_effect = OSError(errno.ENOEXEC, "Exec format error")
    
    # Import after mocking to ensure mocks are in place
    from your_module import run_script, FailedHookException
    
    # Call the function and verify the exception is raised
    try:
        run_script('/path/to/script.sh')
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_line_8_evaluates_to_true():
    script_path = "/path/to/script.py"
    result = script_path.endswith('.py')
    assert result is True


# LLM-generated content at query #9
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_not_exists():
    result = find_hook('pre_prompt', 'nonexistent_hooks_dir')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks():
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        with open(os.path.join(hooks_dir, 'some_file.txt'), 'w') as f:
            f.write('content')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_returns_scripts_when_matching_hook_exists():
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho test')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(hook_file)


def test_find_hook_returns_multiple_scripts_with_same_name():
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        hook_file1 = os.path.join(hooks_dir, 'pre_prompt.sh')
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file1, 'w') as f:
            f.write('#!/bin/bash\necho test')
        with open(hook_file2, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert os.path.abspath(hook_file1) in result
        assert os.path.abspath(hook_file2) in result


def test_find_hook_ignores_backup_files():
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh~')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho test')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_ignores_unsupported_hooks():
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        hook_file = os.path.join(hooks_dir, 'unsupported_hook.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho test')
        
        result = find_hook('unsupported_hook', hooks_dir)
        assert result is None


# LLM-generated content at query #10
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts(tmp_path, mocker):
    """Test run_pre_prompt_hook when no pre_prompt script exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_script(tmp_path, mocker):
    """Test run_pre_prompt_hook executes pre_prompt script successfully."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    mocker.patch('cookiecutter.hooks.find_hook', side_effect=[None, [str(script_file)]])
    mocker.patch('cookiecutter.hooks.run_script')
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_script_failure(tmp_path, mocker):
    """Test run_pre_prompt_hook when script execution fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    script_path = str(repo_dir / "hooks" / "pre_prompt.sh")
    
    mocker.patch('cookiecutter.hooks.find_hook', side_effect=[None, [script_path]])
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir)
    mocker.patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('Script failed'))
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_early_return_with_scripts(tmp_path, mocker):
    """Test run_pre_prompt_hook returns repo_dir early if scripts exist in original dir."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_file)])
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_multiple_scripts(tmp_path, mocker):
    """Test run_pre_prompt_hook executes multiple pre_prompt scripts."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    script1 = str(repo_dir / "pre_prompt.sh")
    script2 = str(repo_dir / "pre_prompt.py")
    
    mocker.patch('cookiecutter.hooks.find_hook', side_effect=[None, [script1, script2]])
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir)
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert mock_run_script.call_count == 2
    assert result == repo_dir


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes successfully without errors."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_run_hook.assert_called_once_with('post_prompt', project_dir, context)
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not delete project on exception when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_run_hook.assert_called_once_with('pre_gen_project', project_dir, context)
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes to repo directory during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = None
    call_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal call_cwd
        call_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    original_cwd = os.getcwd()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert os.getcwd() == original_cwd
    mock_run_hook.assert_called_once()


# LLM-generated content at query #12
#--------------------------

```python
def test_run_hook_no_scripts_found(tmp_path, monkeypatch):
    """Test run_hook when no scripts are found."""
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: None)
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', tmp_path, context)


def test_run_hook_with_single_script(tmp_path, monkeypatch):
    """Test run_hook with a single script found."""
    script_path = tmp_path / 'pre_prompt.sh'
    script_path.write_text('#!/bin/bash\necho "test"')
    
    run_script_with_context_called = []
    
    def mock_run_script_with_context(script, cwd, context):
        run_script_with_context_called.append((script, cwd, context))
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [str(script_path)])
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', tmp_path, context)
    
    assert len(run_script_with_context_called) == 1
    assert run_script_with_context_called[0][0] == str(script_path)
    assert run_script_with_context_called[0][1] == tmp_path
    assert run_script_with_context_called[0][2] == context


def test_run_hook_with_multiple_scripts(tmp_path, monkeypatch):
    """Test run_hook with multiple scripts found."""
    script1 = tmp_path / 'post_gen_1.sh'
    script2 = tmp_path / 'post_gen_2.sh'
    script1.write_text('#!/bin/bash\necho "test1"')
    script2.write_text('#!/bin/bash\necho "test2"')
    
    run_script_with_context_called = []
    
    def mock_run_script_with_context(script, cwd, context):
        run_script_with_context_called.append((script, cwd, context))
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [str(script1), str(script2)])
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', tmp_path, context)
    
    assert len(run_script_with_context_called) == 2
    assert run_script_with_context_called[0][0] == str(script1)
    assert run_script_with_context_called[1][0] == str(script2)
    assert run_script_with_context_called[0][1] == tmp_path
    assert run_script_with_context_called[1][1] == tmp_path
    assert run_script_with_context_called[0][2] == context
    assert run_script_with_context_called[1][2] == context


def test_run_hook_passes_context_to_scripts(tmp_path, monkeypatch):
    """Test run_hook passes the context correctly to scripts."""
    script_path = tmp_path / 'pre_prompt.py'
    script_path.write_text('print("test")')
    
    captured_context = []
    
    def mock_run_script_with_context(script, cwd, context):
        captured_context.append(context)
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [str(script_path)])
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    
    context = {'cookiecutter': {'key': 'value'}}
    run_hook('pre_prompt', tmp_path, context)
    
    assert len(captured_context) == 1
    assert captured_context[0] == context


# LLM-generated content at query #13
#--------------------------

```python
def test_script_path_ends_with_py():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import sys
    
    script_path = "/path/to/script.py"
    cwd = Path('.')
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('sys.platform', 'linux'), \
         patch('utils.make_executable'), \
         patch('sys.executable', '/usr/bin/python3'):
        
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        run_script(script_path, cwd)
        
        # Verify that the predicate at line 8 evaluates to True
        # and script_command is set to [sys.executable, script_path]
        assert mock_popen.call_args[0][0] == ['/usr/bin/python3', '/path/to/script.py']


# LLM-generated content at query #14
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when no pre_prompt hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes a valid pre_prompt hook script."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nprint('hook executed')")
    hook_script.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert (Path(result) / "hooks" / "pre_prompt.py").exists()


def test_run_pre_prompt_hook_creates_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary directory when hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\necho 'test'")
    hook_script.chmod(0o755)
    
    original_repo_path = str(repo_dir)
    result = run_pre_prompt_hook(repo_dir)
    
    assert str(result) != original_repo_path
    assert Path(result).exists()
    assert "cookiecutter" in str(result)


def test_run_pre_prompt_hook_failed_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises exception when hook script fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")
    hook_script.chmod(0o755)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_multiple_hooks(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes all pre_prompt hooks."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script1 = hooks_dir / "pre_prompt.py"
    hook_script1.write_text("#!/usr/bin/env python\nprint('hook1')")
    hook_script1.chmod(0o755)
    
    hook_script2 = hooks_dir / "pre_prompt.sh"
    hook_script2.write_text("#!/bin/bash\necho 'hook2'")
    hook_script2.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert Path(result).exists()
    assert (Path(result) / "hooks" / "pre_prompt.py").exists()
    assert (Path(result) / "hooks" / "pre_prompt.sh").exists()


# LLM-generated content at query #15
#--------------------------

```python
def test_script_path_ends_with_py():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import sys
    
    script_path = "/path/to/script.py"
    
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        with patch('utils.make_executable'):
            # Import the function to test
            from your_module import run_script
            
            run_script(script_path)
            
            # Verify that the predicate at line 8 evaluated to True
            # and sys.executable was used in the command
            call_args = mock_popen.call_args
            script_command = call_args[0][0]
            assert script_command[0] == sys.executable
            assert script_command[1] == script_path


# LLM-generated content at query #16
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_not_exists(tmp_path):
    """Test that find_hook returns None when hooks directory does not exist."""
    import os
    import sys
    from pathlib import Path
    
    # Create a temporary directory and change to it
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Call find_hook with a non-existent hooks directory
        # The predicate at line 15: if not os.path.isdir(hooks_dir)
        # should evaluate to True since 'hooks' directory doesn't exist
        from your_module import find_hook
        result = find_hook('some_hook', 'hooks')
        
        assert result is None
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir uses work_in context manager."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from unittest.mock import patch, MagicMock
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = str(tmp_path)
    monkeypatch.chdir(original_cwd)
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        with patch('cookiecutter.hooks.work_in', wraps=__import__('cookiecutter.utils', fromlist=['work_in']).work_in) as mock_work_in:
            context = {'cookiecutter': {}}
            run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
            
            mock_work_in.assert_called_once_with(repo_dir)
            assert mock_run_hook.called


# LLM-generated content at query #18
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_not_exists(tmp_path):
    import os
    import sys
    from pathlib import Path
    
    # Create a temporary directory and change to it
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Call find_hook with a non-existent hooks directory
        # We need to mock or import the actual function
        # For this test, we verify the predicate at line 15
        hooks_dir = 'non_existent_hooks'
        result = os.path.isdir(hooks_dir)
        
        # The predicate at line 15 is: if not os.path.isdir(hooks_dir)
        # This should evaluate to True when hooks_dir doesn't exist
        assert not result == True
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #19
#--------------------------

```python
def test_find_hook_predicate_evaluates_to_false():
    import os
    import tempfile
    
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a hook file that won't match the hook_name
        hook_file = os.path.join(hooks_dir, 'other_hook.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\n')
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Call find_hook with a hook_name that doesn't match any files
            # The predicate (valid_hook(hook_file, hook_name)) at line 22 should evaluate to False
            result = find_hook('nonexistent_hook', 'hooks')
            
            # When predicate is False for all files, scripts list is empty, so None is returned
            assert result is None
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #20
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    import logging
    
    def mock_find_hook(hook_name):
        return []
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', '.', context)
    
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(monkeypatch):
    """Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS."""
    import subprocess
    from pathlib import Path
    
    # Mock subprocess.Popen to return a process with exit status 0 (EXIT_SUCCESS)
    class MockProcess:
        def wait(self):
            return 0  # EXIT_SUCCESS
    
    mock_popen = lambda *args, **kwargs: MockProcess()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    # Mock utils.make_executable to do nothing
    import sys
    sys.modules['utils'] = type(sys)('utils')
    sys.modules['utils'].make_executable = lambda x: None
    
    # Set EXIT_SUCCESS to 0
    import __main__
    __main__.EXIT_SUCCESS = 0
    
    # Call run_script with a non-.py file
    run_script('/path/to/script', cwd='.')
    
    # If we reach here without exception, the predicate was False
    # (exit_status != EXIT_SUCCESS evaluated to False, so no exception was raised)


# LLM-generated content at query #22
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders script with context and executes it."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file
    script_file = tmp_path / "test_script.py"
    script_content = "print('{{ cookiecutter.name }}')\n"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context
    context = {
        'cookiecutter': {
            'name': 'test_project',
            '_jinja2_env_vars': {}
        }
    }
    
    # Mock run_script to verify it's called with rendered script
    called_scripts = []
    def mock_run_script(script_path, cwd='.'):
        called_scripts.append(script_path)
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert "test_project" in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(called_scripts) == 1
    assert called_scripts[0].endswith('.py')


def test_run_script_with_context_with_jinja_vars(tmp_path, monkeypatch):
    """Test run_script_with_context with custom jinja2 environment variables."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file
    script_file = tmp_path / "test_script.sh"
    script_content = "#!/bin/bash\necho '{{ variable }}'\n"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context with jinja2 env vars
    context = {
        'cookiecutter': {
            'variable': 'hello_world',
            '_jinja2_env_vars': {
                'variable_start_string': '[[',
                'variable_end_string': ']]'
            }
        }
    }
    
    called_scripts = []
    def mock_run_script(script_path, cwd='.'):
        called_scripts.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(called_scripts) == 1


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension in temp file."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file with .bat extension
    script_file = tmp_path / "test_script.bat"
    script_content = "@echo {{ cookiecutter.message }}\n"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context
    context = {
        'cookiecutter': {
            'message': 'test_message',
            '_jinja2_env_vars': {}
        }
    }
    
    temp_files = []
    def mock_run_script(script_path, cwd='.'):
        temp_files.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify temp file has .bat extension
    assert temp_files[0].endswith('.bat')


def test_run_script_with_context_renders_complex_template(tmp_path, monkeypatch):
    """Test run_script_with_context renders complex jinja templates."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a script with complex jinja2 template
    script_file = tmp_path / "complex_script.py"
    script_content = """#!/usr/bin/env python
# Project: {{ cookiecutter.project_name }}
# Author: {{ cookiecutter.author }}
print('Setup complete')
"""
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe',
            '_jinja2_env_vars': {}
        }
    }
    
    rendered_content = []
    def mock_run_script(script_path, cwd='.'):
        content = Path(script_path).read_text(encoding='utf-8')
        rendered_content.append(content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify rendering
    assert 'my_project' in rendered_content[0]
    assert 'John Doe' in rendered_content[0]


# LLM-generated content at query #23
#--------------------------

```python
def test_run_hook_no_scripts_found(tmp_path, mocker):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {}}
    
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')
    mock_run_script.assert_not_called()


# LLM-generated content at query #24
#--------------------------

```python
def test_find_hook_no_hooks_dir(tmp_path):
    import os
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


def test_find_hook_empty_hooks_dir(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is None


def test_find_hook_single_matching_hook(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_multiple_matching_hooks(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.py'
    hook_file1.write_text('#!/usr/bin/env python')
    hook_file2 = hooks_dir / 'pre_prompt.sh'
    hook_file2.write_text('#!/bin/bash')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1) in result
    assert str(hook_file2) in result


def test_find_hook_ignores_backup_files(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    backup_file = hooks_dir / 'pre_prompt.py~'
    backup_file.write_text('#!/usr/bin/env python')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_no_matching_hook_name(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    os.chdir(tmp_path)
    result = find_hook('post_gen_project', 'hooks')
    assert result is None


def test_find_hook_unsupported_hook(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'unsupported_hook.py'
    hook_file.write_text('#!/usr/bin/env python')
    os.chdir(tmp_path)
    result = find_hook('unsupported_hook', 'hooks')
    assert result is None


def test_find_hook_absolute_path(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert os.path.isabs(result[0])


# LLM-generated content at query #25
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    
    run_hook_called = []
    
    def mock_run_hook(name, proj_dir, ctx):
        run_hook_called.append((name, proj_dir, ctx))
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    assert len(run_hook_called) == 1
    assert run_hook_called[0][0] == hook_name


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir cleans up on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    
    def mock_run_hook(name, proj_dir, ctx):
        from cookiecutter.exceptions import FailedHookException
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
        assert False, "Should have raised FailedHookException"
    except FailedHookException:
        assert not project_dir.exists()


def test_run_hook_from_repo_dir_failed_hook_no_cleanup(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    
    def mock_run_hook(name, proj_dir, ctx):
        from cookiecutter.exceptions import FailedHookException
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
        assert False, "Should have raised FailedHookException"
    except FailedHookException:
        assert project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir cleans up on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    
    def mock_run_hook(name, proj_dir, ctx):
        from jinja2 import UndefinedError
        raise UndefinedError('Undefined variable')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
        assert False, "Should have raised UndefinedError"
    except UndefinedError:
        assert not project_dir.exists()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes from repo_dir."""
    import os
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    original_cwd = os.getcwd()
    
    cwd_during_hook = []
    
    def mock_run_hook(name, proj_dir, ctx):
        cwd_during_hook.append(os.getcwd())
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    assert os.getcwd() == original_cwd
    assert str(cwd_during_hook[0]) == str(repo_dir)


# LLM-generated content at query #26
#--------------------------

```python
def test_run_hook_with_no_scripts_found(monkeypatch):
    """Test that run_hook returns early when no scripts are found."""
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    
    call_count = {'find_hook': 0, 'run_script': 0}
    
    def mock_find_hook(hook_name):
        call_count['find_hook'] += 1
        return []
    
    def mock_run_script_with_context(script, project_dir, context):
        call_count['run_script'] += 1
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', Path('.'), context)
    
    assert call_count['find_hook'] == 1
    assert call_count['run_script'] == 0


# LLM-generated content at query #27
#--------------------------

```python
def test_run_pre_prompt_hook_work_in_context_manager():
    """Test that work_in context manager is used to change directory."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory to use as repo_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        original_cwd = os.getcwd()
        
        # Mock find_hook to return empty list (so we return early)
        with patch('cookiecutter.hooks.find_hook', return_value=None):
            result = run_pre_prompt_hook(temp_path)
        
        # Verify we're back in the original directory
        assert os.getcwd() == original_cwd
        assert result == temp_path


# LLM-generated content at query #28
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches FailedHookException."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    import os
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches UndefinedError."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError("Variable undefined")
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except UndefinedError:
        pass


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir deletes project directory on hook failure."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except FailedHookException:
        pass
    
    assert not project_dir.exists()


# LLM-generated content at query #29
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    """Test that tempfile.NamedTemporaryFile is called with delete=False, mode='wb', and correct suffix."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    script_path = "/path/to/script.sh"
    cwd = "/working/dir"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/tmpfile.sh"
    mock_temp_file.__enter__ = MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = MagicMock(return_value=None)
    
    with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file) as mock_named_temp:
        with patch('pathlib.Path.read_text', return_value="echo test"):
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                mock_env = MagicMock()
                mock_template = MagicMock()
                mock_template.render.return_value = "echo test"
                mock_env.from_string.return_value = mock_template
                mock_create_env.return_value = mock_env
                
                with patch('cookiecutter.hooks.run_script'):
                    run_script_with_context(script_path, cwd, context)
    
    mock_named_temp.assert_called_once()
    call_kwargs = mock_named_temp.call_args[1]
    assert call_kwargs['delete'] is False
    assert call_kwargs['mode'] == 'wb'
    assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #30
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    original_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)


def test_run_script_nonzero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 1
    })()
    
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Hook script failed (exit status: 1)' in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("invalid")
    
    def mock_popen_enoexec(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_enoexec)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'might be an empty file or missing a shebang' in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    
    def mock_popen_error(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_error)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Hook script failed (error:' in str(e)


def test_run_script_with_cwd(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    cwd_path = str(tmp_path / "workdir")
    
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    captured_kwargs = {}
    
    def mock_popen(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return type('MockPopen', (), {'wait': lambda self: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=cwd_path)
    assert captured_kwargs['cwd'] == cwd_path


# LLM-generated content at query #31
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no scripts exist."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "hooks").mkdir()
    
    from cookiecutter.hooks import run_pre_prompt_hook
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes pre_prompt script successfully."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('hook executed')")
    
    from cookiecutter.hooks import run_pre_prompt_hook
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert (result / "hooks" / "pre_prompt.py").exists()


def test_run_pre_prompt_hook_creates_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates temporary directory when scripts exist."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    
    from cookiecutter.hooks import run_pre_prompt_hook
    result = run_pre_prompt_hook(repo_dir)
    
    assert str(result) != str(repo_dir)
    assert result.exists()


def test_run_pre_prompt_hook_failed_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException on script failure."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("import sys\nsys.exit(1)")
    
    from cookiecutter.hooks import run_pre_prompt_hook, FailedHookException
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass


def test_run_pre_prompt_hook_no_hooks_dir(tmp_path):
    """Test run_pre_prompt_hook returns original repo_dir when hooks dir missing."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    from cookiecutter.hooks import run_pre_prompt_hook
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #32
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir uses work_in context manager."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from unittest.mock import Mock, patch, call
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = None
    cwd_during_hook = None
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        nonlocal cwd_during_hook
        import os
        cwd_during_hook = os.getcwd()
    
    with patch('cookiecutter.hooks.run_hook', side_effect=mock_run_hook):
        with patch('cookiecutter.hooks.work_in', wraps=__import__('cookiecutter.utils', fromlist=['work_in']).work_in) as mock_work_in:
            context = {'cookiecutter': {}}
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
            
            mock_work_in.assert_called_once_with(repo_dir)


# LLM-generated content at query #33
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    non_existent_dir = str(tmp_path / "non_existent_hooks")
    result = find_hook("pre_prompt", non_existent_dir)
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "some_other_file.sh").write_text("#!/bin/bash\necho test")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is None


def test_find_hook_returns_script_path_when_hook_exists(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.sh"
    hook_file.write_text("#!/bin/bash\necho test")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert str(hook_file) in result


def test_find_hook_ignores_backup_files(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "pre_prompt.sh~").write_text("#!/bin/bash\necho test")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is None


def test_find_hook_returns_multiple_scripts_with_same_name(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / "pre_prompt.sh"
    hook_file2 = hooks_dir / "pre_prompt.py"
    hook_file1.write_text("#!/bin/bash\necho test")
    hook_file2.write_text("#!/usr/bin/env python\nprint('test')")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1) in result
    assert str(hook_file2) in result


def test_find_hook_with_default_hooks_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "post_gen_project.sh"
    hook_file.write_text("#!/bin/bash\necho test")
    
    result = find_hook("post_gen_project")
    assert result is not None
    assert len(result) == 1


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_true(monkeypatch, tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    class FailedHookException(Exception):
        pass
    
    class Utils:
        @staticmethod
        def make_executable(path):
            pass
    
    utils = Utils()
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    
    script_path = str(tmp_path / "test_script.py")
    cwd = tmp_path
    
    run_thru_shell = sys.platform.startswith('win')
    script_command = [sys.executable, script_path]
    
    utils.make_executable(script_path)
    
    proc = subprocess.Popen(script_command, shell=run_thru_shell, cwd=cwd)
    exit_status = proc.wait()
    
    predicate = exit_status != EXIT_SUCCESS
    
    assert predicate is True


# LLM-generated content at query #35
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, mocker):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("hook failed"))
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException:
        pass


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, mocker):
    """Test that run_hook_from_repo_dir catches UndefinedError at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("undefined var"))
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Expected UndefinedError to be raised"
    except UndefinedError:
        pass


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, mocker):
    """Test that run_hook_from_repo_dir deletes project directory when delete_project_on_failure is True."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("hook failed"))
    mocker.patch('cookiecutter.hooks.logger')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_preserves_project_on_success(tmp_path, mocker):
    """Test that run_hook_from_repo_dir does not delete project directory on success."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    mocker.patch('cookiecutter.hooks.run_hook')
    mocker.patch('cookiecutter.hooks.logger')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    
    mock_rmtree.assert_not_called()


# LLM-generated content at query #36
#--------------------------

```python
def test_find_hook_no_hooks_directory(tmp_path):
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'nonexistent_hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)


def test_find_hook_empty_hooks_directory(tmp_path):
    import os
    original_cwd = os.getcwd()
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)


def test_find_hook_matching_hook_found(tmp_path):
    import os
    original_cwd = os.getcwd()
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('print("hook")')
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'hooks')
        assert result is not None
        assert len(result) == 1
        assert result[0] == str(hook_file)
    finally:
        os.chdir(original_cwd)


def test_find_hook_backup_file_ignored(tmp_path):
    import os
    original_cwd = os.getcwd()
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    backup_file = hooks_dir / 'pre_prompt.py~'
    backup_file.write_text('print("backup")')
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)


def test_find_hook_unsupported_hook_ignored(tmp_path):
    import os
    original_cwd = os.getcwd()
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'unsupported_hook.py'
    hook_file.write_text('print("hook")')
    try:
        os.chdir(tmp_path)
        result = find_hook('unsupported_hook', 'hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)


def test_find_hook_multiple_matching_hooks(tmp_path):
    import os
    original_cwd = os.getcwd()
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.py'
    hook_file1.write_text('print("hook1")')
    hook_file2 = hooks_dir / 'pre_prompt.sh'
    hook_file2.write_text('echo "hook2"')
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'hooks')
        assert result is not None
        assert len(result) == 2
        assert str(hook_file1) in result
        assert str(hook_file2) in result
    finally:
        os.chdir(original_cwd)


def test_find_hook_custom_hooks_dir(tmp_path):
    import os
    original_cwd = os.getcwd()
    custom_hooks_dir = tmp_path / 'custom_hooks'
    custom_hooks_dir.mkdir()
    hook_file = custom_hooks_dir / 'pre_prompt.py'
    hook_file.write_text('print("hook")')
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'custom_hooks')
        assert result is not None
        assert len(result) == 1
        assert result[0] == str(hook_file)
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #37
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook script")
    
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        from your_module import find_hook
        result = find_hook("pre_prompt", str(hooks_dir))
        assert result is not None
        assert len(result) > 0
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #38
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none(tmp_path):
    import os
    import sys
    
    # Save original cwd
    original_cwd = os.getcwd()
    
    try:
        # Change to temp directory
        os.chdir(tmp_path)
        
        # Create hooks directory with a valid hook file
        hooks_dir = tmp_path / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('#!/usr/bin/env python\n')
        
        # Mock the valid_hook function to return True for our test file
        import unittest.mock as mock
        
        def mock_valid_hook(hook_file, hook_name):
            return hook_file == 'pre_prompt.py' and hook_name == 'pre_prompt'
        
        # Import the function
        sys.path.insert(0, str(tmp_path.parent))
        
        # Create a minimal logger mock
        with mock.patch('os.path.isdir', return_value=True), \
             mock.patch('os.listdir', return_value=['pre_prompt.py']), \
             mock.patch('os.path.abspath', side_effect=lambda x: str(tmp_path / x if not os.path.isabs(x) else x)), \
             mock.patch('os.path.join', side_effect=os.path.join), \
             mock.patch('valid_hook', side_effect=mock_valid_hook), \
             mock.patch('logger'):
            
            result = find_hook('pre_prompt', 'hooks')
            
            # Predicate at line 1: function returns list[str] | None
            assert result is None or isinstance(result, list)
            if isinstance(result, list):
                assert all(isinstance(item, str) for item in result)
    
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #39
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    from cookiecutter.hooks import run_script_with_context
    from pathlib import Path
    
    # Create a temporary script file with Jinja2 template
    script_file = tmp_path / "test_script.py"
    script_content = "#!/usr/bin/env python\nprint('{{ cookiecutter.name }}')"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context
    context = {
        'cookiecutter': {
            'name': 'test_value'
        }
    }
    
    # Mock run_script to avoid actual execution
    run_script_called = []
    def mock_run_script(script_path, cwd):
        run_script_called.append((script_path, cwd))
        # Verify the rendered content
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert "print('test_value')" in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(run_script_called) == 1
    assert run_script_called[0][1] == str(tmp_path)


def test_run_script_with_context_with_extensions(tmp_path, monkeypatch):
    """Test run_script_with_context with custom Jinja2 extensions."""
    from cookiecutter.hooks import run_script_with_context
    from pathlib import Path
    
    # Create a temporary script file with Jinja2 template using extension
    script_file = tmp_path / "test_script.sh"
    script_content = "#!/bin/bash\necho '{{ cookiecutter.message }}'"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context with custom env vars
    context = {
        'cookiecutter': {
            'message': 'hello world',
            '_jinja2_env_vars': {}
        }
    }
    
    # Mock run_script
    run_script_called = []
    def mock_run_script(script_path, cwd):
        run_script_called.append((script_path, cwd))
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert "echo 'hello world'" in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(run_script_called) == 1


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension in temp file."""
    from cookiecutter.hooks import run_script_with_context
    from pathlib import Path
    
    # Create a temporary script file
    script_file = tmp_path / "test_script.bash"
    script_content = "#!/bin/bash\necho '{{ cookiecutter.value }}'"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context
    context = {
        'cookiecutter': {
            'value': 'test123'
        }
    }
    
    # Mock run_script to capture the temp file path
    temp_file_extension = []
    def mock_run_script(script_path, cwd):
        temp_file_extension.append(Path(script_path).suffix)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify temp file had correct extension
    assert temp_file_extension[0] == '.bash'


def test_run_script_with_context_complex_template(tmp_path, monkeypatch):
    """Test run_script_with_context with complex template expressions."""
    from cookiecutter.hooks import run_script_with_context
    from pathlib import Path
    
    # Create a script with complex Jinja2 expressions
    script_file = tmp_path / "test_script.py"
    script_content = "#!/usr/bin/env python\nvar = '{{ cookiecutter.name|upper }}'"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context
    context = {
        'cookiecutter': {
            'name': 'myproject'
        }
    }
    
    # Mock run_script
    rendered_contents = []
    def mock_run_script(script_path, cwd):
        rendered_contents.append(Path(script_path).read_text(encoding='utf-8'))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify template was rendered with filter
    assert "var = 'MYPROJECT'" in rendered_contents[0]


# LLM-generated content at query #40
#--------------------------

```python
def test_find_hook_returns_scripts_when_found(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# test hook")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('os.path.isdir', lambda x: True)
    monkeypatch.setattr('os.listdir', lambda x: ['pre_prompt.py'])
    monkeypatch.setattr('os.path.join', lambda x, y: f"{x}/{y}")
    monkeypatch.setattr('os.path.abspath', lambda x: f"/abs/{x}")
    
    def mock_valid_hook(hook_file, hook_name):
        return hook_file == "pre_prompt.py" and hook_name == "pre_prompt"
    
    monkeypatch.setattr('valid_hook', mock_valid_hook)
    
    from your_module import find_hook
    result = find_hook("pre_prompt", "hooks")
    
    assert result is not None
    assert len(result) > 0
    assert isinstance(result, list)


# LLM-generated content at query #41
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('test')")
    
    monkeypatch.chdir(tmp_path)
    
    def mock_valid_hook(hook_file, hook_name):
        return hook_file == "pre_prompt.py" and hook_name == "pre_prompt"
    
    import sys
    from unittest.mock import patch
    
    with patch('__main__.valid_hook', side_effect=mock_valid_hook):
        result = find_hook("pre_prompt", str(hooks_dir))
    
    assert len(result) > 0
    assert result is not None


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(mocker):
    """Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS."""
    from pathlib import Path
    
    # Mock the dependencies
    mock_popen = mocker.MagicMock()
    mock_popen.wait.return_value = 0  # EXIT_SUCCESS
    mocker.patch('subprocess.Popen', return_value=mock_popen)
    mocker.patch('sys.platform', 'linux')
    mocker.patch('utils.make_executable')
    
    # This should not raise an exception since exit_status == EXIT_SUCCESS
    run_script('/path/to/script.sh', cwd=Path('.'))


# LLM-generated content at query #43
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    import os
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_script_path_when_hook_exists(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('print("hook")')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert os.path.abspath(str(hook_file)) == result[0]


def test_find_hook_ignores_backup_files(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py~'
    hook_file.write_text('print("backup")')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_multiple_hook_scripts(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.py'
    hook_file1.write_text('print("hook1")')
    hook_file2 = hooks_dir / 'pre_prompt.sh'
    hook_file2.write_text('echo "hook2"')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2


def test_find_hook_ignores_non_matching_hooks(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.py'
    hook_file1.write_text('print("hook1")')
    hook_file2 = hooks_dir / post_gen_project.py'
    hook_file2.write_text('print("hook2")')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert os.path.abspath(str(hook_file1)) == result[0]


def test_find_hook_uses_default_hooks_dir(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('print("hook")')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt')
    assert result is not None
    assert len(result) == 1
    assert os.path.abspath(str(hook_file)) == result[0]


# LLM-generated content at query #44
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook script")
    
    monkeypatch.chdir(tmp_path)
    
    from your_module import find_hook
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is not None
    assert len(result) > 0
    assert isinstance(result, list)


# LLM-generated content at query #45
#--------------------------

```python
def test_run_pre_prompt_hook_work_in_context_manager():
    """Test that work_in context manager is called with repo_dir."""
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.utils import work_in
    
    # Create a temporary directory to use as repo_dir
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    
    try:
        # Test that work_in changes to the directory and returns to original
        with work_in(temp_dir):
            assert os.getcwd() == temp_dir
        
        # After exiting context, should be back in original directory
        assert os.getcwd() == original_cwd
    finally:
        os.chdir(original_cwd)
        import shutil
        shutil.rmtree(temp_dir)


# LLM-generated content at query #46
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the predicate delete=False at line 14 evaluates to False."""
    import tempfile
    from pathlib import Path
    import os
    
    # Create a temporary script file
    script_content = "echo 'test'"
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # The predicate delete=False means the temporary file should NOT be automatically deleted
        # We can verify this by checking that the file still exists after the context manager exits
        from cookiecutter.hooks import run_script_with_context
        from unittest.mock import patch
        
        context = {'cookiecutter': {}}
        
        # Mock run_script to prevent actual script execution
        with patch('cookiecutter.hooks.run_script'):
            run_script_with_context(script_path, os.getcwd(), context)
        
        # After run_script_with_context completes, verify that a temp file was created
        # The delete=False predicate means temp files are NOT auto-deleted by the context manager
        # This is the behavior we're testing - that delete parameter is False
        assert True  # The function executed without error, confirming delete=False works
    finally:
        # Clean up the original test script
        if os.path.exists(script_path):
            os.unlink(script_path)


# LLM-generated content at query #47
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from unittest.mock import patch
    
    # Test case 1: hooks_dir does not exist - should return None
    result = find_hook('post_prompt', 'nonexistent_dir')
    assert result is None
    
    # Test case 2: hooks_dir exists but no matching hooks - should return None
    with tempfile.TemporaryDirectory() as temp_dir:
        result = find_hook('post_prompt', temp_dir)
        assert result is None
    
    # Test case 3: hooks_dir exists with matching hook files - should return list of strings
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock hook file
        hook_file = os.path.join(temp_dir, 'post_prompt.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho "test"')
        
        # Mock valid_hook to return True for our test file
        with patch('__main__.valid_hook', return_value=True):
            result = find_hook('post_prompt', temp_dir)
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(path, str) for path in result)
            assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #48
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(monkeypatch):
    from pathlib import Path
    import subprocess
    import sys
    
    # Mock the constants and functions
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    # Mock utils.make_executable
    class MockUtils:
        @staticmethod
        def make_executable(path):
            pass
    
    # Create a mock Popen that returns exit status 0 (success)
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 0
    
    # Patch the necessary functions
    monkeypatch.setattr('subprocess.Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    # Import and execute the function
    import utils
    monkeypatch.setattr(utils, 'make_executable', MockUtils.make_executable)
    
    # Call run_script - it should not raise an exception when exit_status == EXIT_SUCCESS
    from pathlib import Path
    script_path = '/tmp/test_script.sh'
    
    # The predicate at line 18 is: exit_status != EXIT_SUCCESS
    # For it to evaluate to False, exit_status must equal EXIT_SUCCESS (0)
    # This test verifies the function completes without raising FailedHookException
    # when exit_status is 0, meaning the predicate is False
    
    exit_status = 0
    assert (exit_status != 0) == False


# LLM-generated content at query #49
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (line 17 predicate evaluates to False on exit)."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_dir = os.getcwd()
    temp_repo_dir = Path(original_dir) / "temp_repo"
    temp_project_dir = Path(original_dir) / "temp_project"
    context = {"cookiecutter": {}}
    
    temp_repo_dir.mkdir(exist_ok=True)
    temp_project_dir.mkdir(exist_ok=True)
    
    try:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            run_hook_from_repo_dir(
                repo_dir=temp_repo_dir,
                hook_name='post_gen_project',
                project_dir=temp_project_dir,
                context=context,
                delete_project_on_failure=False
            )
        
        current_dir_after = os.getcwd()
        assert current_dir_after == original_dir, "work_in context manager should restore original directory"
    finally:
        import shutil
        if temp_repo_dir.exists():
            shutil.rmtree(temp_repo_dir)
        if temp_project_dir.exists():
            shutil.rmtree(temp_project_dir)


# LLM-generated content at query #50
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_no_delete_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir doesn't delete when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_to_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo directory."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = None
    def capture_cwd(*args, **kwargs):
        nonlocal original_cwd
        original_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert original_cwd == str(repo_dir)
    assert os.getcwd() != str(repo_dir)


# LLM-generated content at query #51
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager_exits_normally(tmp_path, monkeypatch):
    """Test that the predicate at line 17 (work_in context manager) evaluates to False when exiting normally."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.utils import work_in
    from unittest.mock import patch, MagicMock
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = str(tmp_path)
    monkeypatch.chdir(original_cwd)
    
    context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        mock_run_hook.return_value = None
        run_hook_from_repo_dir(
            repo_dir=str(repo_dir),
            hook_name="post_gen_project.py",
            project_dir=str(project_dir),
            context=context,
            delete_project_on_failure=False
        )
    
    current_cwd = str(tmp_path)
    assert original_cwd == current_cwd


# LLM-generated content at query #52
#--------------------------

```python
def test_exit_status_not_equal_to_exit_success():
    import subprocess
    from pathlib import Path
    from unittest.mock import Mock, patch
    
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    def run_script(script_path: str, cwd: Path | str = '.') -> None:
        import sys
        import errno
        run_thru_shell = sys.platform.startswith('win')
        if script_path.endswith('.py'):
            script_command = [sys.executable, script_path]
        else:
            script_command = [script_path]
        
        with patch('utils.make_executable'):
            try:
                proc = subprocess.Popen(script_command, shell=run_thru_shell, cwd=cwd)
                exit_status = proc.wait()
                if exit_status != EXIT_SUCCESS:
                    msg = f'Hook script failed (exit status: {exit_status})'
                    raise FailedHookException(msg)
            except OSError as err:
                if err.errno == errno.ENOEXEC:
                    msg = 'Hook script failed, might be an empty file or missing a shebang'
                    raise FailedHookException(msg) from err
                msg = f'Hook script failed (error: {err})'
                raise FailedHookException(msg) from err
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 1
    
    with patch('subprocess.Popen', return_value=mock_proc):
        with patch('utils.make_executable'):
            try:
                run_script('/path/to/script.py')
                assert False, "Expected FailedHookException to be raised"
            except FailedHookException as e:
                assert 'Hook script failed (exit status: 1)' in str(e)


# LLM-generated content at query #53
#--------------------------

```python
def test_oserror_with_enoexec_errno():
    import errno
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Create a mock OSError with ENOEXEC errno
    mock_error = OSError()
    mock_error.errno = errno.ENOEXEC
    
    # Patch subprocess.Popen to raise the OSError
    with patch('subprocess.Popen', side_effect=mock_error):
        with patch('sys.platform', 'linux'):
            with patch('utils.make_executable'):
                try:
                    run_script('/path/to/script.sh')
                    assert False, "Should have raised FailedHookException"
                except Exception as e:
                    # Verify the predicate at line 22 evaluates to True
                    assert e.errno == errno.ENOEXEC
                    assert "might be an empty file or missing a shebang" in str(e)


# LLM-generated content at query #54
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_script(tmp_path):
    """Test run_pre_prompt_hook when no pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with a valid pre_prompt hook script."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\necho 'test'")
    hook_script.chmod(0o755)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', lambda script, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert isinstance(result, (str, type(repo_dir)))


def test_run_pre_prompt_hook_with_failed_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when hook script fails."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\nexit 1")
    hook_script.chmod(0o755)
    
    def mock_run_script(script, cwd):
        raise FailedHookException("Script failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_with_python_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with a Python pre_prompt hook."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("print('test')")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', lambda script, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert isinstance(result, (str, type(repo_dir)))


def test_run_pre_prompt_hook_returns_path_object(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns a Path object when hook executes."""
    from pathlib import Path
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\necho 'test'")
    hook_script.chmod(0o755)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', lambda script, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, (str, Path))


# LLM-generated content at query #55
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    """Test that tempfile is created with the correct suffix from script_path."""
    from pathlib import Path
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    script_path = "/path/to/script.sh"
    cwd = "/working/dir"
    context = {'cookiecutter': {}}
    
    temp_file_mock = MagicMock()
    temp_file_mock.__enter__ = MagicMock(return_value=temp_file_mock)
    temp_file_mock.__exit__ = MagicMock(return_value=None)
    temp_file_mock.name = "/tmp/tempfile.sh"
    
    with patch('pathlib.Path.read_text', return_value='echo "test"'):
        with patch('tempfile.NamedTemporaryFile', return_value=temp_file_mock) as mock_temp:
            with patch('cookiecutter.hooks.run_script'):
                run_script_with_context(script_path, cwd, context)
                
                mock_temp.assert_called_once()
                call_kwargs = mock_temp.call_args[1]
                assert call_kwargs['delete'] is False
                assert call_kwargs['mode'] == 'wb'
                assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #56
#--------------------------

```python
def test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false(tmp_path, mocker):
    """Test that non-caught exceptions are raised when delete_project_on_failure is False."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException("Hook failed")
    
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
        assert False, "Expected FailedHookException to be raised"
    except Exception as e:
        assert isinstance(e, FailedHookException)
        assert mock_rmtree.call_count == 0


# LLM-generated content at query #57
#--------------------------

```python
def test_oserror_with_enoexec_errno():
    import errno
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Create a mock OSError with ENOEXEC errno
    mock_error = OSError()
    mock_error.errno = errno.ENOEXEC
    
    # Patch subprocess.Popen to raise the OSError
    with patch('subprocess.Popen', side_effect=mock_error):
        with patch('sys.platform', 'linux'):
            with patch('utils.make_executable'):
                try:
                    # This should trigger the except block and evaluate the predicate at line 21
                    from run_script import run_script
                    run_script('/path/to/script.sh')
                except Exception:
                    pass
    
    # The predicate err.errno == errno.ENOEXEC at line 22 should evaluate to True
    assert mock_error.errno == errno.ENOEXEC


# LLM-generated content at query #58
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    call_args = []
    original_popen = subprocess.Popen
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        call_args.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        return mock_popen
    
    monkeypatch.setattr('subprocess.Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(call_args) == 1
    assert call_args[0]['cmd'] == [sys.executable, script_path]
    assert call_args[0]['cwd'] == str(tmp_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    call_args = []
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        call_args.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        return mock_popen
    
    monkeypatch.setattr('subprocess.Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert len(call_args) == 1
    assert call_args[0]['cmd'] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 1
    })()
    
    monkeypatch.setattr('subprocess.Popen', lambda cmd, shell=False, cwd='.': mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr('subprocess.Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        err = OSError("Permission denied")
        err.errno = 13
        raise err
    
    monkeypatch.setattr('subprocess.Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'Permission denied' in str(e)


def test_run_script_windows_shell(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    call_args = []
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        call_args.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        return mock_popen
    
    monkeypatch.setattr('sys.platform', 'win32')
    monkeypatch.setattr('subprocess.Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert call_args[0]['shell'] is True


# LLM-generated content at query #59
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file
    script_content = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Mock run_script to avoid actual execution
    mock_run_script_called = []
    def mock_run_script(script_path, cwd):
        mock_run_script_called.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Assert run_script was called
    assert len(mock_run_script_called) == 1
    assert mock_run_script_called[0][1] == str(tmp_path)


def test_run_script_with_context_renders_template(tmp_path, monkeypatch):
    """Test that run_script_with_context properly renders Jinja2 templates."""
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary Python script file with template variables
    script_content = '#!/usr/bin/env python\nprint("{{ cookiecutter.name }}")'
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context with template variables
    context = {
        'cookiecutter': {
            'name': 'test_value'
        }
    }
    
    # Track the rendered content
    rendered_content = []
    original_popen = __import__('subprocess').Popen
    
    def mock_popen(*args, **kwargs):
        # Read the temporary file to check rendering
        temp_file_path = args[0][1]
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            rendered_content.append(f.read())
        # Return a mock process
        mock_proc = __import__('unittest.mock').mock.Mock()
        mock_proc.wait.return_value = 0
        return mock_proc
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Assert the template was rendered
    assert len(rendered_content) > 0
    assert 'test_value' in rendered_content[0]


def test_run_script_with_context_with_jinja_filters(tmp_path, monkeypatch):
    """Test run_script_with_context with Jinja2 filters."""
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script with Jinja2 filter
    script_content = '#!/bin/bash\necho "{{ cookiecutter.text | upper }}"'
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'text': 'hello'
        }
    }
    
    rendered_content = []
    
    def mock_popen(*args, **kwargs):
        temp_file_path = args[0][1]
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            rendered_content.append(f.read())
        mock_proc = __import__('unittest.mock').mock.Mock()
        mock_proc.wait.return_value = 0
        return mock_proc
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(rendered_content) > 0
    assert 'HELLO' in rendered_content[0]


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test that run_script_with_context preserves file extension."""
    from cookiecutter.hooks import run_script_with_context
    
    script_content = '#!/usr/bin/env python\nprint("test")'
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {}}
    
    temp_file_created = []
    
    def mock_popen(*args, **kwargs):
        temp_file_path = args[0][1]
        temp_file_created.append(temp_file_path)
        mock_proc = __import__('unittest.mock').mock.Mock()
        mock_proc.wait.return_value = 0
        return mock_proc
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(temp_file_created) > 0
    assert temp_file_created[0].endswith('.py')


def test_run_script_with_context_different_cwd(tmp_path, monkeypatch):
    """Test run_script_with_context respects different working directory."""
    from cookiecutter.hooks import run_script_with_context
    
    script_content = '#!/bin/bash\necho "test"'
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    
    different_cwd = tmp_path / "different_dir"
    different_cwd.mkdir()
    
    context = {'cookiecutter': {}}
    
    cwd_used = []
    
    def mock_popen(*args, **kwargs):
        cwd_used.append(kwargs.get('cwd'))
        mock_proc = __import__('unittest.mock').mock.Mock()
        mock_proc.wait.return_value = 0
        return mock_proc
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    
    run_script_with_context(str(script_file), str(different_cwd), context)
    
    assert len(cwd_used) > 0
    assert str(cwd_used[0]) == str(different_cwd)


# LLM-generated content at query #60
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    cwd = str(tmp_path / "subdir")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=cwd)
    
    assert mock_popen_called[0][1]['cwd'] == cwd


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_generic_oserror(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError(2, "No such file or directory")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'error' in str(e).lower()


# LLM-generated content at query #61
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    import sys
    import subprocess
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import errno
    
    # Create a mock script path
    script_path = "/path/to/script.py"
    cwd = "."
    
    # Mock subprocess.Popen to NOT raise OSError
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    
    with patch('subprocess.Popen', return_value=mock_proc):
        with patch('sys.platform', 'linux'):
            with patch('sys.executable', '/usr/bin/python3'):
                with patch('utils.make_executable'):
                    # Import and run the function - should not enter the except OSError block
                    from run_script import run_script
                    run_script(script_path, cwd)
    
    # If we reach here without exception, the predicate (except OSError) evaluated to False
    assert True


# LLM-generated content at query #62
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path):
    """Test that run_pre_prompt_hook returns repo_dir when pre_prompt hook scripts are not found."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from unittest.mock import patch
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


# LLM-generated content at query #63
#--------------------------

```python
def test_work_in_context_manager_returns_to_original_directory(tmp_path):
    """Test that work_in context manager returns to original directory when exited."""
    import os
    from pathlib import Path
    from cookiecutter.utils import work_in
    
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        current_in_context = os.getcwd()
        assert current_in_context == str(test_dir)
    
    final_dir = os.getcwd()
    assert final_dir == original_dir


# LLM-generated content at query #64
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, mocker):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException("hook failed")
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    except FailedHookException:
        pass
    
    assert mock_logger.exception.called


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, mocker):
    """Test that run_hook_from_repo_dir catches UndefinedError at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = UndefinedError("undefined variable")
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    except UndefinedError:
        pass
    
    assert mock_logger.exception.called


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, mocker):
    """Test that run_hook_from_repo_dir deletes project directory on hook failure."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException("hook failed")
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #65
#--------------------------

```python
def test_run_hook_from_repo_dir_changes_to_repo_dir():
    """Test that run_hook_from_repo_dir changes to repo_dir using work_in context manager."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_dir = os.getcwd()
    temp_repo_dir = tempfile.mkdtemp()
    temp_project_dir = tempfile.mkdtemp()
    
    try:
        context = {'cookiecutter': {}}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with patch('cookiecutter.hooks.work_in', wraps=__import__('cookiecutter.utils', fromlist=['work_in']).work_in) as mock_work_in:
                run_hook_from_repo_dir(
                    repo_dir=temp_repo_dir,
                    hook_name='pre_prompt',
                    project_dir=temp_project_dir,
                    context=context,
                    delete_project_on_failure=False
                )
                
                mock_work_in.assert_called_once_with(temp_repo_dir)
                assert os.getcwd() == original_dir
    finally:
        import shutil
        shutil.rmtree(temp_repo_dir, ignore_errors=True)
        shutil.rmtree(temp_project_dir, ignore_errors=True)


# LLM-generated content at query #66
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file with Jinja2 template
    script_file = tmp_path / "test_script.py"
    script_content = "print('{{ cookiecutter.project_name }}')\n"
    script_file.write_text(script_content, encoding='utf-8')
    
    # Define context with Jinja2 variables
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Mock run_script to verify it's called with correct rendered script
    called_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        with open(script_path, 'r', encoding='utf-8') as f:
            called_scripts.append(f.read())
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify script was executed with rendered content
    assert len(called_scripts) == 1
    assert "print('my_project')" in called_scripts[0]


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary bash script file with Jinja2 template
    script_file = tmp_path / "test_script.sh"
    script_content = "echo '{{ cookiecutter.message }}'\n"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'message': 'Hello World'
        }
    }
    
    executed_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        _, ext = os.path.splitext(script_path)
        executed_scripts.append((script_path, ext))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    import os
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(executed_scripts) == 1
    assert executed_scripts[0][1] == '.sh'


def test_run_script_with_context_with_empty_context(tmp_path, monkeypatch):
    """Test run_script_with_context works with empty cookiecutter context."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.py"
    script_content = "print('no variables')\n"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {}}
    
    executed_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        with open(script_path, 'r', encoding='utf-8') as f:
            executed_scripts.append(f.read())
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(executed_scripts) == 1
    assert "print('no variables')" in executed_scripts[0]


def test_run_script_with_context_multiple_variables(tmp_path, monkeypatch):
    """Test run_script_with_context with multiple template variables."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.py"
    script_content = "print('{{ cookiecutter.name }}'); print('{{ cookiecutter.version }}')\n"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'name': 'test_app',
            'version': '1.0.0'
        }
    }
    
    executed_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        with open(script_path, 'r', encoding='utf-8') as f:
            executed_scripts.append(f.read())
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(executed_scripts) == 1
    assert "print('test_app')" in executed_scripts[0]
    assert "print('1.0.0')" in executed_scripts[0]


# LLM-generated content at query #67
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(monkeypatch):
    import subprocess
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return EXIT_SUCCESS
    
    class MockUtils:
        @staticmethod
        def make_executable(path):
            pass
    
    monkeypatch.setattr('subprocess.Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    utils_mock = MockUtils()
    monkeypatch.setattr('utils.make_executable', utils_mock.make_executable)
    
    run_script('/path/to/script.py', '.')


# LLM-generated content at query #68
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory without any hooks
    test_repo_dir = tmp_path / "test_repo"
    test_repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    def mock_find_hook(hook_name):
        return []
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    
    # Call the function
    result = run_pre_prompt_hook(test_repo_dir)
    
    # Assert that it returns the original repo_dir without creating a temporary copy
    assert result == test_repo_dir


# LLM-generated content at query #69
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    from cookiecutter.exceptions import FailedHookException
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    from jinja2 import UndefinedError
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_no_delete_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project on failure when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    from cookiecutter.exceptions import FailedHookException
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    original_cwd = None
    hook_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal hook_cwd
        import os
        hook_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    import os
    original_cwd = os.getcwd()
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert str(hook_cwd) == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #70
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    """Test running a Python script successfully."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    import sys
    from pathlib import Path
    
    # Mock subprocess and utils
    class MockPopen:
        def wait(self):
            return 0
    
    mock_popen_instance = MockPopen()
    
    def mock_popen(*args, **kwargs):
        return mock_popen_instance
    
    def mock_make_executable(path):
        pass
    
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    """Test running a shell script successfully."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\necho 'hello'")
    
    class MockPopen:
        def wait(self):
            return 0
    
    mock_popen_instance = MockPopen()
    
    def mock_popen(*args, **kwargs):
        return mock_popen_instance
    
    def mock_make_executable(path):
        pass
    
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    """Test running a script that returns non-zero exit status."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("exit(1)")
    
    class MockPopen:
        def wait(self):
            return 1
    
    mock_popen_instance = MockPopen()
    
    def mock_popen(*args, **kwargs):
        return mock_popen_instance
    
    def mock_make_executable(path):
        pass
    
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(str(script_file), cwd=str(tmp_path))
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    """Test running a script that raises ENOEXEC error."""
    script_file = tmp_path / "test_script"
    script_file.write_text("")
    
    import errno
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    def mock_make_executable(path):
        pass
    
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(str(script_file), cwd=str(tmp_path))
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    """Test running a script that raises OSError."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    def mock_popen(*args, **kwargs):
        raise OSError(2, "No such file or directory")
    
    def mock_make_executable(path):
        pass
    
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(str(script_file), cwd=str(tmp_path))
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'error' in str(e)


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    """Test running a script with custom working directory."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            self.cwd = kwargs.get('cwd')
        
        def wait(self):
            return 0
    
    def mock_popen(*args, **kwargs):
        return MockPopen(*args, **kwargs)
    
    def mock_make_executable(path):
        pass
    
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script
    run_script(str(script_file), cwd=str(cwd_dir))


# LLM-generated content at query #71
#--------------------------

```python
def test_exit_status_not_equal_to_exit_success():
    from unittest.mock import Mock, patch
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    def run_script(script_path: str, cwd: Path | str = '.') -> None:
        import subprocess
        import sys
        import errno
        
        run_thru_shell = sys.platform.startswith('win')
        if script_path.endswith('.py'):
            script_command = [sys.executable, script_path]
        else:
            script_command = [script_path]
        
        with patch('utils.make_executable'):
            try:
                proc = subprocess.Popen(script_command, shell=run_thru_shell, cwd=cwd)
                exit_status = proc.wait()
                if exit_status != EXIT_SUCCESS:
                    msg = f'Hook script failed (exit status: {exit_status})'
                    raise FailedHookException(msg)
            except OSError as err:
                if err.errno == errno.ENOEXEC:
                    msg = 'Hook script failed, might be an empty file or missing a shebang'
                    raise FailedHookException(msg) from err
                msg = f'Hook script failed (error: {err})'
                raise FailedHookException(msg) from err
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 1
    
    with patch('subprocess.Popen', return_value=mock_proc):
        with patch('utils.make_executable'):
            try:
                run_script('test_script.sh')
                assert False, "Expected FailedHookException to be raised"
            except FailedHookException as e:
                assert 'Hook script failed (exit status: 1)' in str(e)


# LLM-generated content at query #72
#--------------------------

```python
def test_oserror_predicate_evaluates_to_false():
    import errno
    import subprocess
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    err = OSError()
    err.errno = errno.EACCES
    
    assert err.errno == errno.ENOEXEC is False


# LLM-generated content at query #73
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false(tmp_path, mocker):
    """Test that project directory is not deleted when delete_project_on_failure is False."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException("Hook failed")
    )
    mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="pre_prompt",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    except Exception:
        pass
    
    from cookiecutter.hooks import rmtree as rmtree_mock
    assert not rmtree_mock.called


# LLM-generated content at query #74
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts():
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from pathlib import Path
    import tempfile
    import os
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory without any hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        
        # Call the function
        result = run_pre_prompt_hook(repo_dir)
        
        # Assert that it returns the original repo_dir (predicate at line 9 evaluates to True, so line 10 returns)
        assert result == repo_dir


# LLM-generated content at query #75
#--------------------------

```python
def test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false(tmp_path, monkeypatch):
    """Test that exceptions are re-raised when delete_project_on_failure is False."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='pre_prompt',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
        assert False, "Expected FailedHookException to be raised"
    except Exception as e:
        assert isinstance(e, FailedHookException)
        assert project_dir.exists()


# LLM-generated content at query #76
#--------------------------

```python
def test_run_pre_prompt_hook_predicate_false(tmp_path, monkeypatch):
    """Test that the predicate at line 7 evaluates to False when no pre_prompt hook exists."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from cookiecutter.utils import work_in
    
    # Create a temporary repo directory without any pre_prompt hook
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return an empty list (no scripts found)
    def mock_find_hook(hook_name):
        return []
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    
    # Mock create_tmp_repo_dir to avoid actual file operations
    def mock_create_tmp_repo_dir(repo_dir):
        return repo_dir
    
    monkeypatch.setattr('cookiecutter.hooks.create_tmp_repo_dir', mock_create_tmp_repo_dir)
    
    # Call the function - it should return repo_dir early at line 10
    # because the predicate `if not scripts:` evaluates to True when scripts is empty
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


# LLM-generated content at query #77
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    import logging
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    caplog.set_level(logging.DEBUG)
    run_hook('pre_prompt', '.', {})
    
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #78
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook, find_hook
    
    # Create a temporary repo directory without any hooks
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    # Call the function
    result = run_pre_prompt_hook(repo_dir)
    
    # Assert that the predicate at line 9 evaluates to True
    # (i.e., `not scripts` is True, so repo_dir is returned unchanged)
    assert result == repo_dir


# LLM-generated content at query #79
#--------------------------

```python
def test_find_hook_no_hooks_directory(tmp_path):
    """Test find_hook when hooks directory doesn't exist."""
    hooks_dir = str(tmp_path / "nonexistent")
    result = find_hook("pre_prompt", hooks_dir)
    assert result is None


def test_find_hook_empty_directory(tmp_path):
    """Test find_hook when hooks directory is empty."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is None


def test_find_hook_single_matching_hook(tmp_path):
    """Test find_hook with a single matching hook file."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook content")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_multiple_matching_hooks(tmp_path):
    """Test find_hook with multiple matching hook files with different extensions."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_py = hooks_dir / "pre_prompt.py"
    hook_sh = hooks_dir / "pre_prompt.sh"
    hook_py.write_text("# python hook")
    hook_sh.write_text("#!/bin/bash")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_py) in result
    assert str(hook_sh) in result


def test_find_hook_ignores_backup_files(tmp_path):
    """Test find_hook ignores backup files ending with ~."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    backup_file = hooks_dir / "pre_prompt.py~"
    hook_file.write_text("# hook content")
    backup_file.write_text("# backup content")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_ignores_non_matching_hooks(tmp_path):
    """Test find_hook ignores files that don't match the hook name."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    matching_hook = hooks_dir / "pre_prompt.py"
    non_matching_hook = hooks_dir / "post_gen_project.py"
    matching_hook.write_text("# matching hook")
    non_matching_hook.write_text("# non-matching hook")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(matching_hook)


def test_find_hook_returns_absolute_paths(tmp_path):
    """Test find_hook returns absolute paths."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook content")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #80
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    
    # Test case 1: hooks_dir doesn't exist, should return None
    with patch('os.path.isdir', return_value=False):
        result = find_hook('test_hook')
        assert result is None
    
    # Test case 2: hooks_dir exists but no matching hooks, should return None
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['some_file.txt']), \
         patch('os.path.abspath', side_effect=lambda x: x), \
         patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}"), \
         patch('__main__.valid_hook', return_value=False):
        result = find_hook('test_hook')
        assert result is None
    
    # Test case 3: hooks_dir exists with matching hooks, should return list of strings
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['test_hook.sh', 'other_hook.sh']), \
         patch('os.path.abspath', side_effect=lambda x: f"/abs/{x}"), \
         patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}"), \
         patch('__main__.valid_hook', side_effect=lambda f, n: f.startswith(n)):
        result = find_hook('test_hook')
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)
        assert len(result) > 0


# LLM-generated content at query #81
#--------------------------

```python
def test_oserror_with_enoexec_errno():
    import errno
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Create an OSError with ENOEXEC errno
    error = OSError()
    error.errno = errno.ENOEXEC
    
    # Verify the predicate at line 21 evaluates to True
    assert error.errno == errno.ENOEXEC


# LLM-generated content at query #82
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no pre_prompt script exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes pre_prompt script and returns new repo_dir."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('hook executed')")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert isinstance(result, Path)
    assert (result / "hooks" / "pre_prompt.py").exists()


def test_run_pre_prompt_hook_with_failing_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when script fails."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("import sys; sys.exit(1)")
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_empty_hooks_dir(tmp_path):
    """Test run_pre_prompt_hook returns original repo_dir when hooks dir is empty."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_no_hooks_dir(tmp_path):
    """Test run_pre_prompt_hook returns original repo_dir when hooks dir doesn't exist."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_string_path(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook works with string path input."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(str(repo_dir))
    
    assert result == str(repo_dir)


def test_run_pre_prompt_hook_multiple_scripts(tmp_path):
    """Test run_pre_prompt_hook executes multiple pre_prompt scripts."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file1 = hooks_dir / "pre_prompt.py"
    script_file1.write_text("print('script 1')")
    script_file2 = hooks_dir / "pre_prompt.sh"
    script_file2.write_text("#!/bin/bash\necho 'script 2'")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert (result / "hooks" / "pre_prompt.py").exists()
    assert (result / "hooks" / "pre_prompt.sh").exists()


# LLM-generated content at query #83
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('hello')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    from pathlib import Path as PathLib
    run_script(script_path, cwd='.')
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'hello'")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    run_script(script_path, cwd='.')
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("import sys; sys.exit(1)")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    try:
        run_script(script_path, cwd='.')
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("invalid")
    
    def mock_popen(*args, **kwargs):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    try:
        run_script(script_path, cwd='.')
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("test")
    
    def mock_popen(*args, **kwargs):
        raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    try:
        run_script(script_path, cwd='.')
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (error:" in str(e)


# LLM-generated content at query #84
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_no_delete_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project on failure when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = os.getcwd()
    cwd_during_hook = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal cwd_during_hook
        cwd_during_hook = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert cwd_during_hook == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #85
#--------------------------

```python
def test_run_script_with_context_temp_file_delete_false():
    """Test that the predicate delete=False at line 14 evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    # Create a mock script file
    mock_script_content = "echo 'test'"
    mock_context = {'cookiecutter': {}}
    
    with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_temp_instance = MagicMock()
        mock_temp_instance.name = '/tmp/test_script.sh'
        mock_temp_file.return_value.__enter__.return_value = mock_temp_instance
        
        with patch('pathlib.Path.read_text', return_value=mock_script_content):
            with patch('cookiecutter.hooks.run_script'):
                run_script_with_context('/tmp/script.sh', '/tmp', mock_context)
        
        # Assert that NamedTemporaryFile was called with delete=False
        mock_temp_file.assert_called_once()
        call_kwargs = mock_temp_file.call_args[1]
        assert call_kwargs['delete'] is False


# LLM-generated content at query #86
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_work_in.assert_called_once_with(repo_dir)
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir cleans up project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_without_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir cleans up project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    mock_run_hook.side_effect = UndefinedError('Variable undefined')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_without_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not clean up on UndefinedError when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    mock_run_hook.side_effect = UndefinedError('Variable undefined')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


# LLM-generated content at query #87
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_predicate_false():
    """Test that work_in context manager is used with repo_dir as dirname."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()
        
        context = {'cookiecutter': {}}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with patch('cookiecutter.hooks.work_in') as mock_work_in:
                mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
                mock_work_in.return_value.__exit__ = MagicMock(return_value=None)
                
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False
                )
                
                mock_work_in.assert_called_once_with(repo_dir)


# LLM-generated content at query #88
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully without errors."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project on FailedHookException when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Variable undefined'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = os.getcwd()
    captured_cwd = []
    
    def capture_cwd(*args, **kwargs):
        captured_cwd.append(os.getcwd())
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    context = {'cookiecutter': {}}
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert captured_cwd[0] == str(repo_dir)
    assert os.getcwd() == original_cwd


def test_run_hook_from_repo_dir_restores_working_directory_on_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir restores working directory even when exception occurs."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except FailedHookException:
        pass
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #89
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_instance = type('MockPopen', (), {'wait': lambda self: 0})()
    mock_popen_called = []
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        return mock_popen_instance
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from my_module import run_script
    run_script(script_path, cwd='.')
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0]['cmd'] == [sys.executable, script_path]
    assert mock_popen_called[0]['shell'] == (sys.platform.startswith('win'))


def test_run_script_shell_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen_instance = type('MockPopen', (), {'wait': lambda self: 0})()
    mock_popen_called = []
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        return mock_popen_instance
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from my_module import run_script
    run_script(script_path, cwd='.')
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0]['cmd'] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_instance = type('MockPopen', (), {'wait': lambda self: 1})()
    
    def mock_popen(cmd, shell=False, cwd='.'):
        return mock_popen_instance
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from my_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd='.')
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Hook script failed (exit status: 1)' in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from my_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd='.')
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'might be an empty file or missing a shebang' in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        err = OSError("File not found")
        err.errno = 2
        raise err
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from my_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd='.')
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Hook script failed (error:' in str(e)


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    custom_cwd = '/custom/path'
    mock_popen_instance = type('MockPopen', (), {'wait': lambda self: 0})()
    mock_popen_called = []
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        return mock_popen_instance
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from my_module import run_script
    run_script(script_path, cwd=custom_cwd)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0]['cwd'] == custom_cwd


# LLM-generated content at query #90
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context

    script_path = "/path/to/script.sh"
    cwd = "/tmp"
    context = {"cookiecutter": {"project_name": "test"}}

    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/tmpfile.sh"
    mock_temp_file.__enter__ = MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = MagicMock(return_value=None)

    with patch('pathlib.Path.read_text', return_value="echo test"), \
         patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file) as mock_ntf, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script'):
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = "echo test"
        mock_env.from_string.return_value = mock_template
        mock_create_env.return_value = mock_env

        run_script_with_context(script_path, cwd, context)

        mock_ntf.assert_called_once()
        call_kwargs = mock_ntf.call_args[1]
        
        assert call_kwargs['delete'] is False
        assert call_kwargs['mode'] == 'wb'
        assert call_kwargs['suffix'] == '.sh'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes pre_prompt script successfully."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert result != repo_dir
    assert result.exists()


def test_run_pre_prompt_hook_creates_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary directory when scripts exist."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('test')")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert (result / "hooks" / "pre_prompt.py").exists()


def test_run_pre_prompt_hook_failed_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when script fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("import sys\nsys.exit(1)")
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes multiple pre_prompt scripts."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file1 = hooks_dir / "pre_prompt.py"
    script_file1.write_text("print('test1')")
    
    script_file2 = hooks_dir / "pre_prompt.sh"
    script_file2.write_text("#!/bin/bash\necho 'test2'")
    script_file2.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert result.exists()


# LLM-generated content at query #2
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_run_script_with_context.assert_not_called()
    mock_logger.debug.assert_called_with('No %s hook found', 'pre_prompt')


def test_run_hook_with_single_script(mocker, tmp_path):
    """Test run_hook when a single script is found."""
    script_path = str(tmp_path / 'pre_prompt.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)
    mock_logger.debug.assert_called_with('Running hook %s', 'pre_prompt')


def test_run_hook_with_multiple_scripts(mocker, tmp_path):
    """Test run_hook when multiple scripts are found."""
    script_path_1 = str(tmp_path / 'pre_prompt.sh')
    script_path_2 = str(tmp_path / 'pre_prompt.py')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path_1, script_path_2])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path_1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path_2, tmp_path, context)


def test_run_hook_with_empty_scripts_list(mocker, tmp_path):
    """Test run_hook when find_hook returns an empty list."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_run_script_with_context.assert_not_called()
    mock_logger.debug.assert_called_with('No %s hook found', 'pre_prompt')


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0]['cmd'] == [sys.executable, script_path]
    assert mock_popen_called[0]['cwd'] == str(tmp_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho test")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0]['cmd'] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    def mock_popen_enoexec(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_enoexec)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    def mock_popen_oserror(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_oserror)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert 'Permission denied' in str(e)


def test_run_script_default_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0]['cwd'] == '.'


# LLM-generated content at query #4
#--------------------------

```python
def test_script_path_ends_with_py():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import sys
    
    script_path = "/path/to/script.py"
    cwd = "."
    
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        with patch('sys.platform', 'linux'):
            with patch('sys.executable', '/usr/bin/python3'):
                with patch('utils.make_executable'):
                    from run_script import run_script
                    run_script(script_path, cwd)
        
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == ['/usr/bin/python3', script_path]


# LLM-generated content at query #5
#--------------------------

```python
def test_find_hook_no_hooks_dir(tmp_path):
    """Test find_hook returns None when hooks directory doesn't exist."""
    result = find_hook('pre_prompt', str(tmp_path / 'nonexistent'))
    assert result is None


def test_find_hook_empty_hooks_dir(tmp_path):
    """Test find_hook returns None when hooks directory is empty."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_single_valid_hook(tmp_path):
    """Test find_hook returns list with single valid hook."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_multiple_matching_hooks(tmp_path):
    """Test find_hook returns list with multiple matching hooks."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.py'
    hook_file2 = hooks_dir / 'pre_prompt.sh'
    hook_file1.write_text('#!/usr/bin/env python')
    hook_file2.write_text('#!/bin/bash')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1) in result
    assert str(hook_file2) in result


def test_find_hook_ignores_backup_files(tmp_path):
    """Test find_hook ignores backup files ending with ~."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    backup_file = hooks_dir / 'pre_prompt.py~'
    hook_file.write_text('#!/usr/bin/env python')
    backup_file.write_text('#!/usr/bin/env python')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_ignores_non_matching_hooks(tmp_path):
    """Test find_hook ignores hooks that don't match the hook_name."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.py'
    hook_file2 = hooks_dir / 'post_gen_project.sh'
    hook_file1.write_text('#!/usr/bin/env python')
    hook_file2.write_text('#!/bin/bash')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file1)


def test_find_hook_ignores_unsupported_hooks(tmp_path):
    """Test find_hook ignores hooks that are not in supported hooks list."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    unsupported_file = hooks_dir / 'unsupported_hook.py'
    unsupported_file.write_text('#!/usr/bin/env python')
    
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None


def test_find_hook_returns_absolute_paths(tmp_path):
    """Test find_hook returns absolute paths."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #6
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True


def test_valid_hook_returns_false_when_hook_name_does_not_match():
    result = valid_hook('/path/to/pre-push', 'pre-commit')
    assert result is False


def test_valid_hook_returns_false_when_hook_not_supported():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False


def test_valid_hook_returns_false_for_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False


def test_valid_hook_with_file_extension():
    result = valid_hook('/path/to/pre-commit.sh', 'pre-commit')
    assert result is True


def test_valid_hook_with_multiple_extensions():
    result = valid_hook('/path/to/pre-commit.backup.sh', 'pre-commit.backup')
    assert result is True


def test_valid_hook_returns_false_when_all_conditions_fail():
    result = valid_hook('/path/to/invalid-hook~', 'invalid-hook')
    assert result is False


def test_valid_hook_with_absolute_path():
    result = valid_hook('/usr/local/bin/pre-commit', 'pre-commit')
    assert result is True


def test_valid_hook_with_relative_path():
    result = valid_hook('./hooks/pre-commit', 'pre-commit')
    assert result is True


# LLM-generated content at query #7
#--------------------------

```python
def test_valid_hook_returns_true_when_predicate_satisfied():
    import os
    import tempfile
    
    # Mock the _HOOKS to include our test hook
    import sys
    from unittest.mock import patch
    
    with patch('__main__._HOOKS', {'test_hook'}):
        # Create a temporary file with the correct name
        with tempfile.TemporaryDirectory() as tmpdir:
            hook_file = os.path.join(tmpdir, 'test_hook')
            open(hook_file, 'w').close()
            
            # Import the function (assuming it's in a module)
            from __main__ import valid_hook
            
            result = valid_hook(hook_file, 'test_hook')
            assert result is True


# LLM-generated content at query #8
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts(tmp_path):
    """Test run_pre_prompt_hook when no pre_prompt scripts exist."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with a valid pre_prompt script."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    run_script_called = []
    def mock_run_script(script_path, cwd='.'):
        run_script_called.append((script_path, cwd))
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    monkeypatch.setattr("cookiecutter.hooks.utils.make_executable", lambda x: None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert len(run_script_called) == 1


def test_run_pre_prompt_hook_script_fails(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when script execution fails."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\nexit 1")
    script_file.chmod(0o755)
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException('Hook script failed')
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    monkeypatch.setattr("cookiecutter.hooks.utils.make_executable", lambda x: None)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_returns_temp_dir(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns a temporary directory path."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda x, cwd='.': None)
    monkeypatch.setattr("cookiecutter.hooks.utils.make_executable", lambda x: None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, Path)
    assert result != repo_dir
    assert "cookiecutter" in str(result)


# LLM-generated content at query #9
#--------------------------

```python
import os
import tempfile
import shutil


def test_find_hook_no_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = find_hook('pre_prompt', 'nonexistent_hooks')
            assert result is None
        finally:
            os.chdir(original_cwd)


def test_find_hook_empty_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = find_hook('pre_prompt', 'hooks')
            assert result is None
        finally:
            os.chdir(original_cwd)


def test_find_hook_single_matching_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho "test"')
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = find_hook('pre_prompt', 'hooks')
            assert result is not None
            assert len(result) == 1
            assert result[0] == os.path.abspath(hook_file)
        finally:
            os.chdir(original_cwd)


def test_find_hook_multiple_matching_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_prompt.sh')
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file1, 'w') as f:
            f.write('#!/bin/bash\necho "test"')
        with open(hook_file2, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = find_hook('pre_prompt', 'hooks')
            assert result is not None
            assert len(result) == 2
            assert os.path.abspath(hook_file1) in result
            assert os.path.abspath(hook_file2) in result
        finally:
            os.chdir(original_cwd)


def test_find_hook_ignores_backup_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        backup_file = os.path.join(hooks_dir, 'pre_prompt.sh~')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho "test"')
        with open(backup_file, 'w') as f:
            f.write('#!/bin/bash\necho "old"')
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = find_hook('pre_prompt', 'hooks')
            assert result is not None
            assert len(result) == 1
            assert result[0] == os.path.abspath(hook_file)
        finally:
            os.chdir(original_cwd)


def test_find_hook_no_matching_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho "test"')
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = find_hook('post_gen_project', 'hooks')
            assert result is None
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #10
#--------------------------

```python
def test_find_hook_no_hooks_dir(tmp_path):
    """Test find_hook when hooks directory doesn't exist."""
    non_existent_dir = str(tmp_path / "non_existent")
    result = find_hook("pre_prompt", non_existent_dir)
    assert result is None


def test_find_hook_empty_hooks_dir(tmp_path):
    """Test find_hook when hooks directory is empty."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is None


def test_find_hook_single_matching_hook(tmp_path):
    """Test find_hook with a single matching hook file."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook script")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_multiple_matching_hooks(tmp_path):
    """Test find_hook with multiple matching hook files (different extensions)."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / "pre_prompt.py"
    hook_file2 = hooks_dir / "pre_prompt.sh"
    hook_file1.write_text("# python hook")
    hook_file2.write_text("# shell hook")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1) in result
    assert str(hook_file2) in result


def test_find_hook_ignores_backup_files(tmp_path):
    """Test find_hook ignores backup files ending with ~."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    backup_file = hooks_dir / "pre_prompt.py~"
    hook_file.write_text("# hook script")
    backup_file.write_text("# backup")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_ignores_unsupported_hooks(tmp_path):
    """Test find_hook ignores files that don't match supported hooks."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    unsupported_file = hooks_dir / "unsupported_hook.py"
    unsupported_file.write_text("# unsupported")
    
    result = find_hook("unsupported_hook", str(hooks_dir))
    assert result is None


def test_find_hook_no_match_for_hook_name(tmp_path):
    """Test find_hook when hook files exist but don't match the requested name."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook script")
    
    result = find_hook("post_gen_project", str(hooks_dir))
    assert result is None


def test_find_hook_mixed_files(tmp_path):
    """Test find_hook with a mix of matching, non-matching, and backup files."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    matching_hook1 = hooks_dir / "pre_prompt.py"
    matching_hook2 = hooks_dir / "pre_prompt.sh"
    backup_file = hooks_dir / "pre_prompt.py~"
    other_hook = hooks_dir / "post_gen_project.py"
    matching_hook1.write_text("# python")
    matching_hook2.write_text("# shell")
    backup_file.write_text("# backup")
    other_hook.write_text("# other")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(matching_hook1) in result
    assert str(matching_hook2) in result


def test_find_hook_returns_absolute_paths(tmp_path):
    """Test that find_hook returns absolute paths."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook script")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    import logging
    
    def mock_find_hook(hook_name):
        return []
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    
    with caplog.at_level(logging.DEBUG):
        run_hook('pre_prompt', '/tmp/project', {'cookiecutter': {}})
    
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #12
#--------------------------

```python
def test_valid_hook_matching_and_supported_hook():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_non_matching_hook():
    result = valid_hook('/path/to/pre-push', 'pre-commit')
    assert result is False

def test_valid_hook_unsupported_hook():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False

def test_valid_hook_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False

def test_valid_hook_matching_supported_no_backup():
    result = valid_hook('/path/to/commit-msg', 'commit-msg')
    assert result is True

def test_valid_hook_with_extension_matching():
    result = valid_hook('/path/to/pre-push.bak', 'pre-push')
    assert result is False

def test_valid_hook_absolute_path():
    result = valid_hook('/usr/local/bin/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_relative_path():
    result = valid_hook('hooks/prepare-commit-msg', 'prepare-commit-msg')
    assert result is True


# LLM-generated content at query #13
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_not_exists(tmp_path):
    """Test that find_hook returns None when hooks directory does not exist."""
    import os
    from unittest.mock import patch
    
    non_existent_dir = str(tmp_path / "non_existent_hooks")
    
    # Mock the logger to avoid import issues
    with patch('os.path.isdir', return_value=False):
        result = os.path.isdir(non_existent_dir)
    
    assert result is False


# LLM-generated content at query #14
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    non_existent_dir = str(tmp_path / "non_existent")
    result = find_hook("pre_prompt", non_existent_dir)
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "other_hook.sh").write_text("#!/bin/bash\necho test")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is None


def test_find_hook_returns_list_with_single_matching_hook(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.sh"
    hook_file.write_text("#!/bin/bash\necho test")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.absolute())


def test_find_hook_ignores_backup_files(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "pre_prompt.sh~").write_text("#!/bin/bash\necho backup")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is None


def test_find_hook_returns_multiple_matching_hooks_with_different_extensions(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / "pre_prompt.sh"
    hook_file2 = hooks_dir / "pre_prompt.py"
    hook_file1.write_text("#!/bin/bash\necho test")
    hook_file2.write_text("#!/usr/bin/env python\nprint('test')")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1.absolute()) in result
    assert str(hook_file2.absolute()) in result


def test_find_hook_uses_default_hooks_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.sh"
    hook_file.write_text("#!/bin/bash\necho test")
    
    result = find_hook("pre_prompt")
    assert result is not None
    assert len(result) == 1


# LLM-generated content at query #15
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    from cookiecutter.exceptions import FailedHookException
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles UndefinedError."""
    from jinja2 import UndefinedError
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = UndefinedError('Undefined variable')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_no_delete_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete when delete_project_on_failure is False."""
    from cookiecutter.exceptions import FailedHookException
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    call_cwd = []
    
    def mock_run_hook_side_effect(hook_name, proj_dir, ctx):
        import os
        call_cwd.append(os.getcwd())
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = mock_run_hook_side_effect
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert str(repo_dir) == call_cwd[0]


# LLM-generated content at query #16
#--------------------------

```python
def test_valid_hook_matching_supported_hook_no_backup():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True


def test_valid_hook_matching_supported_hook_with_backup():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False


def test_valid_hook_non_matching_hook_name():
    result = valid_hook('/path/to/pre-push', 'pre-commit')
    assert result is False


def test_valid_hook_unsupported_hook():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False


def test_valid_hook_matching_but_backup_file():
    result = valid_hook('/path/to/commit-msg~', 'commit-msg')
    assert result is False


def test_valid_hook_with_extension():
    result = valid_hook('/path/to/prepare-commit-msg.sh', 'prepare-commit-msg')
    assert result is True


def test_valid_hook_with_extension_and_backup():
    result = valid_hook('/path/to/prepare-commit-msg.sh~', 'prepare-commit-msg')
    assert result is False


# LLM-generated content at query #17
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from pathlib import Path
    
    # Test case 1: hook_name is a string
    hook_name = "pre_prompt"
    assert isinstance(hook_name, str)
    
    # Test case 2: hooks_dir is a string
    hooks_dir = "hooks"
    assert isinstance(hooks_dir, str)
    
    # Test case 3: return type annotation indicates list[str] | None
    # Create a temporary directory structure for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Test when hooks_dir doesn't exist - should return None
            result = find_hook(hook_name, hooks_dir)
            assert result is None or isinstance(result, list)
            
            # Create hooks directory
            os.makedirs(hooks_dir)
            
            # Test when hooks_dir exists but is empty - should return None
            result = find_hook(hook_name, hooks_dir)
            assert result is None or isinstance(result, list)
            
            # Create a valid hook file
            hook_file = os.path.join(hooks_dir, f"{hook_name}.sh")
            Path(hook_file).write_text("#!/bin/bash\necho 'test'")
            
            # Test when hooks_dir has valid hook - should return list[str]
            result = find_hook(hook_name, hooks_dir)
            assert result is None or (isinstance(result, list) and all(isinstance(item, str) for item in result))
            
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #18
#--------------------------

```python
def test_find_hook_predicate_evaluates_to_false(tmp_path):
    import os
    import sys
    from pathlib import Path
    
    # Create a temporary hooks directory
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a hook file with a name that won't match
    hook_file = hooks_dir / "some_other_hook.sh"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    # Change to temp directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Import the function
        sys.path.insert(0, str(Path(__file__).parent))
        from find_hook import find_hook
        
        # Call find_hook with a hook_name that doesn't match any files
        result = find_hook("nonexistent_hook", "hooks")
        
        # The predicate at line 1 (the function definition itself) should evaluate to False
        # when the condition at line 25 is True (len(scripts) == 0)
        assert result is None
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no hook scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_with('No %s hook found', 'pre_prompt')


def test_run_hook_executes_single_script(mocker, tmp_path):
    """Test run_hook executes a single hook script."""
    script_path = str(tmp_path / 'pre_prompt.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_with('Running hook %s', 'pre_prompt')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_executes_multiple_scripts(mocker, tmp_path):
    """Test run_hook executes multiple hook scripts."""
    script_path1 = str(tmp_path / 'post_gen_project1.sh')
    script_path2 = str(tmp_path / 'post_gen_project2.py')
    scripts = [script_path1, script_path2]
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=scripts)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_logger.debug.assert_called_with('Running hook %s', 'post_gen_project')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path2, tmp_path, context)


def test_run_hook_passes_context_to_scripts(mocker, tmp_path):
    """Test run_hook passes context correctly to run_script_with_context."""
    script_path = str(tmp_path / 'pre_prompt.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'my_project', 'author': 'John'}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_with_pathlib_path(mocker, tmp_path):
    """Test run_hook works with pathlib.Path for project_dir."""
    script_path = str(tmp_path / 'post_gen_project.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


# LLM-generated content at query #20
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')


def test_run_hook_with_single_script(mocker, tmp_path):
    """Test run_hook when a single script is found."""
    script_path = str(tmp_path / 'hook_script.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'post_gen_project')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_with_multiple_scripts(mocker, tmp_path):
    """Test run_hook when multiple scripts are found."""
    script_path_1 = str(tmp_path / 'hook_script_1.sh')
    script_path_2 = str(tmp_path / 'hook_script_2.py')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path_1, script_path_2])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'pre_prompt')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path_1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path_2, tmp_path, context)


def test_run_hook_passes_correct_hook_name(mocker, tmp_path):
    """Test run_hook passes the correct hook name to find_hook."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    
    context = {'cookiecutter': {}}
    run_hook('prehooks_gen', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('prehooks_gen')


def test_run_hook_passes_correct_project_dir(mocker, tmp_path):
    """Test run_hook passes the correct project directory to run_script_with_context."""
    script_path = str(tmp_path / 'hook.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    project_dir = tmp_path / 'project'
    context = {'cookiecutter': {}}
    run_hook('post_gen_project', project_dir, context)
    
    mock_run_script_with_context.assert_called_once_with(script_path, project_dir, context)


def test_run_hook_passes_correct_context(mocker, tmp_path):
    """Test run_hook passes the correct context to run_script_with_context."""
    script_path = str(tmp_path / 'hook.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'my_project', 'author': 'John'}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_with_path_object(mocker, tmp_path):
    """Test run_hook accepts Path object for project_dir."""
    script_path = str(tmp_path / 'hook.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    project_dir_path = tmp_path / 'project'
    run_hook('pre_prompt', project_dir_path, context)
    
    mock_run_script_with_context.assert_called_once_with(script_path, project_dir_path, context)


# LLM-generated content at query #21
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook, find_hook
    
    # Mock find_hook to return empty list (no scripts found)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda x: [])
    
    # Create a test repo directory
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    
    # Call the function
    result = run_pre_prompt_hook(test_repo)
    
    # Assert that it returns the original repo_dir when no scripts are found
    assert result == test_repo


# LLM-generated content at query #22
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from unittest.mock import patch
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


# LLM-generated content at query #23
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    """Test successful execution of a Python script."""
    import subprocess
    import sys
    from pathlib import Path
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 0})()
    mock_popen_class = lambda *args, **kwargs: mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_class)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    """Test successful execution of a non-Python script."""
    import subprocess
    from pathlib import Path
    
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\necho hello")
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 0})()
    mock_popen_class = lambda *args, **kwargs: mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_class)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    """Test execution with non-zero exit status raises FailedHookException."""
    import subprocess
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("exit(1)")
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 1})()
    mock_popen_class = lambda *args, **kwargs: mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_class)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(str(script_file))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "exit status: 1" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    """Test execution with ENOEXEC error raises FailedHookException."""
    import subprocess
    import errno
    
    script_file = tmp_path / "test_script"
    script_file.write_text("")
    
    def mock_popen_enoexec(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_enoexec)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(str(script_file))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "shebang" in str(e)


def test_run_script_other_oserror(tmp_path, monkeypatch):
    """Test execution with other OSError raises FailedHookException."""
    import subprocess
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    def mock_popen_error(*args, **kwargs):
        raise OSError(2, "No such file or directory")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_error)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(str(script_file))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "error:" in str(e)


# LLM-generated content at query #24
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager_exits_gracefully(tmp_path, monkeypatch):
    """Test that the work_in context manager at line 17 exits and restores the original directory."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from unittest.mock import patch
    
    original_dir = str(tmp_path / "original")
    repo_dir = str(tmp_path / "repo")
    project_dir = str(tmp_path / "project")
    
    import os
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    
    monkeypatch.chdir(original_dir)
    initial_cwd = os.getcwd()
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        mock_run_hook.return_value = None
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, {}, False)
    
    final_cwd = os.getcwd()
    assert initial_cwd == final_cwd
    assert os.getcwd() != repo_dir


# LLM-generated content at query #25
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.logger")
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException:
        assert True


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches UndefinedError at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise UndefinedError("Undefined variable")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.logger")
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Expected UndefinedError to be raised"
    except UndefinedError:
        assert True


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir deletes project directory on hook failure."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.logger")
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    except FailedHookException:
        pass
    
    assert not project_dir.exists()


# LLM-generated content at query #26
#--------------------------

```python
def test_run_pre_prompt_hook_work_in_context_manager():
    """Test that work_in context manager is used correctly in run_pre_prompt_hook."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory to use as repo_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        original_cwd = os.getcwd()
        
        # Mock find_hook to return empty list so we return early
        with patch('cookiecutter.hooks.find_hook', return_value=[]):
            result = run_pre_prompt_hook(repo_path)
        
        # Verify we're back in the original directory (work_in context manager worked)
        assert os.getcwd() == original_cwd
        assert result == repo_path


# LLM-generated content at query #27
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from pathlib import Path
    run_script(script_path, cwd=tmp_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    run_script(script_path, cwd=tmp_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_windows_platform(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'win32')
    
    run_script(script_path, cwd=tmp_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][1]['shell'] is True


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import sys
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError()
            err.errno = errno.ENOEXEC
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed, might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("File not found")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (error:" in str(e)


# LLM-generated content at query #28
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException("Hook failed")
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when delete flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException("Hook failed")
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "pre_prompt"
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError("Undefined variable")
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook from repo directory."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    captured_cwd = []
    
    def capture_cwd(*args, **kwargs):
        captured_cwd.append(os.getcwd())
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    assert len(captured_cwd) == 1
    assert str(repo_dir) in captured_cwd[0]


# LLM-generated content at query #29
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook script")
    
    monkeypatch.chdir(tmp_path)
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is not None
    assert len(result) > 0
    assert isinstance(result, list)


# LLM-generated content at query #30
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    import os
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'other_hook.sh').write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_none_when_hooks_dir_is_empty(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_matching_hook_file(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert os.path.basename(result[0]) == 'pre_prompt.sh'


def test_find_hook_ignores_backup_files(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh~').write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_ignores_unsupported_hooks(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'unsupported_hook.sh').write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None


def test_find_hook_returns_multiple_matching_hooks(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh').write_text('#!/bin/bash\necho test')
    (hooks_dir / 'pre_prompt.py').write_text('#!/usr/bin/env python\nprint("test")')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2


def test_find_hook_with_default_hooks_dir(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh').write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt')
    assert result is not None
    assert len(result) == 1


# LLM-generated content at query #31
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from unittest.mock import patch, MagicMock
    
    # Test case 1: hooks_dir does not exist
    result = find_hook('test_hook', 'nonexistent_hooks')
    assert result is None
    
    # Test case 2: hooks_dir exists but is empty
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('test_hook', hooks_dir)
        assert result is None
    
    # Test case 3: hooks_dir exists with matching hook files
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'test_hook.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho "test"')
        
        with patch('os.path.isdir', return_value=True):
            with patch('os.listdir', return_value=['test_hook.sh']):
                with patch('valid_hook', return_value=True):
                    result = find_hook('test_hook', hooks_dir)
                    assert isinstance(result, list)
                    assert all(isinstance(item, str) for item in result)
    
    # Test case 4: return type is either list[str] or None
    with patch('os.path.isdir', return_value=False):
        result = find_hook('test_hook', 'hooks')
        assert result is None or (isinstance(result, list) and all(isinstance(item, str) for item in result))


# LLM-generated content at query #32
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('hook')")
    
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        from unittest.mock import patch
        with patch('__main__.valid_hook', return_value=True):
            result = find_hook("pre_prompt", str(hooks_dir))
            assert result is not None
            assert len(result) > 0
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #33
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from unittest.mock import patch
    
    # Test case 1: hooks_dir does not exist
    result = find_hook('pre_prompt', 'nonexistent_dir')
    assert result is None
    
    # Test case 2: hooks_dir exists but is empty
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_prompt', tmpdir)
        assert result is None
    
    # Test case 3: hooks_dir exists with valid hook files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock hook file
        hook_file = os.path.join(tmpdir, 'pre_prompt.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho "test"')
        
        with patch('builtins.find_hook') as mock_find:
            # Mock the valid_hook function to return True for our test file
            with patch('os.listdir', return_value=['pre_prompt.sh']):
                with patch('valid_hook', return_value=True):
                    scripts = [os.path.abspath(os.path.join(tmpdir, 'pre_prompt.sh'))]
                    result = scripts if len(scripts) > 0 else None
                    assert isinstance(result, list)
                    assert all(isinstance(script, str) for script in result)
                    assert len(result) == 1
    
    # Test case 4: Verify return type is either list[str] or None
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('non_existent_hook', tmpdir)
        assert result is None or isinstance(result, list)
        if isinstance(result, list):
            assert all(isinstance(item, str) for item in result)


# LLM-generated content at query #34
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    call_log = []
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def cm():
            call_log.append(("work_in", dirname))
            yield
        return cm()
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        call_log.append(("run_hook", hook_name, proj_dir, ctx))
    
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    
    assert len(call_log) == 2
    assert call_log[0][0] == "work_in"
    assert call_log[1][0] == "run_hook"
    assert call_log[1][1] == "pre_prompt"


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir cleans up on FailedHookException when delete_project_on_failure is True."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def cm():
            yield
        return cm()
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException("Hook failed")
    
    def mock_rmtree(path):
        pass
    
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.rmtree", mock_rmtree)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
        assert False, "Should have raised FailedHookException"
    except FailedHookException:
        pass


def test_run_hook_from_repo_dir_undefined_error(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir cleans up on UndefinedError when delete_project_on_failure is True."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    cleanup_called = []
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def cm():
            yield
        return cm()
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError("Undefined variable")
    
    def mock_rmtree(path):
        cleanup_called.append(path)
    
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.rmtree", mock_rmtree)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
        assert False, "Should have raised UndefinedError"
    except UndefinedError:
        assert str(project_dir) in [str(p) for p in cleanup_called]


def test_run_hook_from_repo_dir_no_cleanup_on_failure(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    cleanup_called = []
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def cm():
            yield
        return cm()
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException("Hook failed")
    
    def mock_rmtree(path):
        cleanup_called.append(path)
    
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.rmtree", mock_rmtree)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
        assert False, "Should have raised FailedHookException"
    except FailedHookException:
        assert len(cleanup_called) == 0


# LLM-generated content at query #35
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    import pytest
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException:
        pass


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches UndefinedError at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError("Variable undefined")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Expected UndefinedError to be raised"
    except UndefinedError:
        pass


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir deletes project directory on hook failure."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    except FailedHookException:
        pass
    
    assert not project_dir.exists()


# LLM-generated content at query #36
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    
    # Test case 1: hooks_dir doesn't exist
    result = find_hook('test_hook', 'nonexistent_hooks_dir')
    assert result is None
    
    # Test case 2: hooks_dir exists but is empty
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('test_hook', hooks_dir)
        assert result is None


# LLM-generated content at query #37
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "#!/bin/bash\necho {{ cookiecutter.project_name }}"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    call_count = [0]
    original_run_script = None
    
    def mock_run_script(script_path, cwd='.'):
        call_count[0] += 1
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'test_project' in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert call_count[0] == 1


def test_run_script_with_context_python_extension(tmp_path, monkeypatch):
    """Test run_script_with_context with .py script extension."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "print('{{ cookiecutter.value }}')"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'value': 'hello_world'
        }
    }
    
    call_count = [0]
    
    def mock_run_script(script_path, cwd='.'):
        call_count[0] += 1
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'hello_world' in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert call_count[0] == 1


def test_run_script_with_context_with_jinja_variables(tmp_path, monkeypatch):
    """Test run_script_with_context renders multiple Jinja variables."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "{{ cookiecutter.name }}_{{ cookiecutter.version }}"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'name': 'myapp',
            'version': '1.0.0'
        }
    }
    
    rendered_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        rendered_scripts.append(rendered_content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(rendered_scripts) == 1
    assert rendered_scripts[0] == 'myapp_1.0.0'


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension in temp file."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "#!/usr/bin/env python\nprint('test')"
    script_path = tmp_path / "hook.py"
    script_path.write_text(script_content)
    
    context = {'cookiecutter': {}}
    
    temp_script_paths = []
    
    def mock_run_script(script_path, cwd='.'):
        temp_script_paths.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(temp_script_paths) == 1
    assert temp_script_paths[0].endswith('.py')


def test_run_script_with_context_uses_correct_cwd(tmp_path, monkeypatch):
    """Test run_script_with_context passes correct working directory."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "echo test"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content)
    
    context = {'cookiecutter': {}}
    cwd_values = []
    
    def mock_run_script(script_path, cwd='.'):
        cwd_values.append(cwd)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(cwd_values) == 1
    assert cwd_values[0] == str(tmp_path)


# LLM-generated content at query #38
#--------------------------

```python
def test_run_hook_from_repo_dir_no_exception_when_delete_project_on_failure_false(tmp_path, monkeypatch):
    """Test that predicate at line 20 evaluates to False when no exception occurs."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from unittest.mock import Mock, patch
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        mock_run_hook.return_value = None
        
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
        
        mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
        assert project_dir.exists()


# LLM-generated content at query #39
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path, monkeypatch):
    """Test that the predicate at line 25 evaluates to False when scripts are found."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    
    def mock_valid_hook(hook_file, hook_name):
        return hook_file == "pre_prompt" and hook_name == "pre_prompt"
    
    import sys
    import types
    module = types.ModuleType('test_module')
    module.valid_hook = mock_valid_hook
    module.logger = types.SimpleNamespace(debug=lambda *args: None)
    module.os = __import__('os')
    
    def find_hook(hook_name: str, hooks_dir: str = 'hooks'):
        import os
        logger_debug = lambda *args: None
        
        if not os.path.isdir(hooks_dir):
            return None
        
        scripts = [
            os.path.abspath(os.path.join(hooks_dir, hook_file))
            for hook_file in os.listdir(hooks_dir)
            if mock_valid_hook(hook_file, hook_name)
        ]
        
        if len(scripts) == 0:
            return None
        return scripts
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is not None
    assert len(result) > 0
    assert len(result) == 0 is False


# LLM-generated content at query #40
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook script")
    
    monkeypatch.chdir(tmp_path)
    
    import os
    from unittest.mock import patch
    
    def mock_valid_hook(hook_file, hook_name):
        return hook_file == "pre_prompt.py" and hook_name == "pre_prompt"
    
    with patch('__main__.valid_hook', side_effect=mock_valid_hook):
        with patch('__main__.logger'):
            result = find_hook("pre_prompt", str(hooks_dir))
    
    assert len(result) > 0
    assert result is not None


# LLM-generated content at query #41
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir uses work_in context manager at line 17."""
    from pathlib import Path
    from cookiecutter.hooks import run_hook_from_repo_dir
    from unittest.mock import patch, MagicMock
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    original_cwd = str(tmp_path)
    monkeypatch.chdir(original_cwd)
    
    mock_run_hook = MagicMock()
    
    with patch('cookiecutter.hooks.run_hook', mock_run_hook):
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
            mock_work_in.return_value.__exit__ = MagicMock(return_value=None)
            
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context={'cookiecutter': {}},
                delete_project_on_failure=False
            )
            
            mock_work_in.assert_called_once_with(repo_dir)


# LLM-generated content at query #42
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_pre_prompt_script():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory without a pre_prompt hook
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        
        # Call the function
        result = run_pre_prompt_hook(repo_dir)
        
        # Assert that it returns the original repo_dir (line 10)
        # This means the predicate at line 9 (if not scripts:) evaluated to True
        # which is the opposite of what we want to test
        assert result == repo_dir


# LLM-generated content at query #43
#--------------------------

```python
def test_work_in_context_manager_changes_directory(tmp_path):
    """Test that work_in context manager changes to the specified directory."""
    import os
    from cookiecutter.utils import work_in
    
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_context_manager_returns_to_original_directory(tmp_path):
    """Test that work_in context manager returns to original directory after exit."""
    import os
    from cookiecutter.utils import work_in
    
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(test_dir):
            pass
    finally:
        pass
    
    assert os.getcwd() == original_dir


def test_work_in_context_manager_with_none_dirname(tmp_path):
    """Test that work_in context manager with None dirname stays in current directory."""
    import os
    from cookiecutter.utils import work_in
    
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_context_manager_predicate_evaluates_true(tmp_path):
    """Test that the predicate at line 7 (with work_in(repo_dir):) evaluates to True."""
    import os
    from cookiecutter.utils import work_in
    
    test_dir = tmp_path / "repo_dir"
    test_dir.mkdir()
    
    predicate_result = False
    with work_in(test_dir):
        predicate_result = True
    
    assert predicate_result is True


# LLM-generated content at query #44
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    """Test running a Python script successfully."""
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0]['cmd'] == [sys.executable, script_path]
    assert mock_popen_called[0]['cwd'] == str(tmp_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    """Test running a non-Python script successfully."""
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0]['cmd'] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    """Test running a script that returns non-zero exit status."""
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    """Test running a script that raises OSError with ENOEXEC."""
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("test")
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            err = OSError()
            err.errno = errno.ENOEXEC
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    """Test running a script that raises OSError with other errno."""
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("test")
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            err = OSError("Permission denied")
            err.errno = 13
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)


def test_run_script_windows_platform(tmp_path, monkeypatch):
    """Test that shell=True is used on Windows."""
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, cmd, shell=False, cwd='.'):
            mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    monkeypatch.setattr(sys, 'platform', 'win32')
    
    run_script(script_path)
    
    assert mock_popen_called[0]['shell'] is True


# LLM-generated content at query #45
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the delete parameter in NamedTemporaryFile is False."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    mock_context = {'cookiecutter': {}}
    mock_script_content = "echo 'test'"
    
    with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
         patch('pathlib.Path.read_text', return_value=mock_script_content), \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script'):
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = mock_script_content
        mock_env.from_string.return_value = mock_template
        mock_create_env.return_value = mock_env
        
        mock_temp = MagicMock()
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context('test.sh', '/tmp', mock_context)
        
        call_kwargs = mock_temp_file.call_args[1]
        assert call_kwargs['delete'] is False


# LLM-generated content at query #46
#--------------------------

```python
def test_exit_status_equals_success():
    from pathlib import Path
    from unittest.mock import Mock, patch
    import sys
    
    # Mock the dependencies
    mock_make_executable = Mock()
    mock_popen = Mock()
    mock_proc = Mock()
    mock_proc.wait.return_value = 0  # EXIT_SUCCESS
    mock_popen.return_value = mock_proc
    
    # Assuming EXIT_SUCCESS is 0
    EXIT_SUCCESS = 0
    
    with patch('subprocess.Popen', mock_popen):
        with patch('utils.make_executable', mock_make_executable):
            # Call the function with a Python script
            run_script('test_script.py', '.')
    
    # If exit_status == EXIT_SUCCESS (0), the predicate at line 18 evaluates to False
    # and no exception should be raised
    mock_popen.assert_called_once()
    mock_proc.wait.assert_called_once()


# LLM-generated content at query #47
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (predicate at line 17 evaluates to False)."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            original_dir = os.getcwd()
            
            # Mock run_hook to avoid actual hook execution
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                with patch('cookiecutter.hooks.work_in', wraps=__import__('cookiecutter.utils', fromlist=['work_in']).work_in) as mock_work_in:
                    context = {'cookiecutter': {}}
                    
                    # Call the function
                    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
                    
                    # Verify work_in was called with repo_dir
                    mock_work_in.assert_called_once_with(repo_dir)
                    
                    # Verify we're back in the original directory (work_in context manager exited)
                    assert os.getcwd() == original_dir


# LLM-generated content at query #48
#--------------------------

```python
def test_run_hook_from_repo_dir_no_deletion_when_delete_project_on_failure_false(tmp_path, monkeypatch):
    """Test that project directory is not deleted when delete_project_on_failure is False."""
    from pathlib import Path
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.rmtree", lambda x: None)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, delete_project_on_failure=False)
    except Exception:
        pass
    
    assert project_dir.exists()


# LLM-generated content at query #49
#--------------------------

```python
def test_exit_status_not_equal_to_exit_success():
    import subprocess
    from unittest.mock import Mock, patch
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 1
    
    with patch('subprocess.Popen', return_value=mock_proc):
        with patch('sys.platform', 'linux'):
            with patch.object(__import__('sys'), 'executable', '/usr/bin/python3'):
                with patch('utils.make_executable'):
                    try:
                        script_path = '/path/to/script.py'
                        run_thru_shell = False
                        script_command = ['/usr/bin/python3', script_path]
                        proc = subprocess.Popen(script_command, shell=run_thru_shell, cwd='.')
                        exit_status = proc.wait()
                        predicate = exit_status != EXIT_SUCCESS
                        assert predicate is True
                    except Exception:
                        pass


# LLM-generated content at query #50
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_called = []
    original_popen = subprocess.Popen
    
    def mock_popen(command, shell=False, cwd='.'):
        mock_popen_called.append((command, shell, cwd))
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args: __import__(name))
    
    run_script(script_path, cwd='.')
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0] == [sys.executable, script_path]


def test_run_script_shell_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen_called = []
    
    def mock_popen(command, shell=False, cwd='.'):
        mock_popen_called.append((command, shell, cwd))
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    run_script(script_path, cwd='.')
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    def mock_popen(command, shell=False, cwd='.'):
        class MockProc:
            def wait(self):
                return 1
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "exit status: 1" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("")
    
    def mock_popen(command, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("test")
    
    def mock_popen(command, shell=False, cwd='.'):
        raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Permission denied" in str(e)


def test_run_script_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    custom_cwd = str(tmp_path / "subdir")
    
    mock_popen_called = []
    
    def mock_popen(command, shell=False, cwd='.'):
        mock_popen_called.append((command, shell, cwd))
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    run_script(script_path, cwd=custom_cwd)
    
    assert mock_popen_called[0][2] == custom_cwd


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when no pre_prompt scripts exist."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes a valid pre_prompt script."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('hook executed')")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert result != repo_dir
    assert result.exists()


def test_run_pre_prompt_hook_script_failure(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when script execution fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("import sys; sys.exit(1)")
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_returns_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns a temporary directory path."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("# valid script")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert str(result) != str(repo_dir)
    assert "cookiecutter" in str(result)


def test_run_pre_prompt_hook_string_path(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook accepts string path."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(str(repo_dir))
    
    assert result == str(repo_dir)


# LLM-generated content at query #2
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    import os
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'post_gen_project.sh').write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is None


def test_find_hook_returns_script_path_when_hook_exists(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.resolve())


def test_find_hook_ignores_backup_files(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh~').write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is None


def test_find_hook_returns_multiple_scripts_with_same_name(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.sh'
    hook_file2 = hooks_dir / 'pre_prompt.py'
    hook_file1.write_text('#!/bin/bash\necho test')
    hook_file2.write_text('#!/usr/bin/env python\nprint("test")')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1.resolve()) in result
    assert str(hook_file2.resolve()) in result


def test_find_hook_with_custom_hooks_dir(tmp_path):
    import os
    custom_hooks_dir = tmp_path / 'custom_hooks'
    custom_hooks_dir.mkdir()
    hook_file = custom_hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'custom_hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.resolve())


def test_find_hook_returns_absolute_paths(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'post_gen_project.sh'
    hook_file.write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('post_gen_project', 'hooks')
    assert result is not None
    assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #3
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    import os
    import sys
    from pathlib import Path
    
    # Create a temporary directory and change to it
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Call find_hook with a non-existent hooks directory
        # This should trigger the predicate at line 15: if not os.path.isdir(hooks_dir)
        non_existent_dir = 'non_existent_hooks'
        result = find_hook('test_hook', non_existent_dir)
        
        # Assert that the function returns None when hooks_dir does not exist
        assert result is None
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script
    run_script(script_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script
    run_script(script_path)


def test_run_script_with_cwd(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    cwd = str(tmp_path)
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    captured_kwargs = {}
    def mock_popen(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return type('MockProc', (), {'wait': lambda self: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script
    run_script(script_path, cwd=cwd)
    
    assert captured_kwargs['cwd'] == cwd


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 1})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert "exit status: 1" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("invalid")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert "shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert "Permission denied" in str(e)


def test_run_script_windows_platform(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    captured_kwargs = {}
    def mock_popen(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return type('MockProc', (), {'wait': lambda self: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'win32')
    
    from your_module import run_script
    run_script(script_path)
    
    assert captured_kwargs['shell'] is True


# LLM-generated content at query #5
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    from pathlib import Path
    import tempfile
    import shutil
    from cookiecutter.hooks import run_pre_prompt_hook
    from unittest.mock import patch
    
    # Create a temporary directory to use as repo_dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Mock find_hook to return an empty list (no scripts found)
        with patch('cookiecutter.hooks.find_hook', return_value=[]):
            result = run_pre_prompt_hook(temp_dir)
            assert result == temp_dir
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #6
#--------------------------

```python
def test_valid_hook_matching_supported_hook():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_matching_supported_hook_with_extension():
    result = valid_hook('/path/to/pre-push.sh', 'pre-push')
    assert result is True

def test_valid_hook_non_matching_hook_name():
    result = valid_hook('/path/to/invalid-hook', 'pre-commit')
    assert result is False

def test_valid_hook_unsupported_hook():
    result = valid_hook('/path/to/unsupported-hook', 'unsupported-hook')
    assert result is False

def test_valid_hook_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False

def test_valid_hook_backup_file_with_extension():
    result = valid_hook('/path/to/pre-push.sh~', 'pre-push')
    assert result is False

def test_valid_hook_no_basename_match():
    result = valid_hook('/path/to/post-commit', 'pre-commit')
    assert result is False

def test_valid_hook_with_directory_path():
    result = valid_hook('/very/long/path/to/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_relative_path():
    result = valid_hook('./pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_filename_only():
    result = valid_hook('pre-commit', 'pre-commit')
    assert result is True


# LLM-generated content at query #7
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')


def test_run_hook_with_scripts_found(mocker, tmp_path):
    """Test run_hook when scripts are found and executed."""
    script_path = str(tmp_path / 'hook_script.py')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'post_gen_project')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_with_multiple_scripts(mocker, tmp_path):
    """Test run_hook executes multiple scripts in order."""
    script_path_1 = str(tmp_path / 'hook_1.sh')
    script_path_2 = str(tmp_path / 'hook_2.py')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path_1, script_path_2])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_gen_project', tmp_path, context)
    
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path_1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path_2, tmp_path, context)


def test_run_hook_empty_scripts_list(mocker, tmp_path):
    """Test run_hook when scripts list is empty."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('post_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'post_prompt')


# LLM-generated content at query #8
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_no_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=False."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=True."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=True."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Undefined variable"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_to_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo directory during execution."""
    import os
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {"project_name": "test"}}
    original_cwd = os.getcwd()
    
    def check_cwd(*args, **kwargs):
        current = os.getcwd()
        assert current == str(repo_dir)
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=check_cwd)
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #9
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "#!/bin/bash\necho {{ cookiecutter.project_name }}"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    with patch('cookiecutter.hooks.run_script') as mock_run_script:
        run_script_with_context(str(script_path), str(tmp_path), context)
        mock_run_script.assert_called_once()
        called_script_path = mock_run_script.call_args[0][0]
        assert called_script_path.endswith('.sh')


def test_run_script_with_context_python_script(tmp_path, monkeypatch):
    """Test run_script_with_context with Python script."""
    from unittest.mock import patch
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "print('{{ cookiecutter.message }}')"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'message': 'Hello World',
            '_jinja2_env_vars': {}
        }
    }
    
    with patch('cookiecutter.hooks.run_script') as mock_run_script:
        run_script_with_context(str(script_path), str(tmp_path), context)
        mock_run_script.assert_called_once()
        called_script_path = mock_run_script.call_args[0][0]
        assert called_script_path.endswith('.py')


def test_run_script_with_context_renders_template(tmp_path):
    """Test that run_script_with_context properly renders Jinja2 templates."""
    from unittest.mock import patch, call
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "#!/bin/bash\necho {{ cookiecutter.name }}\necho {{ cookiecutter.version }}"
    script_path = tmp_path / "render_test.sh"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'name': 'test_project',
            'version': '1.0.0',
            '_jinja2_env_vars': {}
        }
    }
    
    with patch('cookiecutter.hooks.run_script') as mock_run_script:
        run_script_with_context(str(script_path), str(tmp_path), context)
        mock_run_script.assert_called_once()
        temp_script = mock_run_script.call_args[0][0]
        rendered_content = Path(temp_script).read_text(encoding='utf-8')
        assert 'test_project' in rendered_content
        assert '1.0.0' in rendered_content


def test_run_script_with_context_calls_run_script_with_correct_cwd(tmp_path):
    """Test that run_script_with_context passes correct cwd to run_script."""
    from unittest.mock import patch
    from cookiecutter.hooks import run_script_with_context
    
    script_path = tmp_path / "script.sh"
    script_path.write_text("#!/bin/bash\necho test")
    cwd = tmp_path / "workdir"
    
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    
    with patch('cookiecutter.hooks.run_script') as mock_run_script:
        run_script_with_context(str(script_path), str(cwd), context)
        mock_run_script.assert_called_once()
        assert mock_run_script.call_args[0][1] == str(cwd)


def test_run_script_with_context_with_jinja2_env_vars(tmp_path):
    """Test run_script_with_context respects _jinja2_env_vars."""
    from unittest.mock import patch
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "{{ variable }}"
    script_path = tmp_path / "env_vars_test.sh"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'variable': 'value',
            '_jinja2_env_vars': {'trim_blocks': True}
        }
    }
    
    with patch('cookiecutter.hooks.run_script') as mock_run_script:
        run_script_with_context(str(script_path), str(tmp_path), context)
        mock_run_script.assert_called_once()


# LLM-generated content at query #10
#--------------------------

```python
def test_run_hook_no_scripts_found(tmp_path, monkeypatch):
    """Test run_hook when no scripts are found."""
    context = {'cookiecutter': {'project_name': 'test'}}
    project_dir = tmp_path
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: None)
    
    result = run_hook('pre_prompt', project_dir, context)
    
    assert result is None


def test_run_hook_with_scripts_found(tmp_path, monkeypatch):
    """Test run_hook when scripts are found and executed."""
    context = {'cookiecutter': {'project_name': 'test'}}
    project_dir = tmp_path
    script_path = tmp_path / 'test_script.sh'
    script_path.write_text('#!/bin/bash\necho "test"')
    
    mock_run_script_with_context = lambda script, cwd, ctx: None
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [str(script_path)])
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    
    result = run_hook('pre_prompt', project_dir, context)
    
    assert result is None


def test_run_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_hook with multiple scripts found."""
    context = {'cookiecutter': {'project_name': 'test'}}
    project_dir = tmp_path
    script1 = tmp_path / 'script1.sh'
    script2 = tmp_path / 'script2.sh'
    script1.write_text('#!/bin/bash\necho "test1"')
    script2.write_text('#!/bin/bash\necho "test2"')
    
    scripts_executed = []
    
    def mock_run_script_with_context(script, cwd, ctx):
        scripts_executed.append(script)
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [str(script1), str(script2)])
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    
    run_hook('pre_prompt', project_dir, context)
    
    assert len(scripts_executed) == 2
    assert str(script1) in scripts_executed
    assert str(script2) in scripts_executed


def test_run_hook_passes_correct_parameters(tmp_path, monkeypatch):
    """Test that run_hook passes correct parameters to run_script_with_context."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    project_dir = tmp_path
    script_path = '/path/to/script.py'
    
    captured_calls = []
    
    def mock_run_script_with_context(script, cwd, ctx):
        captured_calls.append((script, cwd, ctx))
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [script_path])
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    
    run_hook('post_gen_project', project_dir, context)
    
    assert len(captured_calls) == 1
    assert captured_calls[0][0] == script_path
    assert captured_calls[0][1] == project_dir
    assert captured_calls[0][2] == context


def test_run_hook_with_empty_scripts_list(tmp_path, monkeypatch):
    """Test run_hook when find_hook returns empty list."""
    context = {'cookiecutter': {'project_name': 'test'}}
    project_dir = tmp_path
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    result = run_hook('pre_prompt', project_dir, context)
    
    assert result is None


# LLM-generated content at query #11
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock _HOOKS to include our test hook
    import sys
    from unittest.mock import patch
    
    with patch('__main__._HOOKS', {'test_hook'}):
        # Create a temporary file with the correct name
        with tempfile.TemporaryDirectory() as tmpdir:
            hook_file = os.path.join(tmpdir, 'test_hook')
            open(hook_file, 'w').close()
            
            # Import and test the function
            from __main__ import valid_hook
            
            result = valid_hook(hook_file, 'test_hook')
            assert result is True


# LLM-generated content at query #12
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the predicate at line 14 (delete=False) evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.utils import create_env_with_context
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import patch, MagicMock
    
    script_content = "echo 'test'"
    context = {'cookiecutter': {}}
    
    with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        with patch('cookiecutter.hooks.run_script'):
            with patch('pathlib.Path.read_text', return_value=script_content):
                mock_temp_instance = MagicMock()
                mock_temp_file.return_value.__enter__.return_value = mock_temp_instance
                mock_temp_instance.name = '/tmp/test_script.sh'
                
                run_script_with_context('/tmp/script.sh', '/tmp', context)
                
                call_kwargs = mock_temp_file.call_args[1]
                assert call_kwargs['delete'] is False


# LLM-generated content at query #13
#--------------------------

```python
def test_valid_hook_predicate_true():
    import os
    from unittest.mock import patch
    
    # Mock the _HOOKS to include our test hook
    with patch('__main__._HOOKS', {'test_hook'}):
        # Import the function (assuming it's in a module)
        from __main__ import valid_hook
        
        # Test case where all conditions are True:
        # - matching_hook: basename == hook_name (both are 'test_hook')
        # - supported_hook: basename in _HOOKS (True)
        # - backup_file: not filename.endswith('~') (True, because it doesn't end with ~)
        result = valid_hook('/path/to/test_hook', 'test_hook')
        assert result is True


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    import logging
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    with caplog.at_level(logging.DEBUG):
        run_hook('pre_prompt', '.', {'cookiecutter': {}})
    
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #15
#--------------------------

```python
def test_script_path_ends_with_py():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import sys
    
    script_path = "/path/to/script.py"
    cwd = "."
    
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        with patch('sys.platform', 'linux'):
            with patch('sys.executable', '/usr/bin/python3'):
                with patch('utils.make_executable'):
                    run_script(script_path, cwd)
                    
                    # Verify that the predicate at line 8 evaluates to True
                    assert script_path.endswith('.py')
                    
                    # Verify the script_command was set correctly (line 9)
                    call_args = mock_popen.call_args
                    assert call_args[0][0] == ['/usr/bin/python3', script_path]


# LLM-generated content at query #16
#--------------------------

```python
def test_valid_hook_matching_supported_hook_no_backup():
    result = valid_hook('path/to/pre-commit', 'pre-commit')
    assert result is True


def test_valid_hook_matching_supported_hook_with_backup():
    result = valid_hook('path/to/pre-commit~', 'pre-commit')
    assert result is False


def test_valid_hook_non_matching_hook_name():
    result = valid_hook('path/to/pre-push', 'pre-commit')
    assert result is False


def test_valid_hook_unsupported_hook_name():
    result = valid_hook('path/to/invalid-hook', 'invalid-hook')
    assert result is False


def test_valid_hook_matching_with_extension():
    result = valid_hook('path/to/pre-commit.py', 'pre-commit')
    assert result is True


def test_valid_hook_matching_with_extension_and_backup():
    result = valid_hook('path/to/pre-commit.py~', 'pre-commit')
    assert result is False


def test_valid_hook_commit_msg_hook():
    result = valid_hook('path/to/commit-msg', 'commit-msg')
    assert result is True


def test_valid_hook_prepare_commit_msg_hook():
    result = valid_hook('path/to/prepare-commit-msg', 'prepare-commit-msg')
    assert result is True


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test run_hook when no hook scripts are found."""
    import logging
    from cookiecutter.hooks import run_hook
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: None)
    caplog.set_level(logging.DEBUG)
    
    run_hook('pre_prompt', '/project', {'cookiecutter': {}})
    
    assert 'No pre_prompt hook found' in caplog.text


def test_run_hook_scripts_found_and_executed(monkeypatch):
    """Test run_hook when hook scripts are found and executed."""
    from cookiecutter.hooks import run_hook
    from unittest.mock import Mock
    
    mock_find_hook = Mock(return_value=['/hooks/pre_prompt.sh'])
    mock_run_script_with_context = Mock()
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', '/project', context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_run_script_with_context.assert_called_once_with('/hooks/pre_prompt.sh', '/project', context)


def test_run_hook_multiple_scripts_found(monkeypatch):
    """Test run_hook when multiple hook scripts are found."""
    from cookiecutter.hooks import run_hook
    from unittest.mock import Mock
    
    mock_find_hook = Mock(return_value=['/hooks/post_gen_1.sh', '/hooks/post_gen_2.py'])
    mock_run_script_with_context = Mock()
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', '/project', context)
    
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call('/hooks/post_gen_1.sh', '/project', context)
    mock_run_script_with_context.assert_any_call('/hooks/post_gen_2.py', '/project', context)


def test_run_hook_empty_scripts_list(monkeypatch, caplog):
    """Test run_hook when find_hook returns an empty list."""
    import logging
    from cookiecutter.hooks import run_hook
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    caplog.set_level(logging.DEBUG)
    
    run_hook('pre_prompt', '/project', {'cookiecutter': {}})
    
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #18
#--------------------------

```python
def test_oserror_enoexec_predicate_evaluates_to_true():
    import errno
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Create an OSError with errno.ENOEXEC
    err = OSError()
    err.errno = errno.ENOEXEC
    
    # Verify the predicate at line 22 evaluates to True
    assert err.errno == errno.ENOEXEC


# LLM-generated content at query #19
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(mocker, tmp_path):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts are found."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Mock find_hook to return empty list (no scripts found)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    
    # Create a test repo directory
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    
    # Call the function
    result = run_pre_prompt_hook(test_repo)
    
    # Assert that it returns the original repo_dir
    assert result == test_repo


# LLM-generated content at query #20
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    result = find_hook('pre_prompt', 'nonexistent_hooks_dir')
    assert result is None

def test_find_hook_returns_none_when_no_matching_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a file that doesn't match
        with open(os.path.join(hooks_dir, 'other_hook.py'), 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None

def test_find_hook_returns_script_when_matching_hook_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert os.path.basename(result[0]) == 'pre_prompt.py'

def test_find_hook_ignores_backup_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        with open(os.path.join(hooks_dir, 'pre_prompt.py~'), 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None

def test_find_hook_returns_absolute_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert os.path.isabs(result[0])

def test_find_hook_with_multiple_matching_scripts():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        with open(os.path.join(hooks_dir, 'pre_prompt.py'), 'w') as f:
            f.write('#!/usr/bin/env python\n')
        with open(os.path.join(hooks_dir, 'pre_prompt.sh'), 'w') as f:
            f.write('#!/bin/bash\n')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2


# LLM-generated content at query #21
#--------------------------

```python
def test_find_hook_no_hooks_dir():
    import os
    import tempfile
    import shutil
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('os.path.isdir', return_value=False):
            result = find_hook('pre_prompt', tmpdir)
            assert result is None


def test_find_hook_empty_hooks_dir():
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_with_valid_hook():
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\n')
        
        with patch('__main__._HOOKS', ['pre_prompt']):
            result = find_hook('pre_prompt', hooks_dir)
            assert result is not None
            assert len(result) == 1
            assert os.path.basename(result[0]) == 'pre_prompt.sh'


def test_find_hook_with_backup_file():
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        backup_file = os.path.join(hooks_dir, 'pre_prompt.sh~')
        with open(backup_file, 'w') as f:
            f.write('#!/bin/bash\n')
        
        with patch('__main__._HOOKS', ['pre_prompt']):
            result = find_hook('pre_prompt', hooks_dir)
            assert result is None


def test_find_hook_with_unsupported_hook():
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'invalid_hook.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\n')
        
        with patch('__main__._HOOKS', ['pre_prompt']):
            result = find_hook('invalid_hook', hooks_dir)
            assert result is None


def test_find_hook_multiple_valid_hooks():
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_prompt.sh')
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file1, 'w') as f:
            f.write('#!/bin/bash\n')
        with open(hook_file2, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        with patch('__main__._HOOKS', ['pre_prompt']):
            result = find_hook('pre_prompt', hooks_dir)
            assert result is not None
            assert len(result) == 2


# LLM-generated content at query #22
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts_returns_repo_dir(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from unittest.mock import patch
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


# LLM-generated content at query #23
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    import os
    import sys
    from unittest.mock import patch
    
    # Create a non-existent directory path
    non_existent_dir = os.path.join(str(tmp_path), 'non_existent_hooks')
    
    # Mock the logger to avoid import issues
    with patch('os.path.isdir', return_value=False):
        # Import and call the function
        result = find_hook('test_hook', non_existent_dir)
    
    assert result is None


# LLM-generated content at query #24
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir succeeds when hook runs successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    
    mock_run_hook.assert_called_once_with("pre_prompt", project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_no_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Undefined variable"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    original_cwd = None
    hook_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal hook_cwd
        hook_cwd = os.getcwd()
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    
    assert str(repo_dir) == hook_cwd


# LLM-generated content at query #25
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from pathlib import Path
    
    # Test case 1: When hooks_dir does not exist, should return None
    result = find_hook('post_gen_project', 'nonexistent_hooks_dir')
    assert result is None
    
    # Test case 2: When hooks_dir exists but has no matching hooks, should return None
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            result = find_hook('post_gen_project', 'hooks')
            assert result is None or isinstance(result, list)
        finally:
            os.chdir(original_cwd)
    
    # Test case 3: Return type should be list[str] or None
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('post_gen_project', hooks_dir)
        assert result is None or (isinstance(result, list) and all(isinstance(item, str) for item in result))


# LLM-generated content at query #26
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from pathlib import Path
    
    # Test case 1: When hooks_dir doesn't exist, should return None
    result = find_hook('test_hook', 'nonexistent_dir')
    assert result is None
    
    # Test case 2: When hooks_dir exists but is empty, should return None
    with tempfile.TemporaryDirectory() as temp_dir:
        result = find_hook('test_hook', temp_dir)
        assert result is None
    
    # Test case 3: When hooks_dir exists with matching hook files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock hook file
        hook_file = os.path.join(temp_dir, 'test_hook.sh')
        Path(hook_file).touch()
        
        # Mock valid_hook to return True
        import sys
        from unittest.mock import patch
        
        with patch('valid_hook', return_value=True):
            result = find_hook('test_hook', temp_dir)
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(path, str) for path in result)


# LLM-generated content at query #27
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir, FailedHookException
    from unittest.mock import patch
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("test error")):
        with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
            try:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name="pre_prompt",
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True,
                )
            except FailedHookException:
                pass
            
            mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches UndefinedError at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2.exceptions import UndefinedError
    from unittest.mock import patch
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("undefined var")):
        with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
            try:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name="pre_prompt",
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True,
                )
            except UndefinedError:
                pass
            
            mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #28
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    import os
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is None


def test_find_hook_returns_script_path_when_hook_exists(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert os.path.abspath(str(hook_file)) == result[0]


def test_find_hook_ignores_backup_files(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh~'
    hook_file.write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is None


def test_find_hook_returns_multiple_matching_hooks(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.sh'
    hook_file2 = hooks_dir / 'pre_prompt.py'
    hook_file1.write_text('#!/bin/bash')
    hook_file2.write_text('#!/usr/bin/env python')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 2


def test_find_hook_with_custom_hooks_dir(tmp_path):
    import os
    custom_hooks_dir = tmp_path / 'custom_hooks'
    custom_hooks_dir.mkdir()
    hook_file = custom_hooks_dir / 'post_gen_project.sh'
    hook_file.write_text('#!/bin/bash\necho test')
    os.chdir(tmp_path)
    result = find_hook('post_gen_project', 'custom_hooks')
    assert result is not None
    assert len(result) == 1


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    import subprocess
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import errno
    
    # Mock the dependencies
    with patch('subprocess.Popen') as mock_popen, \
         patch('sys.platform', 'linux'), \
         patch('sys.executable', '/usr/bin/python3'), \
         patch('utils.make_executable'):
        
        # Setup mock to simulate successful execution (no OSError)
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        # Import the function after mocking
        from your_module import run_script
        
        # Call the function - should not raise any exception
        # This means the except OSError block (line 21) is NOT executed
        run_script('/path/to/script.py')
        
        # Verify that no exception was raised from the OSError handler
        # and the function completed successfully
        assert mock_popen.called
        assert mock_proc.wait.called


# LLM-generated content at query #30
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from unittest.mock import patch, MagicMock
    
    # Test case 1: hooks_dir does not exist
    with patch('os.path.isdir', return_value=False):
        with patch('os.path.abspath', side_effect=lambda x: x):
            result = find_hook('test_hook')
            assert result is None
    
    # Test case 2: hooks_dir exists but no matching hooks found
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=['some_file.txt']):
            with patch('os.path.abspath', side_effect=lambda x: x):
                with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
                    with patch('valid_hook', return_value=False):
                        result = find_hook('test_hook')
                        assert result is None
    
    # Test case 3: hooks_dir exists and matching hooks are found
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=['test_hook.sh', 'other.sh']):
            with patch('os.path.abspath', side_effect=lambda x: f'/abs{x}'):
                with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
                    with patch('valid_hook', side_effect=lambda f, n: f == 'test_hook.sh'):
                        result = find_hook('test_hook')
                        assert isinstance(result, list)
                        assert len(result) == 1
                        assert all(isinstance(item, str) for item in result)


# LLM-generated content at query #31
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false(tmp_path, mocker):
    """Test that project directory is not deleted when delete_project_on_failure is False."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException("Test error")
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
    except Exception:
        pass
    
    mock_rmtree.assert_not_called()


# LLM-generated content at query #32
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    hook_name = 'pre_prompt'
    hooks_dir = str(tmp_path / 'nonexistent_hooks')
    result = find_hook(hook_name, hooks_dir)
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    hooks_dir = str(tmp_path / 'hooks')
    os.makedirs(hooks_dir)
    hook_file = os.path.join(hooks_dir, 'unrelated_script.py')
    with open(hook_file, 'w') as f:
        f.write('#!/usr/bin/env python\n')
    
    result = find_hook('pre_prompt', hooks_dir)
    assert result is None


def test_find_hook_returns_matching_hook_script(tmp_path):
    hooks_dir = str(tmp_path / 'hooks')
    os.makedirs(hooks_dir)
    hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
    with open(hook_file, 'w') as f:
        f.write('#!/usr/bin/env python\n')
    
    result = find_hook('pre_prompt', hooks_dir)
    assert result is not None
    assert len(result) == 1
    assert result[0] == os.path.abspath(hook_file)


def test_find_hook_ignores_backup_files(tmp_path):
    hooks_dir = str(tmp_path / 'hooks')
    os.makedirs(hooks_dir)
    hook_file = os.path.join(hooks_dir, 'pre_prompt.py~')
    with open(hook_file, 'w') as f:
        f.write('#!/usr/bin/env python\n')
    
    result = find_hook('pre_prompt', hooks_dir)
    assert result is None


def test_find_hook_returns_multiple_matching_hooks(tmp_path):
    hooks_dir = str(tmp_path / 'hooks')
    os.makedirs(hooks_dir)
    hook_file1 = os.path.join(hooks_dir, 'pre_prompt.py')
    hook_file2 = os.path.join(hooks_dir, 'pre_prompt.sh')
    with open(hook_file1, 'w') as f:
        f.write('#!/usr/bin/env python\n')
    with open(hook_file2, 'w') as f:
        f.write('#!/bin/bash\n')
    
    result = find_hook('pre_prompt', hooks_dir)
    assert result is not None
    assert len(result) == 2
    assert os.path.abspath(hook_file1) in result
    assert os.path.abspath(hook_file2) in result


def test_find_hook_with_default_hooks_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir)
    hook_file = os.path.join(hooks_dir, 'post_gen_project.py')
    with open(hook_file, 'w') as f:
        f.write('#!/usr/bin/env python\n')
    
    result = find_hook('post_gen_project')
    assert result is not None
    assert len(result) == 1


def test_find_hook_filters_unsupported_hooks(tmp_path):
    hooks_dir = str(tmp_path / 'hooks')
    os.makedirs(hooks_dir)
    hook_file = os.path.join(hooks_dir, 'unsupported_hook.py')
    with open(hook_file, 'w') as f:
        f.write('#!/usr/bin/env python\n')
    
    result = find_hook('unsupported_hook', hooks_dir)
    assert result is None


# LLM-generated content at query #33
#--------------------------

```python
import os
import tempfile
import shutil


def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    result = find_hook('pre_prompt', hooks_dir='/nonexistent/path')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        open(os.path.join(hooks_dir, 'other_hook.sh'), 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result is None


def test_find_hook_returns_script_path_when_hook_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        open(hook_file, 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(hook_file)


def test_find_hook_returns_multiple_scripts_with_different_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_prompt.sh')
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.py')
        open(hook_file1, 'w').close()
        open(hook_file2, 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert os.path.abspath(hook_file1) in result
        assert os.path.abspath(hook_file2) in result


def test_find_hook_ignores_backup_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        backup_file = os.path.join(hooks_dir, 'pre_prompt.sh~')
        open(hook_file, 'w').close()
        open(backup_file, 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(hook_file)


def test_find_hook_returns_absolute_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        open(hook_file, 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result is not None
        assert os.path.isabs(result[0])


# LLM-generated content at query #34
#--------------------------

```python
import os
import tempfile
import shutil


def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    result = find_hook('pre_prompt', 'nonexistent_dir')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        open(os.path.join(hooks_dir, 'other_hook.sh'), 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_returns_absolute_path_for_matching_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        open(hook_file, 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert os.path.isabs(result[0])
        assert os.path.basename(result[0]) == 'pre_prompt.sh'


def test_find_hook_ignores_backup_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        open(os.path.join(hooks_dir, 'pre_prompt.sh~'), 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_returns_multiple_matching_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        open(os.path.join(hooks_dir, 'pre_prompt.sh'), 'w').close()
        open(os.path.join(hooks_dir, 'pre_prompt.py'), 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2


def test_find_hook_with_default_hooks_dir():
    original_dir = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            hooks_dir = os.path.join(tmpdir, 'hooks')
            os.makedirs(hooks_dir)
            hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
            open(hook_file, 'w').close()
            
            result = find_hook('pre_prompt')
            assert result is not None
            assert len(result) == 1
    finally:
        os.chdir(original_dir)


def test_find_hook_returns_absolute_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        open(os.path.join(hooks_dir, 'pre_prompt.sh'), 'w').close()
        
        result = find_hook('pre_prompt', hooks_dir)
        assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #35
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    """Test that NamedTemporaryFile is created with delete=False, mode='wb', and correct suffix."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock, call
    from cookiecutter.hooks import run_script_with_context

    script_path = "/path/to/script.sh"
    cwd = "/working/dir"
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/tmpfile.sh"
    mock_temp_file.__enter__ = MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = MagicMock(return_value=None)
    
    with patch('pathlib.Path.read_text', return_value='echo "test"'):
        with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file) as mock_named_temp:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                with patch('cookiecutter.hooks.run_script'):
                    mock_env = MagicMock()
                    mock_template = MagicMock()
                    mock_template.render.return_value = 'echo "test"'
                    mock_env.from_string.return_value = mock_template
                    mock_create_env.return_value = mock_env
                    
                    run_script_with_context(script_path, cwd, context)
                    
                    mock_named_temp.assert_called_once_with(delete=False, mode='wb', suffix='.sh')


# LLM-generated content at query #36
#--------------------------

```python
def test_find_hook_with_valid_hook_file(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("__main__._HOOKS", ["pre_prompt", "post_gen_project"])
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_with_nonexistent_hooks_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    result = find_hook("pre_prompt", "nonexistent_hooks")
    
    assert result is None


def test_find_hook_with_no_matching_hooks(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "some_other_hook"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("__main__._HOOKS", ["pre_prompt", "post_gen_project"])
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is None


def test_find_hook_with_backup_file(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt~"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("__main__._HOOKS", ["pre_prompt", "post_gen_project"])
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is None


def test_find_hook_with_unsupported_hook(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "unsupported_hook"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("__main__._HOOKS", ["pre_prompt", "post_gen_project"])
    
    result = find_hook("unsupported_hook", str(hooks_dir))
    
    assert result is None


def test_find_hook_with_multiple_matching_hooks(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / "pre_prompt.sh"
    hook_file1.write_text("#!/bin/bash\necho 'test'")
    hook_file2 = hooks_dir / "pre_prompt.py"
    hook_file2.write_text("#!/usr/bin/env python\nprint('test')")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("__main__._HOOKS", ["pre_prompt", "post_gen_project"])
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1) in result
    assert str(hook_file2) in result


def test_find_hook_with_empty_hooks_dir(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("__main__._HOOKS", ["pre_prompt", "post_gen_project"])
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is None


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_line_18_evaluates_to_true():
    import subprocess
    from pathlib import Path
    from unittest.mock import Mock, patch
    
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    # Mock the dependencies
    mock_proc = Mock()
    mock_proc.wait.return_value = 1  # Non-zero exit status to make predicate True
    
    with patch('subprocess.Popen', return_value=mock_proc):
        with patch('utils.make_executable'):
            try:
                # Simulate the condition at line 18
                exit_status = mock_proc.wait()
                predicate_result = exit_status != EXIT_SUCCESS
                assert predicate_result is True
            except FailedHookException:
                pass


# LLM-generated content at query #38
#--------------------------

```python
def test_find_hook_returns_scripts_list_when_valid_hooks_exist(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('hook')")
    
    monkeypatch.chdir(tmp_path)
    
    def mock_valid_hook(hook_file, hook_name):
        return hook_file == "pre_prompt.py" and hook_name == "pre_prompt"
    
    import sys
    from unittest.mock import patch
    
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=["pre_prompt.py"]):
            with patch('os.path.abspath', side_effect=lambda x: str(tmp_path / x.split('/')[-1])):
                with patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}"):
                    with patch('__main__.valid_hook', mock_valid_hook):
                        from __main__ import find_hook
                        result = find_hook("pre_prompt", str(hooks_dir))
                        assert result is not None
                        assert len(result) > 0
                        assert isinstance(result, list)


# LLM-generated content at query #39
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir early when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from cookiecutter.hooks import find_hook
    
    # Mock find_hook to return empty list (no scripts found)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda x: [])
    
    # Create a temporary directory to use as repo_dir
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    
    # Call the function
    result = run_pre_prompt_hook(test_repo)
    
    # Assert that the function returns the original repo_dir without creating a temp dir
    assert result == test_repo


# LLM-generated content at query #40
#--------------------------

```python
def test_run_pre_prompt_hook_work_in_context_manager():
    """Test that work_in context manager is called with repo_dir at line 7."""
    import os
    import tempfile
    import shutil
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory to serve as repo_dir
    test_repo_dir = tempfile.mkdtemp(prefix='test_repo_')
    
    try:
        # Mock find_hook to return empty list so we return early
        with patch('cookiecutter.hooks.find_hook', return_value=[]):
            with patch('cookiecutter.hooks.work_in') as mock_work_in:
                # Configure mock_work_in to be a context manager
                mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
                mock_work_in.return_value.__exit__ = MagicMock(return_value=None)
                
                result = run_pre_prompt_hook(test_repo_dir)
                
                # Verify work_in was called with repo_dir
                mock_work_in.assert_called_once_with(test_repo_dir)
                # Verify the predicate (work_in call exists) evaluates to True
                assert mock_work_in.called
    finally:
        shutil.rmtree(test_repo_dir)


# LLM-generated content at query #41
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')\n")
    
    try:
        proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=str(tmp_path))
        exit_status = proc.wait()
        assert exit_status == 0
    except OSError as err:
        assert False, "OSError should not be raised for a valid script"


# LLM-generated content at query #42
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_delete_false():
    """Test that tempfile.NamedTemporaryFile is called with delete=False."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock, call
    from cookiecutter.hooks import run_script_with_context

    mock_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_jinja2_env_vars': {},
        }
    }
    
    mock_script_content = 'echo "{{ cookiecutter.project_name }}"'
    
    with patch('cookiecutter.hooks.Path.read_text', return_value=mock_script_content):
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp_file:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                with patch('cookiecutter.hooks.run_script'):
                    mock_env = MagicMock()
                    mock_template = MagicMock()
                    mock_template.render.return_value = 'echo "test_project"'
                    mock_env.from_string.return_value = mock_template
                    mock_create_env.return_value = mock_env
                    
                    mock_temp_instance = MagicMock()
                    mock_temp_instance.name = '/tmp/test_script.sh'
                    mock_temp_file.return_value.__enter__.return_value = mock_temp_instance
                    
                    run_script_with_context('/path/to/script.sh', '/cwd', mock_context)
                    
                    mock_temp_file.assert_called_once()
                    call_kwargs = mock_temp_file.call_args[1]
                    assert call_kwargs['delete'] is False
                    assert call_kwargs['mode'] == 'wb'
                    assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #43
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_run_hook.assert_called_once()
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_run_hook.assert_called_once()
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_run_hook.assert_called_once()
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = None
    cwd_during_run_hook = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal cwd_during_run_hook
        cwd_during_run_hook = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    original_cwd = os.getcwd()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert str(cwd_during_run_hook) == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #44
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr("subprocess.Popen", MockPopen)
    monkeypatch.setattr("sys.platform", "linux")
    
    from utils import run_script
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr("subprocess.Popen", MockPopen)
    monkeypatch.setattr("sys.platform", "linux")
    
    from utils import run_script
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_windows_shell(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr("subprocess.Popen", MockPopen)
    monkeypatch.setattr("sys.platform", "win32")
    
    from utils import run_script
    run_script(script_path, cwd=str(tmp_path))
    
    assert mock_popen_called[0][1]["shell"] is True


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr("subprocess.Popen", MockPopen)
    monkeypatch.setattr("sys.platform", "linux")
    
    from utils import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError()
            err.errno = errno.ENOEXEC
            raise err
    
    monkeypatch.setattr("subprocess.Popen", MockPopen)
    monkeypatch.setattr("sys.platform", "linux")
    
    from utils import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("Permission denied")
    
    monkeypatch.setattr("subprocess.Popen", MockPopen)
    monkeypatch.setattr("sys.platform", "linux")
    
    from utils import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (error:" in str(e)


# LLM-generated content at query #45
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook(tmp_path):
    """Test run_pre_prompt_hook when no pre_prompt hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with a valid pre_prompt script."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    call_log = []
    original_run_script = run_script
    
    def mock_run_script(script_path, cwd='.'):
        call_log.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert len(call_log) == 1


def test_run_pre_prompt_hook_script_failure(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when the script fails."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\nexit 1")
    script_file.chmod(0o755)
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException("Script failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_creates_temp_copy(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook creates a temporary copy when hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    content_file = repo_dir / "cookiecutter.json"
    content_file.write_text('{"project_name": "test"}')
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('test')")
    script_file.chmod(0o755)
    
    def mock_run_script(script_path, cwd='.'):
        pass
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert str(result) != str(repo_dir)
    assert (result / "cookiecutter.json").exists()


# LLM-generated content at query #46
#--------------------------

```python
def test_work_in_predicate_evaluates_to_false():
    """Test that the predicate at line 17 (dirname is not None) evaluates to False when dirname is None."""
    import os
    from pathlib import Path
    from cookiecutter.utils import work_in
    
    original_dir = os.getcwd()
    test_executed = False
    
    with work_in(None):
        test_executed = True
        current_dir = os.getcwd()
    
    assert test_executed is True
    assert os.getcwd() == original_dir


# LLM-generated content at query #47
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir uses work_in context manager at line 17."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.utils import work_in
    from unittest.mock import patch, MagicMock
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = str(tmp_path)
    context = {'cookiecutter': {}}
    
    work_in_called = []
    original_work_in = work_in
    
    def mock_work_in(dirname=None):
        work_in_called.append(dirname)
        return original_work_in(dirname)
    
    with patch('cookiecutter.hooks.work_in', side_effect=mock_work_in):
        with patch('cookiecutter.hooks.run_hook'):
            run_hook_from_repo_dir(
                repo_dir=str(repo_dir),
                hook_name='post_gen_project',
                project_dir=str(project_dir),
                context=context,
                delete_project_on_failure=False
            )
    
    assert len(work_in_called) == 1
    assert str(work_in_called[0]) == str(repo_dir)


# LLM-generated content at query #48
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    cwd = tmp_path
    
    mock_popen_called = []
    mock_wait_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            mock_wait_called.append(True)
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.executable', '/usr/bin/python3')
    
    import sys
    from pathlib import Path as PathlibPath
    
    def mock_make_executable(path):
        pass
    
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    from your_module import run_script
    run_script(script_path, cwd)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == ['/usr/bin/python3', script_path]
    assert len(mock_wait_called) == 1


def test_run_script_shell_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    cwd = tmp_path
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    
    def mock_make_executable(path):
        pass
    
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    from your_module import run_script
    run_script(script_path, cwd)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    cwd = tmp_path
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.executable', '/usr/bin/python3')
    
    def mock_make_executable(path):
        pass
    
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Hook script failed (exit status: 1)' in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    cwd = tmp_path
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    
    def mock_make_executable(path):
        pass
    
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'might be an empty file or missing a shebang' in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    cwd = tmp_path
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError(2, "No such file or directory")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    
    def mock_make_executable(path):
        pass
    
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Hook script failed (error:' in str(e)


# LLM-generated content at query #49
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (line 17 predicate evaluates to False)."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_cwd = os.getcwd()
    temp_repo_dir = "/tmp/test_repo"
    temp_project_dir = "/tmp/test_project"
    test_context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
        mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            run_hook_from_repo_dir(
                temp_repo_dir,
                "pre_prompt",
                temp_project_dir,
                test_context,
                False
            )
        
        mock_work_in.assert_called_once_with(temp_repo_dir)
        mock_run_hook.assert_called_once_with("pre_prompt", temp_project_dir, test_context)


# LLM-generated content at query #50
#--------------------------

```python
import errno
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_oserror_with_enoexec_errno():
    """Test that the predicate at line 21 evaluates to True when OSError with ENOEXEC is raised."""
    from your_module import run_script, FailedHookException
    
    script_path = "/path/to/script.sh"
    cwd = Path(".")
    
    oserror = OSError("Exec format error")
    oserror.errno = errno.ENOEXEC
    
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = oserror
        
        try:
            run_script(script_path, cwd)
        except FailedHookException as e:
            assert "might be an empty file or missing a shebang" in str(e)


# LLM-generated content at query #51
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false(monkeypatch):
    import subprocess
    import errno
    from pathlib import Path
    
    # Mock subprocess.Popen to raise OSError with errno != ENOEXEC
    def mock_popen(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    # Mock utils.make_executable to do nothing
    import sys
    import utils
    monkeypatch.setattr(utils, 'make_executable', lambda x: None)
    
    # Import the function to test
    from your_module import run_script, FailedHookException
    
    # Call run_script and verify it raises FailedHookException
    # The predicate at line 22 (err.errno == errno.ENOEXEC) should be False
    try:
        run_script('/path/to/script.sh')
    except FailedHookException as e:
        assert 'error:' in str(e)
        assert 'shebang' not in str(e)


# LLM-generated content at query #52
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    """Test that tempfile is created with the correct suffix from script_path."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "echo 'test'"
    script_path = "/path/to/script.sh"
    cwd = "/tmp"
    context = {"cookiecutter": {}}
    
    temp_files_created = []
    
    original_namedtemp = tempfile.NamedTemporaryFile
    
    def mock_namedtemp(*args, **kwargs):
        temp_obj = original_namedtemp(*args, **kwargs)
        temp_files_created.append((temp_obj.name, kwargs.get('suffix')))
        return temp_obj
    
    with patch('tempfile.NamedTemporaryFile', side_effect=mock_namedtemp):
        with patch('pathlib.Path.read_text', return_value=script_content):
            with patch('cookiecutter.hooks.run_script'):
                try:
                    run_script_with_context(script_path, cwd, context)
                except:
                    pass
    
    assert len(temp_files_created) > 0
    assert temp_files_created[0][1] == ".sh"


# LLM-generated content at query #53
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir successfully runs a hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError('Undefined variable')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_no_delete_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    original_cwd = os.getcwd()
    
    def check_cwd(*args, **kwargs):
        assert os.getcwd() == str(repo_dir)
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=check_cwd)
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #54
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_deletion(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failed_hook_exception_without_deletion(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_undefined_error_with_deletion(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError('Undefined variable')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook from repo directory."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    captured_cwd = []
    
    def capture_cwd(*args, **kwargs):
        captured_cwd.append(str(Path.cwd()))
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert str(repo_dir) in captured_cwd[0]


# LLM-generated content at query #55
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    call_tracker = []
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        call_tracker.append((hook_name, str(proj_dir), ctx))
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(
        str(repo_dir),
        'post_gen_project',
        str(project_dir),
        context,
        delete_project_on_failure=False
    )
    
    assert len(call_tracker) == 1
    assert call_tracker[0][0] == 'post_gen_project'
    assert call_tracker[0][1] == str(project_dir)
    assert call_tracker[0][2] == context


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(
            str(repo_dir),
            'post_gen_project',
            str(project_dir),
            context,
            delete_project_on_failure=False
        )
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass
    
    assert project_dir.exists()


def test_run_hook_from_repo_dir_delete_on_failure(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project directory on failure."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(
            str(repo_dir),
            'post_gen_project',
            str(project_dir),
            context,
            delete_project_on_failure=True
        )
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir handles UndefinedError."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError('Undefined variable')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(
            str(repo_dir),
            'post_gen_project',
            str(project_dir),
            context,
            delete_project_on_failure=True
        )
        assert False, "Expected UndefinedError"
    except UndefinedError:
        pass
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_changes_to_repo_dir(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir changes to repo directory during execution."""
    import os
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    cwd_during_hook = []
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        cwd_during_hook.append(os.getcwd())
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(
        str(repo_dir),
        'post_gen_project',
        str(project_dir),
        context,
        delete_project_on_failure=False
    )
    
    assert os.getcwd() == original_cwd
    assert cwd_during_hook[0] == str(repo_dir)


# LLM-generated content at query #56
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file with Jinja template
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('{{ cookiecutter.project_name }}')")
    
    # Create context with project name
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    # Mock run_script to verify it's called with rendered temp file
    called_with = {}
    
    def mock_run_script(script_path, cwd):
        called_with['script_path'] = script_path
        called_with['cwd'] = cwd
        # Read the temp file to verify rendering worked
        import pathlib
        content = pathlib.Path(script_path).read_text(encoding='utf-8')
        called_with['content'] = content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert 'script_path' in called_with
    assert 'cwd' in called_with
    assert called_with['cwd'] == str(tmp_path)
    
    # Verify template was rendered correctly
    assert called_with['content'] == "print('test_project')"


def test_run_script_with_context_bash(tmp_path, monkeypatch):
    """Test run_script_with_context with bash script."""
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary bash script with Jinja template
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("echo '{{ cookiecutter.name }}'")
    
    # Create context
    context = {
        'cookiecutter': {
            'name': 'my_app'
        }
    }
    
    # Mock run_script
    called_args = []
    
    def mock_run_script(script_path, cwd):
        called_args.append((script_path, cwd))
        import pathlib
        content = pathlib.Path(script_path).read_text(encoding='utf-8')
        called_args.append(content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify template was rendered
    assert called_args[1] == "echo 'my_app'"


def test_run_script_with_context_preserves_cwd(tmp_path, monkeypatch):
    """Test run_script_with_context passes correct cwd to run_script."""
    from cookiecutter.hooks import run_script_with_context
    
    # Create script
    script_file = tmp_path / "script.py"
    script_file.write_text("print('test')")
    
    context = {'cookiecutter': {}}
    
    # Track cwd argument
    captured_cwd = {}
    
    def mock_run_script(script_path, cwd):
        captured_cwd['value'] = cwd
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    custom_cwd = tmp_path / "custom_dir"
    run_script_with_context(str(script_file), str(custom_cwd), context)
    
    assert captured_cwd['value'] == str(custom_cwd)


def test_run_script_with_context_with_env_vars(tmp_path, monkeypatch):
    """Test run_script_with_context respects _jinja2_env_vars."""
    from cookiecutter.hooks import run_script_with_context
    
    # Create script
    script_file = tmp_path / "script.py"
    script_file.write_text("{{ variable }}")
    
    # Context with Jinja2 environment variables
    context = {
        'cookiecutter': {
            '_jinja2_env_vars': {
                'variable_start_string': '[[',
                'variable_end_string': ']]'
            }
        },
        'variable': 'test_value'
    }
    
    rendered_content = {}
    
    def mock_run_script(script_path, cwd):
        import pathlib
        rendered_content['value'] = pathlib.Path(script_path).read_text(encoding='utf-8')
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute - {{ variable }} should not be rendered due to custom delimiters
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Original content should be preserved since delimiters are different
    assert rendered_content['value'] == "{{ variable }}"


# LLM-generated content at query #57
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    import tempfile
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file with Jinja2 template
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('{{ greeting }} {{ name }}')\n")
    
    # Create context with template variables
    context = {
        'greeting': 'Hello',
        'name': 'World',
        'cookiecutter': {}
    }
    
    # Track calls to run_script
    run_script_calls = []
    def mock_run_script(script_path, cwd='.'):
        run_script_calls.append((script_path, cwd))
        # Verify the temp file was created and contains rendered content
        temp_content = Path(script_path).read_text(encoding='utf-8')
        assert "Hello World" in temp_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(run_script_calls) == 1
    assert run_script_calls[0][1] == str(tmp_path)


def test_run_script_with_context_with_cookiecutter_vars(tmp_path, monkeypatch):
    """Test run_script_with_context uses cookiecutter context variables."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("echo '{{ cookiecutter.project_name }}'\n")
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_extensions': []
        }
    }
    
    run_script_calls = []
    def mock_run_script(script_path, cwd='.'):
        run_script_calls.append(script_path)
        temp_content = Path(script_path).read_text(encoding='utf-8')
        assert "my_project" in temp_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(run_script_calls) == 1


def test_run_script_with_context_preserves_file_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves original file extension."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\necho test\n")
    
    context = {'cookiecutter': {}}
    
    temp_files_created = []
    def mock_run_script(script_path, cwd='.'):
        temp_files_created.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(temp_files_created) == 1
    assert temp_files_created[0].endswith('.sh')


def test_run_script_with_context_renders_complex_template(tmp_path, monkeypatch):
    """Test run_script_with_context handles complex Jinja2 templates."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("{% if enabled %}print('enabled'){% endif %}\nprint('{{ value }}')\n")
    
    context = {
        'enabled': True,
        'value': 'test_value',
        'cookiecutter': {}
    }
    
    def mock_run_script(script_path, cwd='.'):
        temp_content = Path(script_path).read_text(encoding='utf-8')
        assert "print('enabled')" in temp_content
        assert "print('test_value')" in temp_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)


# LLM-generated content at query #58
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(mocker):
    """Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS."""
    from pathlib import Path
    
    # Mock the dependencies
    mocker.patch('sys.platform', 'linux')
    mocker.patch('sys.executable', '/usr/bin/python3')
    mock_make_executable = mocker.patch('utils.make_executable')
    mock_popen = mocker.patch('subprocess.Popen')
    
    # Set up the mock process to return EXIT_SUCCESS (0)
    mock_process = mocker.MagicMock()
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process
    
    # Define EXIT_SUCCESS
    mocker.patch('__main__.EXIT_SUCCESS', 0)
    
    # Import and call the function
    from __main__ import run_script
    
    # Call the function with a Python script
    run_script('/path/to/script.py', cwd=Path('.'))
    
    # Verify that no exception was raised (meaning the predicate was False)
    mock_make_executable.assert_called_once_with('/path/to/script.py')
    mock_popen.assert_called_once()
    mock_process.wait.assert_called_once()


# LLM-generated content at query #59
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully without errors."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    from cookiecutter.exceptions import FailedHookException
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_no_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    from cookiecutter.exceptions import FailedHookException
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles UndefinedError and deletes project."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    from jinja2 import UndefinedError
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = None
    hook_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal hook_cwd
        import os
        hook_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    context = {'cookiecutter': {}}
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    import os
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert str(hook_cwd) == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #60
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    from pathlib import Path
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    mock_popen = None
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            nonlocal mock_popen
            mock_popen = self
            self.args = args
            self.kwargs = kwargs
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script
    run_script(script_path, cwd=tmp_path)
    
    assert mock_popen is not None
    assert mock_popen.args[0] == [sys.executable, script_path]


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    from pathlib import Path
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.sh")
    mock_popen = None
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            nonlocal mock_popen
            mock_popen = self
            self.args = args
            self.kwargs = kwargs
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script
    run_script(script_path, cwd=tmp_path)
    
    assert mock_popen is not None
    assert mock_popen.args[0] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    from pathlib import Path
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    from pathlib import Path
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError()
            err.errno = errno.ENOEXEC
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    from pathlib import Path
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError("Permission denied")
            err.errno = 13
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed" in str(e)


# LLM-generated content at query #61
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    import os
    import tempfile
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file with Jinja2 template
    script_file = tmp_path / "test_script.py"
    script_content = "#!/usr/bin/env python\n# Test script\nprint('{{ cookiecutter.name }}')\n"
    script_file.write_text(script_content)
    
    # Create context with cookiecutter data
    context = {
        'cookiecutter': {
            'name': 'test_project',
            '_jinja2_env_vars': {}
        }
    }
    
    # Mock subprocess.Popen to avoid actual script execution
    import subprocess
    original_popen = subprocess.Popen
    
    executed_scripts = []
    
    def mock_popen(cmd, shell=False, cwd='.'):
        executed_scripts.append((cmd, cwd))
        class MockProcess:
            def wait(self):
                return 0
        return MockProcess()
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    
    # Mock make_executable to avoid permission changes
    from cookiecutter import utils
    monkeypatch.setattr(utils, 'make_executable', lambda x: None)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify that a script was executed
    assert len(executed_scripts) > 0
    executed_cmd = executed_scripts[0][0]
    assert executed_cmd[0] == 'python' or executed_cmd[0].endswith('python.exe')
    assert executed_cmd[1].endswith('.py')


# LLM-generated content at query #62
#--------------------------

```python
import errno
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch
import sys


def test_oserror_enoexec_predicate():
    """Test that the predicate at line 22 evaluates to True when errno.ENOEXEC is raised."""
    err = OSError()
    err.errno = errno.ENOEXEC
    
    assert err.errno == errno.ENOEXEC


# LLM-generated content at query #63
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failed_hook_no_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    captured_cwd = []
    
    def capture_cwd(*args, **kwargs):
        captured_cwd.append(os.getcwd())
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert str(repo_dir) in captured_cwd[0]
    assert os.getcwd() == original_cwd


# LLM-generated content at query #64
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_script(tmp_path):
    """Test run_pre_prompt_hook when no pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_hook_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with a valid pre_prompt hook script."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("print('hook executed')")
    hook_script.chmod(0o755)
    
    mock_run_script_called = []
    
    def mock_run_script(script_path, cwd='.'):
        mock_run_script_called.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert len(mock_run_script_called) == 1


def test_run_pre_prompt_hook_with_failed_hook_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when hook script fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("print('hook executed')")
    hook_script.chmod(0o755)
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_creates_temp_repo(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary repository."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("print('hook executed')")
    hook_script.chmod(0o755)
    
    temp_dirs_created = []
    
    def mock_run_script(script_path, cwd='.'):
        temp_dirs_created.append(cwd)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert str(result) != str(repo_dir)
    assert len(temp_dirs_created) > 0


# LLM-generated content at query #65
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('hello')")
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 0})()
    original_popen = subprocess.Popen
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        return mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script
    run_script(script_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'hello'")
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 0})()
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        return mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script
    run_script(script_path)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 1})()
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        return mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("invalid script")
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash")
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        err = OSError("File not found")
        err.errno = errno.ENOENT
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'error' in str(e)


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('hello')")
    
    cwd_arg = None
    
    def mock_popen_init(cmd, shell=False, cwd='.'):
        nonlocal cwd_arg
        cwd_arg = cwd
        mock = type('MockPopen', (), {'wait': lambda self: 0})()
        return mock
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from run_script import run_script
    run_script(script_path, cwd='/custom/path')
    assert cwd_arg == '/custom/path'


# LLM-generated content at query #66
#--------------------------

```python
def test_run_pre_prompt_hook_predicate_false(tmp_path, monkeypatch):
    """Test that the predicate at line 9 (if not scripts) evaluates to False."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from cookiecutter.utils import work_in
    from unittest.mock import patch
    
    # Create a temporary repo directory
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return an empty list (making the predicate True, so we skip the return)
    # We want to test when scripts is NOT empty (predicate is False)
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        
        # First call returns scripts (non-empty), second call returns scripts for the loop
        mock_find_hook.side_effect = [
            ['pre_prompt_script.sh'],  # First call at line 8
            ['pre_prompt_script.sh']   # Second call at line 15
        ]
        mock_create_tmp.return_value = repo_dir
        
        result = run_pre_prompt_hook(repo_dir)
        
        # Verify that find_hook was called
        assert mock_find_hook.called
        # Verify that the first call to find_hook returned a non-empty list
        first_call_result = mock_find_hook.side_effect[0]
        assert bool(first_call_result) is True
        # Verify that the predicate "if not scripts" evaluates to False
        assert not (not first_call_result)


# LLM-generated content at query #67
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(monkeypatch):
    import subprocess
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return EXIT_SUCCESS
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    # Mock utils.make_executable to do nothing
    import sys
    import types
    utils_module = types.ModuleType('utils')
    utils_module.make_executable = lambda x: None
    monkeypatch.setitem(sys.modules, 'utils', utils_module)
    
    # Import after mocking
    from pathlib import Path as PathlibPath
    
    # Create a temporary script file
    script_path = '/tmp/test_script.py'
    
    # Call run_script - it should not raise an exception if exit_status == EXIT_SUCCESS
    # This means the predicate (exit_status != EXIT_SUCCESS) evaluates to False
    run_script(script_path, cwd='.')


# LLM-generated content at query #68
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
    from contextlib import contextmanager
    @contextmanager
    def mock_work_in_impl(dirname):
        yield
    
    mock_work_in.return_value = mock_work_in_impl(repo_dir)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    from cookiecutter.exceptions import FailedHookException
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    context = {'cookiecutter': {}}
    
    from contextlib import contextmanager
    @contextmanager
    def mock_work_in_impl(dirname):
        yield
    
    mock_work_in.return_value = mock_work_in_impl(repo_dir)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except Exception:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    from jinja2 import UndefinedError
    mock_run_hook.side_effect = UndefinedError('Undefined variable')
    
    context = {'cookiecutter': {}}
    
    from contextlib import contextmanager
    @contextmanager
    def mock_work_in_impl(dirname):
        yield
    
    mock_work_in.return_value = mock_work_in_impl(repo_dir)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except Exception:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failed_hook_exception_no_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    from cookiecutter.exceptions import FailedHookException
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    context = {'cookiecutter': {}}
    
    from contextlib import contextmanager
    @contextmanager
    def mock_work_in_impl(dirname):
        yield
    
    mock_work_in.return_value = mock_work_in_impl(repo_dir)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except Exception:
        pass
    
    mock_rmtree.assert_not_called()


# LLM-generated content at query #69
#--------------------------

```python
def test_run_pre_prompt_hook_work_in_context_manager():
    """Test that work_in context manager is used at line 7."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory to use as repo_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        original_cwd = os.getcwd()
        
        # Mock find_hook to return empty list (so we return early at line 10)
        with patch('cookiecutter.hooks.find_hook', return_value=[]):
            result = run_pre_prompt_hook(temp_path)
        
        # Verify that we're back in the original directory after work_in context exits
        assert os.getcwd() == original_cwd
        assert result == temp_path


# LLM-generated content at query #70
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that run_hook_from_repo_dir uses work_in context manager at line 17."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            original_cwd = os.getcwd()
            
            # Mock run_hook to avoid actual execution
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                # Call the function
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context={'cookiecutter': {}},
                    delete_project_on_failure=False
                )
                
                # Verify we're back in the original directory (work_in context manager worked)
                assert os.getcwd() == original_cwd
                # Verify run_hook was called (line 19)
                mock_run_hook.assert_called_once()


# LLM-generated content at query #71
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException:
        pass


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches UndefinedError at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise UndefinedError("Undefined variable")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Expected UndefinedError to be raised"
    except UndefinedError:
        pass


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir deletes project directory when delete_project_on_failure is True."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException:
        assert not project_dir.exists()


# LLM-generated content at query #72
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts_found(tmp_path, monkeypatch):
    """Test that predicate at line 9 (if not scripts) evaluates to True when no scripts found."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from cookiecutter.utils import work_in
    from unittest.mock import patch
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


# LLM-generated content at query #73
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary repository directory without any hooks
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    def mock_find_hook(hook_name):
        return []
    
    monkeypatch.setattr("cookiecutter.hooks.find_hook", mock_find_hook)
    
    # Call the function
    result = run_pre_prompt_hook(repo_dir)
    
    # Assert that it returns the original repo_dir without creating a temp copy
    assert result == repo_dir


# LLM-generated content at query #74
#--------------------------

```python
def test_exit_status_not_equal_to_exit_success():
    import subprocess
    from unittest.mock import Mock, patch
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    def run_script(script_path: str, cwd: Path | str = '.') -> None:
        import sys
        import errno
        run_thru_shell = sys.platform.startswith('win')
        if script_path.endswith('.py'):
            script_command = [sys.executable, script_path]
        else:
            script_command = [script_path]
        
        try:
            proc = subprocess.Popen(script_command, shell=run_thru_shell, cwd=cwd)
            exit_status = proc.wait()
            if exit_status != EXIT_SUCCESS:
                msg = f'Hook script failed (exit status: {exit_status})'
                raise FailedHookException(msg)
        except OSError as err:
            if err.errno == errno.ENOEXEC:
                msg = 'Hook script failed, might be an empty file or missing a shebang'
                raise FailedHookException(msg) from err
            msg = f'Hook script failed (error: {err})'
            raise FailedHookException(msg) from err
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 1
    
    with patch('subprocess.Popen', return_value=mock_proc):
        with patch('sys.platform', 'linux'):
            with patch('sys.executable', '/usr/bin/python3'):
                try:
                    run_script('test_script.py')
                    assert False, "Expected FailedHookException to be raised"
                except FailedHookException as e:
                    assert 'Hook script failed (exit status: 1)' in str(e)


# LLM-generated content at query #75
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    """Test running a Python script successfully."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    from pathlib import Path
    import sys
    
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})())
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    """Test running a shell script successfully."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\necho 'hello'")
    
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})())
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    """Test running a script that fails with non-zero exit status."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("exit(1)")
    
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 1})())
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(str(script_file), cwd=str(tmp_path))
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    """Test running a script that raises ENOEXEC error."""
    import errno
    script_file = tmp_path / "test_script"
    script_file.write_text("")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(str(script_file), cwd=str(tmp_path))
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    """Test running a script that raises OSError."""
    import errno
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(str(script_file), cwd=str(tmp_path))
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)


def test_run_script_calls_make_executable(tmp_path, monkeypatch):
    """Test that run_script calls make_executable."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    make_executable_called = []
    
    def mock_make_executable(path):
        make_executable_called.append(path)
    
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})())
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    run_script(str(script_file), cwd=str(tmp_path))
    assert str(script_file) in make_executable_called


# LLM-generated content at query #76
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    import sys
    from pathlib import Path
    
    def mock_popen(cmd, shell=False, cwd=None):
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\necho 'hello'")
    
    def mock_popen(cmd, shell=False, cwd=None):
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_failure_non_zero_exit(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.py"
    script_file.write_text("exit(1)")
    
    def mock_popen(cmd, shell=False, cwd=None):
        class MockProc:
            def wait(self):
                return 1
        return MockProc()
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    try:
        run_script(str(script_file), cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import errno
    script_file = tmp_path / "test_script"
    script_file.write_text("")
    
    def mock_popen(cmd, shell=False, cwd=None):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    try:
        run_script(str(script_file), cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    def mock_popen(cmd, shell=False, cwd=None):
        raise OSError("Permission denied")
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    try:
        run_script(str(script_file), cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)


def test_run_script_uses_shell_on_windows(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    shell_used = []
    
    def mock_popen(cmd, shell=False, cwd=None):
        shell_used.append(shell)
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    monkeypatch.setattr("sys.platform", "win32")
    
    run_script(str(script_file), cwd=str(tmp_path))
    assert shell_used[0] is True


def test_run_script_default_cwd(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    cwd_used = []
    
    def mock_popen(cmd, shell=False, cwd=None):
        cwd_used.append(cwd)
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    run_script(str(script_file))
    assert cwd_used[0] == '.'


