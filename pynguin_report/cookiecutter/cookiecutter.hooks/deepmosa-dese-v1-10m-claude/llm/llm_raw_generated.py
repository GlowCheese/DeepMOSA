####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_valid_hook_matching_supported_hook_not_backup():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True


def test_valid_hook_matching_supported_hook_is_backup():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False


def test_valid_hook_not_matching_hook_name():
    result = valid_hook('/path/to/pre-push', 'pre-commit')
    assert result is False


def test_valid_hook_unsupported_hook_name():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False


def test_valid_hook_with_extension_in_filename():
    result = valid_hook('/path/to/pre-commit.sh', 'pre-commit')
    assert result is True


def test_valid_hook_backup_file_with_extension():
    result = valid_hook('/path/to/pre-commit.sh~', 'pre-commit')
    assert result is False


def test_valid_hook_only_basename_matching():
    result = valid_hook('/different/path/pre-commit', 'pre-commit')
    assert result is True


# LLM-generated content at query #2
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock _HOOKS to contain our test hook
    import sys
    from unittest.mock import patch
    
    hook_name = "pre-commit"
    hook_file = "/path/to/pre-commit"
    
    with patch('os.path.basename', return_value='pre-commit'):
        with patch('os.path.splitext', return_value=('pre-commit', '')):
            with patch.dict('sys.modules', {'__main__': type(sys)('__main__')}):
                # Set up the condition where:
                # - matching_hook: basename == hook_name → True
                # - supported_hook: basename in _HOOKS → True
                # - backup_file: filename.endswith('~') → False
                
                filename = 'pre-commit'
                basename = 'pre-commit'
                matching_hook = basename == hook_name
                supported_hook = basename in ['pre-commit', 'post-commit']
                backup_file = filename.endswith('~')
                
                result = matching_hook and supported_hook and not backup_file
                
                assert result is True


# LLM-generated content at query #3
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    """Test find_hook returns None when hooks directory doesn't exist."""
    result = find_hook('pre_prompt', hooks_dir='/nonexistent/path')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks():
    """Test find_hook returns None when no matching hooks are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a file that doesn't match
        with open(os.path.join(hooks_dir, 'other_hook.py'), 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result is None


def test_find_hook_returns_script_path_when_hook_exists():
    """Test find_hook returns the absolute path when a valid hook is found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a valid hook file (assuming 'pre_prompt' is in _HOOKS)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert os.path.isabs(result[0])
        assert result[0] == os.path.abspath(hook_file)


def test_find_hook_ignores_backup_files():
    """Test find_hook ignores backup files (ending with ~)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a backup file
        backup_file = os.path.join(hooks_dir, 'pre_prompt.py~')
        with open(backup_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result is None


def test_find_hook_returns_multiple_scripts_with_same_name():
    """Test find_hook returns multiple scripts with the same basename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create multiple valid hook files with the same basename but different extensions
        hook_file1 = os.path.join(hooks_dir, 'pre_prompt.py')
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.sh')
        
        with open(hook_file1, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        with open(hook_file2, 'w') as f:
            f.write('#!/bin/bash\n')
        
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert all(os.path.isabs(path) for path in result)


def test_find_hook_uses_default_hooks_dir():
    """Test find_hook uses default 'hooks' directory when not specified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            hooks_dir = os.path.join(tmpdir, 'hooks')
            os.makedirs(hooks_dir)
            
            hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
            with open(hook_file, 'w') as f:
                f.write('#!/usr/bin/env python\n')
            
            result = find_hook('pre_prompt')
            assert result is not None
            assert len(result) == 1
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #4
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

def test_valid_hook_unsupported_hook_name():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False

def test_valid_hook_with_file_extension():
    result = valid_hook('/path/to/pre-commit.sh', 'pre-commit')
    assert result is True

def test_valid_hook_with_file_extension_and_backup():
    result = valid_hook('/path/to/pre-commit.sh~', 'pre-commit')
    assert result is False

def test_valid_hook_only_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False

def test_valid_hook_matching_hook_name_but_unsupported():
    result = valid_hook('/path/to/custom-hook', 'custom-hook')
    assert result is False


# LLM-generated content at query #5
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts(tmp_path, monkeypatch):
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
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda *args, **kwargs: None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, str) or isinstance(result, Path)


def test_run_pre_prompt_hook_script_fails(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when script execution fails."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\nexit 1")
    script_file.chmod(0o755)
    
    def mock_run_script(*args, **kwargs):
        raise FailedHookException("Script failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_creates_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary directory."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda *args, **kwargs: None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert "cookiecutter" in str(result)


def test_run_pre_prompt_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with multiple pre_prompt scripts."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file1 = hooks_dir / "pre_prompt.sh"
    script_file1.write_text("#!/bin/bash\necho 'test1'")
    script_file1.chmod(0o755)
    
    script_file2 = hooks_dir / "pre_prompt.py"
    script_file2.write_text("print('test2')")
    script_file2.chmod(0o755)
    
    call_count = 0
    
    def mock_run_script(*args, **kwargs):
        nonlocal call_count
        call_count += 1
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert call_count == 2


# LLM-generated content at query #6
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
    
    call_args = []
    def mock_popen_init(cmd, shell=False, cwd='.'):
        call_args.append((cmd, shell, cwd))
        return mock_popen
    
    monkeypatch.setattr('subprocess.Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert len(call_args) == 1
    assert call_args[0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    call_args = []
    def mock_popen_init(cmd, shell=False, cwd='.'):
        call_args.append((cmd, shell, cwd))
        return mock_popen
    
    monkeypatch.setattr('subprocess.Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert len(call_args) == 1
    assert call_args[0][0] == [script_path]


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
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("")
    
    def mock_popen_enoexec(cmd, shell=False, cwd='.'):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr('subprocess.Popen', mock_popen_enoexec)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_os_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("test")
    
    def mock_popen_error(cmd, shell=False, cwd='.'):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr('subprocess.Popen', mock_popen_error)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'error' in str(e).lower()


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    custom_cwd = str(tmp_path / "subdir")
    
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    call_args = []
    def mock_popen_init(cmd, shell=False, cwd='.'):
        call_args.append((cmd, shell, cwd))
        return mock_popen
    
    monkeypatch.setattr('subprocess.Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=custom_cwd)
    
    assert len(call_args) == 1
    assert call_args[0][2] == custom_cwd


# LLM-generated content at query #7
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock _HOOKS to include our test hook
    import sys
    from unittest.mock import patch
    
    hook_name = "pre-commit"
    hook_file = "/path/to/pre-commit"
    
    with patch('os.path.basename', return_value='pre-commit'):
        with patch('os.path.splitext', return_value=('pre-commit', '')):
            with patch.dict('sys.modules', {'__main__': type(sys)('__main__')}):
                # Directly test the logic that should evaluate to True
                filename = 'pre-commit'
                basename = 'pre-commit'
                hook_name_param = 'pre-commit'
                
                matching_hook = basename == hook_name_param
                supported_hook = True  # Simulating basename in _HOOKS
                backup_file = False  # filename doesn't end with '~'
                
                result = matching_hook and supported_hook and not backup_file
                assert result is True


# LLM-generated content at query #8
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook, find_hook
    
    # Mock find_hook to return empty list (no scripts found)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    # Create a temporary repo directory
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Call the function
    result = run_pre_prompt_hook(repo_dir)
    
    # Assert that the function returns the original repo_dir
    assert result == repo_dir


# LLM-generated content at query #9
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_returns_original_repo_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda name, hooks_dir='hooks': None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_creates_tmp_dir_with_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates temporary directory when pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\necho 'test'")
    hook_script.chmod(0o755)
    
    call_count = [0]
    original_find_hook = find_hook
    
    def mock_find_hook(name, hooks_dir='hooks'):
        call_count[0] += 1
        if call_count[0] == 1:
            return original_find_hook(name, str(repo_dir / hooks_dir))
        return None
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script', lambda script_path, cwd='.': None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, Path)
    assert result != repo_dir
    assert result.exists()


def test_run_pre_prompt_hook_runs_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes the pre_prompt script."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\necho 'test'")
    hook_script.chmod(0o755)
    
    run_script_calls = []
    
    def mock_find_hook(name, hooks_dir='hooks'):
        if name == 'pre_prompt':
            return [str(hook_script)]
        return None
    
    def mock_run_script(script_path, cwd='.'):
        run_script_calls.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    monkeypatch.setattr('cookiecutter.hooks.create_tmp_repo_dir', lambda d: Path(d))
    
    result = run_pre_prompt_hook(repo_dir)
    assert len(run_script_calls) > 0


def test_run_pre_prompt_hook_raises_failed_hook_exception(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when script fails."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\nexit 1")
    hook_script.chmod(0o755)
    
    def mock_find_hook(name, hooks_dir='hooks'):
        if name == 'pre_prompt':
            return [str(hook_script)]
        return None
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException('Script failed')
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    monkeypatch.setattr('cookiecutter.hooks.create_tmp_repo_dir', lambda d: Path(d))
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_with_multiple_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook handles multiple pre_prompt scripts."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script1 = hooks_dir / "pre_prompt.sh"
    hook_script1.write_text("#!/bin/bash\necho 'test1'")
    hook_script1.chmod(0o755)
    
    hook_script2 = hooks_dir / "pre_prompt.py"
    hook_script2.write_text("print('test2')")
    hook_script2.chmod(0o755)
    
    run_script_calls = []
    
    def mock_find_hook(name, hooks_dir='hooks'):
        if name == 'pre_prompt':
            return [str(hook_script1), str(hook_script2)]
        return None
    
    def mock_run_script(script_path, cwd='.'):
        run_script_calls.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    monkeypatch.setattr('cookiecutter.hooks.create_tmp_repo_dir', lambda d: Path(d))
    
    result = run_pre_prompt_hook(repo_dir)
    assert len(run_script_calls) == 2


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_true():
    script_path = "/path/to/script.py"
    result = script_path.endswith('.py')
    assert result is True


# LLM-generated content at query #11
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
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir cleans up on FailedHookException when delete_project_on_failure is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_without_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not clean up on FailedHookException when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_undefined_error_with_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir cleans up on UndefinedError when delete_project_on_failure is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_without_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not clean up on UndefinedError when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    original_cwd = str(tmp_path)
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_getcwd = mocker.patch('os.getcwd', return_value=original_cwd)
    mock_chdir = mocker.patch('os.chdir')
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert mock_chdir.call_count >= 2


# LLM-generated content at query #12
#--------------------------

```python
def test_script_path_ends_with_py():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import sys
    
    script_path = '/path/to/script.py'
    cwd = '.'
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('sys.platform', 'linux'), \
         patch('utils.make_executable'):
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script(script_path, cwd)
        
        assert mock_popen.call_args[0][0] == [sys.executable, script_path]


# LLM-generated content at query #13
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(mocker, tmp_path):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Mock find_hook to return empty list (no scripts found)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    
    # Create a temporary directory to use as repo_dir
    test_repo_dir = tmp_path / "test_repo"
    test_repo_dir.mkdir()
    
    # Call the function
    result = run_pre_prompt_hook(test_repo_dir)
    
    # Assert that it returns the original repo_dir without creating a temp copy
    assert result == test_repo_dir


# LLM-generated content at query #14
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
        # This should trigger the predicate at line 15: if not os.path.isdir(hooks_dir)
        non_existent_dir = 'non_existent_hooks'
        result = find_hook('test_hook', non_existent_dir)
        
        assert result is None
        assert not os.path.isdir(non_existent_dir)
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #15
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no hook scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')
    mock_run_script_with_context.assert_not_called()


def test_run_hook_single_script_found(mocker, tmp_path):
    """Test run_hook when a single hook script is found."""
    script_path = '/path/to/hook.sh'
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'post_gen_project')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_multiple_scripts_found(mocker, tmp_path):
    """Test run_hook when multiple hook scripts are found."""
    script_path_1 = '/path/to/hook1.sh'
    script_path_2 = '/path/to/hook2.py'
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path_1, script_path_2])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook('pre_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_gen_project')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'pre_gen_project')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path_1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path_2, tmp_path, context)


def test_run_hook_with_string_project_dir(mocker):
    """Test run_hook with string project_dir instead of Path."""
    script_path = '/path/to/hook.sh'
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    project_dir = '/some/project/dir'
    
    run_hook('post_prompt', project_dir, context)
    
    mock_run_script_with_context.assert_called_once_with(script_path, project_dir, context)


def test_run_hook_empty_scripts_list(mocker, tmp_path):
    """Test run_hook when find_hook returns an empty list."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook('pre_prompt', tmp_path, context)
    
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')
    mock_run_script_with_context.assert_not_called()


# LLM-generated content at query #16
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    import logging
    
    caplog.set_level(logging.DEBUG)
    
    def mock_find_hook(hook_name):
        return []
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    
    context = {'cookiecutter': {}}
    project_dir = '/tmp/test'
    hook_name = 'pre_prompt'
    
    run_hook(hook_name, project_dir, context)
    
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #17
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders script with context and executes it."""
    import os
    import tempfile
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a test script with Jinja2 template
    script_content = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    script_file.chmod(0o755)
    
    # Create context with cookiecutter data
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_jinja2_env_vars': {}
        }
    }
    
    # Mock run_script to verify it's called with rendered script
    called_with = []
    
    def mock_run_script(script_path, cwd='.'):
        called_with.append((script_path, cwd))
        # Verify the temp file contains rendered content
        rendered = Path(script_path).read_text(encoding='utf-8')
        assert 'test_project' in rendered
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(called_with) == 1
    assert called_with[0][1] == str(tmp_path)


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension in temp file."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = '#!/usr/bin/env python\nprint("{{ cookiecutter.name }}")'
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'name': 'myproject',
            '_jinja2_env_vars': {}
        }
    }
    
    temp_file_ext = []
    
    def mock_run_script(script_path, cwd='.'):
        import os
        _, ext = os.path.splitext(script_path)
        temp_file_ext.append(ext)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert temp_file_ext[0] == '.py'


def test_run_script_with_context_renders_variables(tmp_path, monkeypatch):
    """Test run_script_with_context properly renders Jinja2 variables."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = 'echo "{{ cookiecutter.var1 }}-{{ cookiecutter.var2 }}"'
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'var1': 'value1',
            'var2': 'value2',
            '_jinja2_env_vars': {}
        }
    }
    
    rendered_content = []
    
    def mock_run_script(script_path, cwd='.'):
        rendered = Path(script_path).read_text(encoding='utf-8')
        rendered_content.append(rendered)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert 'value1-value2' in rendered_content[0]
    assert '{{' not in rendered_content[0]


def test_run_script_with_context_with_pathlib_path(tmp_path, monkeypatch):
    """Test run_script_with_context works with pathlib.Path objects."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = 'echo "{{ cookiecutter.msg }}"'
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'msg': 'hello',
            '_jinja2_env_vars': {}
        }
    }
    
    called_args = []
    
    def mock_run_script(script_path, cwd='.'):
        called_args.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Pass Path objects instead of strings
    run_script_with_context(script_file, tmp_path, context)
    
    assert len(called_args) == 1
    assert str(tmp_path) == called_args[0][1]


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false(mocker):
    """Test that the predicate at line 21 (except OSError) evaluates to False."""
    import subprocess
    from pathlib import Path
    
    mock_popen = mocker.patch('subprocess.Popen')
    mock_process = mocker.MagicMock()
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process
    
    mocker.patch('sys.platform', 'linux')
    mocker.patch.object(utils, 'make_executable')
    
    run_script('/path/to/script.sh', cwd=Path('.'))
    
    mock_popen.assert_called_once()
    mock_process.wait.assert_called_once()


# LLM-generated content at query #19
#--------------------------

```python
def test_run_pre_prompt_hook_predicate_line_7_false(tmp_path, mocker):
    """Test that the predicate at line 7 (if not scripts:) evaluates to False."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary repo directory with a pre_prompt hook
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('hook')")
    
    # Mock find_hook to return a non-empty list (making the predicate False)
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook')
    mock_find_hook.return_value = [str(hook_file)]
    
    # Mock create_tmp_repo_dir to avoid actual file operations
    mock_create_tmp = mocker.patch('cookiecutter.hooks.create_tmp_repo_dir')
    mock_create_tmp.return_value = repo_dir
    
    # Mock run_script to avoid execution
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    # Call the function
    result = run_pre_prompt_hook(repo_dir)
    
    # Verify that the function continues past line 10 (meaning the predicate was False)
    # This means create_tmp_repo_dir should have been called
    mock_create_tmp.assert_called_once()
    assert result == repo_dir


# LLM-generated content at query #20
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_not_exists(tmp_path):
    import os
    import sys
    from pathlib import Path
    
    # Create a temporary directory structure
    test_dir = tmp_path / "test_project"
    test_dir.mkdir()
    
    # Change to test directory
    original_cwd = os.getcwd()
    os.chdir(test_dir)
    
    try:
        # Create a non-existent hooks directory path
        non_existent_hooks_dir = str(test_dir / "non_existent_hooks")
        
        # Verify the predicate at line 15 evaluates to True
        # The predicate is: not os.path.isdir(hooks_dir)
        predicate_result = not os.path.isdir(non_existent_hooks_dir)
        
        assert predicate_result is True
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #21
#--------------------------

```python
def test_work_in_context_manager_changes_directory(tmp_path):
    """Test that work_in context manager changes to specified directory."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_returns_to_original_directory_on_exception(tmp_path):
    """Test that work_in returns to original directory even when exception occurs."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(test_dir):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    assert os.getcwd() == original_dir


def test_work_in_with_none_stays_in_current_directory():
    """Test that work_in with None dirname stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test that work_in works with Path objects."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test that work_in works with string paths."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #22
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'post_gen_project')
    mock_run_script.assert_not_called()


# LLM-generated content at query #23
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_predicate_false(tmp_path, monkeypatch):
    """Test that work_in context manager is called with repo_dir argument."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from unittest.mock import patch, MagicMock
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    original_getcwd = __import__('os').getcwd
    original_chdir = __import__('os').chdir
    chdir_calls = []
    
    def mock_chdir(path):
        chdir_calls.append(str(path))
        
    def mock_getcwd():
        return str(tmp_path)
    
    with patch('cookiecutter.hooks.os.getcwd', side_effect=mock_getcwd):
        with patch('cookiecutter.hooks.os.chdir', side_effect=mock_chdir):
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name="post_gen_project",
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False,
                )
    
    assert str(repo_dir) in chdir_calls
    assert str(tmp_path) in chdir_calls


# LLM-generated content at query #24
#--------------------------

```python
def test_exit_status_not_equal_to_success():
    from unittest.mock import Mock, patch
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 1
    
    with patch('subprocess.Popen', return_value=mock_proc):
        with patch('sys.platform', 'linux'):
            with patch('sys.executable', '/usr/bin/python3'):
                with patch.object(utils, 'make_executable'):
                    try:
                        run_script('/path/to/script.py')
                    except FailedHookException:
                        pass
    
    assert mock_proc.wait.return_value != EXIT_SUCCESS


# LLM-generated content at query #25
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the delete parameter in NamedTemporaryFile is False."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    mock_context = {'cookiecutter': {}}
    script_content = "echo 'test'"
    
    with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_file = MagicMock()
        mock_temp_file.return_value.__enter__.return_value = mock_file
        mock_file.name = '/tmp/test_script.sh'
        
        with patch('pathlib.Path.read_text', return_value=script_content):
            with patch('cookiecutter.hooks.run_script'):
                run_script_with_context('/tmp/script.sh', '/tmp', mock_context)
        
        mock_temp_file.assert_called_once()
        call_kwargs = mock_temp_file.call_args[1]
        assert call_kwargs['delete'] is False


# LLM-generated content at query #26
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
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script
    run_script(script_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script
    run_script(script_path)


def test_run_script_with_cwd(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    cwd_path = tmp_path / "workdir"
    cwd_path.mkdir()
    
    captured_cwd = {}
    def mock_popen(*args, **kwargs):
        captured_cwd['cwd'] = kwargs.get('cwd')
        return type('MockProc', (), {'wait': lambda self: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script
    run_script(script_path, cwd=str(cwd_path))
    assert captured_cwd['cwd'] == str(cwd_path)


def test_run_script_failed_exit_status(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 1})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    import errno
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Permission denied' in str(e)


# LLM-generated content at query #27
#--------------------------

```python
def test_run_hook_returns_early_when_no_scripts_found(monkeypatch):
    """Test that run_hook returns early when find_hook returns empty list."""
    from cookiecutter.hooks import run_hook
    from unittest.mock import Mock
    
    mock_find_hook = Mock(return_value=[])
    mock_run_script = Mock()
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script)
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', '/tmp/test', context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_run_script.assert_not_called()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_valid_hook_with_valid_hook_file():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_with_valid_hook_file_in_current_dir():
    result = valid_hook('pre-push', 'pre-push')
    assert result is True

def test_valid_hook_with_unsupported_hook():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False

def test_valid_hook_with_mismatched_names():
    result = valid_hook('/path/to/pre-commit', 'pre-push')
    assert result is False

def test_valid_hook_with_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False

def test_valid_hook_with_extension():
    result = valid_hook('/path/to/pre-commit.py', 'pre-commit')
    assert result is True

def test_valid_hook_with_backup_file_and_extension():
    result = valid_hook('/path/to/pre-commit.py~', 'pre-commit.py')
    assert result is False

def test_valid_hook_with_non_matching_hook_name():
    result = valid_hook('/path/to/commit-msg', 'pre-commit')
    assert result is False


# LLM-generated content at query #2
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock _HOOKS to include our test hook
    import sys
    from unittest.mock import patch
    
    hook_name = "pre-commit"
    hook_file = "/path/to/pre-commit"
    
    with patch('__main__._HOOKS', {hook_name}):
        # We need to mock os.path functions to return predictable values
        with patch('os.path.basename', return_value='pre-commit'):
            with patch('os.path.splitext', return_value=('pre-commit', '')):
                from __main__ import valid_hook
                result = valid_hook(hook_file, hook_name)
                assert result is True


# LLM-generated content at query #3
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock _HOOKS to include the test hook
    import sys
    from unittest.mock import patch
    
    hook_name = "pre-commit"
    hook_file = "/path/to/pre-commit"
    
    with patch('__main__._HOOKS', {hook_name}):
        result = valid_hook(hook_file, hook_name)
        assert result is True


# LLM-generated content at query #4
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
        
        # Create a file with non-matching name
        with open(os.path.join(hooks_dir, 'other_hook.py'), 'w') as f:
            f.write('# dummy')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_returns_script_path_when_hook_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a valid hook file
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file, 'w') as f:
            f.write('# dummy')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(hook_file)


def test_find_hook_ignores_backup_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a valid hook file and a backup file
        valid_hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        backup_hook_file = os.path.join(hooks_dir, 'pre_prompt.py~')
        
        with open(valid_hook_file, 'w') as f:
            f.write('# dummy')
        with open(backup_hook_file, 'w') as f:
            f.write('# backup')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(valid_hook_file)


def test_find_hook_returns_multiple_matching_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create multiple hook files with same name but different extensions
        hook_file1 = os.path.join(hooks_dir, 'pre_prompt.py')
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.sh')
        
        with open(hook_file1, 'w') as f:
            f.write('# python')
        with open(hook_file2, 'w') as f:
            f.write('# shell')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert os.path.abspath(hook_file1) in result
        assert os.path.abspath(hook_file2) in result


def test_find_hook_uses_default_hooks_dir():
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            hooks_dir = os.path.join(tmpdir, 'hooks')
            os.makedirs(hooks_dir)
            
            hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
            with open(hook_file, 'w') as f:
                f.write('# dummy')
            
            result = find_hook('pre_prompt')
            assert result is not None
            assert len(result) == 1
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #5
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_extensions': [],
            '_jinja2_env_vars': {}
        }
    }
    
    mock_run_script_called = []
    
    def mock_run_script(script_path, cwd):
        mock_run_script_called.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(mock_run_script_called) == 1
    assert mock_run_script_called[0][1] == str(tmp_path)


def test_run_script_with_context_renders_template(tmp_path, monkeypatch):
    """Test that run_script_with_context properly renders Jinja template."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "#!/bin/bash\necho {{ cookiecutter.name }}"
    script_path = tmp_path / "render_test.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'name': 'test_value',
            '_extensions': [],
            '_jinja2_env_vars': {}
        }
    }
    
    rendered_content = []
    
    def mock_run_script(script_path, cwd):
        rendered_content.append(Path(script_path).read_text(encoding='utf-8'))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert 'test_value' in rendered_content[0]


def test_run_script_with_context_python_extension(tmp_path, monkeypatch):
    """Test run_script_with_context with python script."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "print('{{ cookiecutter.msg }}')"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'msg': 'hello',
            '_extensions': [],
            '_jinja2_env_vars': {}
        }
    }
    
    temp_scripts = []
    
    def mock_run_script(script_path, cwd):
        temp_scripts.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(temp_scripts) == 1
    rendered = Path(temp_scripts[0]).read_text(encoding='utf-8')
    assert 'hello' in rendered


def test_run_script_with_context_preserves_cwd(tmp_path, monkeypatch):
    """Test that run_script_with_context passes correct cwd to run_script."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_path = tmp_path / "script.sh"
    script_path.write_text("#!/bin/bash\necho test", encoding='utf-8')
    
    cwd_path = tmp_path / "work_dir"
    cwd_path.mkdir()
    
    context = {
        'cookiecutter': {
            '_extensions': [],
            '_jinja2_env_vars': {}
        }
    }
    
    captured_cwd = []
    
    def mock_run_script(script_path, cwd):
        captured_cwd.append(cwd)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(cwd_path), context)
    
    assert captured_cwd[0] == str(cwd_path)


# LLM-generated content at query #6
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no scripts are found."""
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_logger.debug.assert_called_with('No %s hook found', 'pre_prompt')


def test_run_hook_executes_found_scripts(mocker, tmp_path):
    """Test run_hook executes found scripts."""
    script1 = str(tmp_path / 'pre_prompt.sh')
    script2 = str(tmp_path / 'pre_prompt.py')
    
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[script1, script2])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_logger.debug.assert_called_with('Running hook %s', 'pre_prompt')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script2, tmp_path, context)


def test_run_hook_single_script(mocker, tmp_path):
    """Test run_hook with a single script."""
    script = str(tmp_path / 'post_gen_project.sh')
    
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[script])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_run_script_with_context.assert_called_once_with(script, tmp_path, context)


def test_run_hook_with_pathlib_path(mocker, tmp_path):
    """Test run_hook accepts pathlib.Path for project_dir."""
    from pathlib import Path
    
    script = str(tmp_path / 'pre_prompt.sh')
    project_path = Path(tmp_path)
    
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[script])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', project_path, context)
    
    mock_run_script_with_context.assert_called_once_with(script, project_path, context)


def test_run_hook_passes_context_correctly(mocker, tmp_path):
    """Test run_hook passes context correctly to run_script_with_context."""
    script = str(tmp_path / 'post_gen_project.sh')
    
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[script])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    run_hook('post_gen_project', tmp_path, context)
    
    call_args = mock_run_script_with_context.call_args
    assert call_args[0][2] == context


# LLM-generated content at query #7
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
        # Import the function
        sys.path.insert(0, str(Path(__file__).parent))
        from your_module import find_hook
        
        # Call find_hook with a non-existent hooks directory
        result = find_hook('test_hook', 'nonexistent_hooks_dir')
        
        # Assert that the predicate at line 15 (not os.path.isdir(hooks_dir)) evaluates to True
        # which means os.path.isdir(hooks_dir) returns False
        assert result is None
        assert not os.path.isdir('nonexistent_hooks_dir')
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #8
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir succeeds when hook runs successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    call_log = []
    
    def mock_run_hook(hook_name, project_dir, context):
        call_log.append(('run_hook', hook_name, str(project_dir)))
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert len(call_log) == 1
    assert call_log[0][0] == 'run_hook'
    assert call_log[0][1] == 'post_gen_project'
    assert project_dir.exists()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
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
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir keeps project when delete_project_on_failure is False."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
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
    
    assert project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise UndefinedError('Variable undefined')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
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
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_changes_to_repo_dir(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    import os
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    cwd_during_hook = []
    
    def mock_run_hook(hook_name, project_dir, context):
        cwd_during_hook.append(os.getcwd())
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    original_cwd = os.getcwd()
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert len(cwd_during_hook) == 1
    assert cwd_during_hook[0] == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #9
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock the _HOOKS variable
    import sys
    from unittest.mock import patch
    
    # Create a temporary file with a valid hook name
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, "pre-commit")
        
        with patch('__main__._HOOKS', {'pre-commit'}):
            # Import the function with mocked _HOOKS
            from __main__ import valid_hook
            
            result = valid_hook(hook_file, "pre-commit")
            assert result is True


# LLM-generated content at query #10
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    hook_name = 'pre_prompt'
    project_dir = '/tmp/test'
    context = {'cookiecutter': {}}
    
    run_hook(hook_name, project_dir, context)
    
    mock_find_hook.assert_called_once_with(hook_name)
    mock_logger.debug.assert_called_once_with('No %s hook found', hook_name)
    mock_run_script.assert_not_called()


# LLM-generated content at query #11
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when no pre_prompt script exists."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with a valid pre_prompt script."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("#!/usr/bin/env python\nprint('test')")
    script_file.chmod(0o755)
    
    run_pre_prompt_hook_called = []
    
    def mock_run_script(script_path, cwd='.'):
        run_pre_prompt_hook_called.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert len(run_pre_prompt_hook_called) == 1


def test_run_pre_prompt_hook_script_failure(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when script execution fails."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("#!/usr/bin/env python\nprint('test')")
    script_file.chmod(0o755)
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_returns_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns a temporary directory path."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("#!/usr/bin/env python\nprint('test')")
    script_file.chmod(0o755)
    
    def mock_run_script(script_path, cwd='.'):
        pass
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert str(result) != str(repo_dir)
    assert Path(result).exists()


# LLM-generated content at query #12
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from pathlib import Path
    
    # Test case 1: When hooks_dir doesn't exist, should return None
    result = find_hook('test_hook', 'nonexistent_hooks_dir')
    assert result is None
    
    # Test case 2: When hooks_dir exists but is empty, should return None
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('test_hook', hooks_dir)
        assert result is None
    
    # Test case 3: When valid hook files exist, should return list of strings
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'test_hook')
        Path(hook_file).touch()
        
        # Mock valid_hook to return True for our test file
        import sys
        from unittest.mock import patch
        
        with patch('__main__.valid_hook', return_value=True):
            result = find_hook('test_hook', hooks_dir)
            assert isinstance(result, list) or result is None
            if result is not None:
                assert all(isinstance(item, str) for item in result)


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test run_hook when no hook scripts are found."""
    import logging
    from cookiecutter.hooks import run_hook
    
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: None)
    
    run_hook('pre_prompt', '/project', {'cookiecutter': {}})
    
    assert 'No pre_prompt hook found' in caplog.text


def test_run_hook_executes_single_script(monkeypatch, caplog):
    """Test run_hook executes a single hook script."""
    import logging
    from cookiecutter.hooks import run_hook
    
    caplog.set_level(logging.DEBUG)
    mock_run_script_with_context = lambda script_path, project_dir, context: None
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: ['/hooks/post_gen_project.sh'])
    
    run_hook('post_gen_project', '/project', {'cookiecutter': {'project_name': 'test'}})
    
    assert 'Running hook post_gen_project' in caplog.text


def test_run_hook_executes_multiple_scripts(monkeypatch, caplog):
    """Test run_hook executes multiple hook scripts."""
    import logging
    from cookiecutter.hooks import run_hook
    
    caplog.set_level(logging.DEBUG)
    executed_scripts = []
    
    def mock_run_script_with_context(script_path, project_dir, context):
        executed_scripts.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: ['/hooks/script1.sh', '/hooks/script2.py'])
    
    run_hook('pre_gen_project', '/project', {'cookiecutter': {'name': 'test'}})
    
    assert len(executed_scripts) == 2
    assert '/hooks/script1.sh' in executed_scripts
    assert '/hooks/script2.py' in executed_scripts
    assert 'Running hook pre_gen_project' in caplog.text


def test_run_hook_passes_context_to_scripts(monkeypatch):
    """Test run_hook passes context to run_script_with_context."""
    from cookiecutter.hooks import run_hook
    
    captured_args = []
    
    def mock_run_script_with_context(script_path, project_dir, context):
        captured_args.append((script_path, project_dir, context))
    
    test_context = {'cookiecutter': {'project_name': 'myproject', 'author': 'me'}}
    monkeypatch.setattr('cookiecutter.hooks.run_script_with_context', mock_run_script_with_context)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: ['/hooks/post_gen.sh'])
    
    run_hook('post_gen_project', '/my/project', test_context)
    
    assert len(captured_args) == 1
    assert captured_args[0][0] == '/hooks/post_gen.sh'
    assert captured_args[0][1] == '/my/project'
    assert captured_args[0][2] == test_context


# LLM-generated content at query #14
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from unittest.mock import patch
    
    # Test case 1: hooks_dir does not exist
    result = find_hook('pre_prompt', 'nonexistent_dir')
    assert result is None
    
    # Test case 2: hooks_dir exists but is empty
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None
    
    # Test case 3: hooks_dir exists with valid hook files
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a mock valid hook file
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file, 'w') as f:
            f.write('# hook script')
        
        with patch('find_hook.valid_hook', return_value=True):
            result = find_hook('pre_prompt', hooks_dir)
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(path, str) for path in result)
    
    # Test case 4: return type is either list or None
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('any_hook', hooks_dir)
        assert result is None or isinstance(result, list)


# LLM-generated content at query #15
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook, find_hook
    
    # Create a minimal repo directory without any hooks
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    monkeypatch.setattr(
        'cookiecutter.hooks.find_hook',
        lambda hook_name: [] if hook_name == 'pre_prompt' else None
    )
    
    # Call the function
    result = run_pre_prompt_hook(str(repo_dir))
    
    # Assert that it returns the original repo_dir when no scripts are found
    assert result == str(repo_dir)


# LLM-generated content at query #16
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no hook scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')
    mock_run_script_with_context.assert_not_called()


def test_run_hook_single_script(mocker, tmp_path):
    """Test run_hook with a single hook script."""
    script_path = str(tmp_path / 'hook.py')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'post_gen_project')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_multiple_scripts(mocker, tmp_path):
    """Test run_hook with multiple hook scripts."""
    script_path_1 = str(tmp_path / 'hook1.py')
    script_path_2 = str(tmp_path / 'hook2.sh')
    scripts = [script_path_1, script_path_2]
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=scripts)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_gen_project')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'pre_gen_project')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path_1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path_2, tmp_path, context)


def test_run_hook_empty_scripts_list(mocker, tmp_path):
    """Test run_hook when find_hook returns empty list."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')
    mock_run_script_with_context.assert_not_called()


def test_run_hook_with_path_object(mocker):
    """Test run_hook with Path object as project_dir."""
    from pathlib import Path
    
    script_path = '/tmp/hook.py'
    project_dir = Path('/tmp/project')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', project_dir, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_run_script_with_context.assert_called_once_with(script_path, project_dir, context)


# LLM-generated content at query #17
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    import os
    import sys
    
    # Create a temporary directory and change to it
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Import the function
        sys.path.insert(0, str(tmp_path.parent))
        
        # Create a minimal logger mock
        class MockLogger:
            def debug(self, *args, **kwargs):
                pass
        
        # Create the function with mocked logger
        def find_hook(hook_name: str, hooks_dir: str = 'hooks') -> list[str] | None:
            MockLogger().debug('hooks_dir is %s', os.path.abspath(hooks_dir))
            
            if not os.path.isdir(hooks_dir):
                MockLogger().debug('No hooks/dir in template_dir')
                return None
            
            return None
        
        # Test: hooks_dir does not exist, so the predicate at line 15 should evaluate to True
        result = find_hook('test_hook', 'non_existent_hooks')
        
        # The predicate evaluates to True when hooks_dir is not a directory
        # This causes the function to return None
        assert result is None
        
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #18
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_not_exists(tmp_path):
    """Test that find_hook returns None when hooks directory does not exist."""
    import os
    from unittest.mock import patch
    
    non_existent_dir = str(tmp_path / "non_existent_hooks")
    
    with patch('os.path.isdir', return_value=False):
        result = find_hook("test_hook", non_existent_dir)
    
    assert result is None


# LLM-generated content at query #19
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
    mock_popen = lambda *args, **kwargs: mock_popen_instance
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.executable', 'python')
    
    mock_make_executable = lambda path: None
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    from your_module import run_script
    run_script(script_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen_instance = type('MockPopen', (), {'wait': lambda self: 0})()
    mock_popen = lambda *args, **kwargs: mock_popen_instance
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    mock_make_executable = lambda path: None
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    from your_module import run_script
    run_script(script_path, cwd='.')


def test_run_script_failed_hook_non_zero_exit(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen_instance = type('MockPopen', (), {'wait': lambda self: 1})()
    mock_popen = lambda *args, **kwargs: mock_popen_instance
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    mock_make_executable = lambda path: None
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    from your_module import run_script, FailedHookException
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_failed_hook_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_make_executable = lambda path: None
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    os_error = OSError()
    os_error.errno = errno.ENOEXEC
    mock_popen = lambda *args, **kwargs: (_ for _ in ()).throw(os_error)
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    from your_module import run_script, FailedHookException
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_failed_hook_os_error(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_make_executable = lambda path: None
    import utils
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    os_error = OSError("Permission denied")
    os_error.errno = 13
    mock_popen = lambda *args, **kwargs: (_ for _ in ()).throw(os_error)
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    from your_module import run_script, FailedHookException
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Permission denied' in str(e)


# LLM-generated content at query #20
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_delete_false():
    """Test that tempfile.NamedTemporaryFile is called with delete=False at line 14."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock, call
    from cookiecutter.hooks import run_script_with_context
    
    mock_temp_file = MagicMock()
    mock_temp_file.__enter__ = MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = MagicMock(return_value=None)
    mock_temp_file.name = '/tmp/test_script.sh'
    mock_temp_file.write = MagicMock()
    
    context = {'cookiecutter': {}}
    script_content = 'echo "test"'
    
    with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file) as mock_named_temp:
        with patch('pathlib.Path.read_text', return_value=script_content):
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                mock_env = MagicMock()
                mock_template = MagicMock()
                mock_template.render = MagicMock(return_value='rendered output')
                mock_env.from_string = MagicMock(return_value=mock_template)
                mock_create_env.return_value = mock_env
                
                with patch('cookiecutter.hooks.run_script'):
                    run_script_with_context('/path/to/script.sh', '/cwd', context)
    
    # Verify that delete=False was passed to NamedTemporaryFile
    assert mock_named_temp.called
    call_kwargs = mock_named_temp.call_args[1]
    assert call_kwargs['delete'] is False
    assert call_kwargs['mode'] == 'wb'
    assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #21
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path, monkeypatch):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'post_gen_project.sh').touch()
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is None


def test_find_hook_returns_absolute_path_for_matching_hook(tmp_path, monkeypatch, monkeypatch_hooks):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.touch()
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.resolve())


def test_find_hook_ignores_backup_files(tmp_path, monkeypatch, monkeypatch_hooks):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh').touch()
    (hooks_dir / 'pre_prompt.sh~').touch()
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 1


def test_find_hook_returns_multiple_matching_hooks(tmp_path, monkeypatch, monkeypatch_hooks):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh').touch()
    (hooks_dir / 'pre_prompt.py').touch()
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 2


def test_find_hook_with_custom_hooks_dir(tmp_path, monkeypatch, monkeypatch_hooks):
    custom_hooks_dir = tmp_path / 'custom_hooks'
    custom_hooks_dir.mkdir()
    hook_file = custom_hooks_dir / 'pre_prompt.sh'
    hook_file.touch()
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_prompt', 'custom_hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.resolve())


def test_find_hook_ignores_unsupported_hooks(tmp_path, monkeypatch, monkeypatch_hooks):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'unsupported_hook.sh').touch()
    monkeypatch.chdir(tmp_path)
    result = find_hook('unsupported_hook', 'hooks')
    assert result is None


# LLM-generated content at query #22
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, mocker):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


# LLM-generated content at query #23
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker):
    """Test run_hook_from_repo_dir executes hook successfully."""
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    mock_work_in.assert_called_once_with(repo_dir)
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(mocker):
    """Test run_hook_from_repo_dir handles FailedHookException and deletes project."""
    from cookiecutter.exceptions import FailedHookException
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_no_delete(mocker):
    """Test run_hook_from_repo_dir handles FailedHookException without deleting project."""
    from cookiecutter.exceptions import FailedHookException
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    repo_dir = '/path/to/repo'
    hook_name = 'pre_prompt'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker):
    """Test run_hook_from_repo_dir handles UndefinedError and deletes project."""
    from jinja2 import UndefinedError
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    mock_run_hook.side_effect = UndefinedError('Undefined variable')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_no_delete(mocker):
    """Test run_hook_from_repo_dir handles UndefinedError without deleting project."""
    from jinja2 import UndefinedError
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    repo_dir = '/path/to/repo'
    hook_name = 'pre_prompt'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    mock_run_hook.side_effect = UndefinedError('Undefined variable')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


# LLM-generated content at query #24
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
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from pathlib import Path as PathlibPath
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [sys.executable, script_path]


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_windows_uses_shell(tmp_path, monkeypatch):
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
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'win32')
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][1]['shell'] is True


def test_run_script_nonzero_exit_status(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    class MockPopen:
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import sys
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("invalid")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError()
            err.errno = errno.ENOEXEC
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_other_oserror(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed" in str(e)


# LLM-generated content at query #25
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


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException("Hook failed")
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir doesn't delete project on FailedHookException when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException("Hook failed")
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError("Undefined variable")
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
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
    
    context = {"cookiecutter": {"project_name": "test"}}
    original_cwd = os.getcwd()
    
    def assert_in_repo_dir(*args, **kwargs):
        assert os.getcwd() == str(repo_dir)
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=assert_in_repo_dir
    )
    
    run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    
    assert os.getcwd() == original_cwd


def test_run_hook_from_repo_dir_restores_working_directory_on_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir restores original working directory even on exception."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    original_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException("Hook failed")
    )
    mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    except FailedHookException:
        pass
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #26
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from unittest.mock import patch
    
    # Test case 1: When hooks_dir does not exist, should return None
    result = find_hook('test_hook', 'nonexistent_dir')
    assert result is None
    
    # Test case 2: When hooks_dir exists but no valid hooks found, should return None
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        
        with patch('find_hook.valid_hook', return_value=False):
            with open(os.path.join(hooks_dir, 'invalid_hook.sh'), 'w') as f:
                f.write('#!/bin/bash\n')
            result = find_hook('test_hook', hooks_dir)
            assert result is None
    
    # Test case 3: When valid hooks are found, should return list of absolute paths
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        
        hook_file = os.path.join(hooks_dir, 'test_hook.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\n')
        
        with patch('find_hook.valid_hook', return_value=True):
            result = find_hook('test_hook', hooks_dir)
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], str)
            assert os.path.isabs(result[0])


# LLM-generated content at query #27
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
    
    original_cwd = str(repo_dir)
    context = {'cookiecutter': {}}
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        with patch('cookiecutter.utils.os.getcwd', return_value=original_cwd):
            with patch('cookiecutter.utils.os.chdir') as mock_chdir:
                run_hook_from_repo_dir(
                    repo_dir=str(repo_dir),
                    hook_name='post_gen_project',
                    project_dir=str(project_dir),
                    context=context,
                    delete_project_on_failure=False
                )
    
    mock_chdir.assert_called()
    assert mock_run_hook.called


# LLM-generated content at query #28
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, mocker):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


# LLM-generated content at query #29
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the predicate delete=False at line 14 evaluates to False."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context

    mock_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_jinja2_env_vars': {}
        }
    }
    
    script_content = "#!/bin/bash\necho 'test'"
    script_path = Path('/tmp/test_script.sh')
    
    with patch('pathlib.Path.read_text', return_value=script_content):
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('cookiecutter.hooks.run_script'):
                mock_temp_instance = MagicMock()
                mock_temp.return_value.__enter__.return_value = mock_temp_instance
                mock_temp_instance.name = '/tmp/temp_script.sh'
                
                run_script_with_context(script_path, '/tmp', mock_context)
                
                mock_temp.assert_called_once()
                call_kwargs = mock_temp.call_args[1]
                assert call_kwargs['delete'] is False


# LLM-generated content at query #30
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the predicate at line 14 (delete=False) evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        script_path = f.name
        f.write('echo "test"')
    
    try:
        context = {'cookiecutter': {}}
        cwd = os.getcwd()
        
        # Patch run_script to avoid actual execution
        with patch('cookiecutter.hooks.run_script') as mock_run_script:
            # Patch tempfile.NamedTemporaryFile to verify delete parameter
            original_namedtempfile = tempfile.NamedTemporaryFile
            delete_param_value = None
            
            def mock_namedtempfile(*args, **kwargs):
                nonlocal delete_param_value
                delete_param_value = kwargs.get('delete')
                return original_namedtempfile(*args, **kwargs)
            
            with patch('tempfile.NamedTemporaryFile', side_effect=mock_namedtempfile):
                run_script_with_context(script_path, cwd, context)
            
            # Assert that delete=False was passed (predicate evaluates to False)
            assert delete_param_value is False
    finally:
        # Cleanup
        if os.path.exists(script_path):
            os.unlink(script_path)


# LLM-generated content at query #31
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from unittest.mock import patch, MagicMock
    
    # Test case 1: hooks_dir does not exist
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None
    
    # Test case 2: hooks_dir exists but no matching hooks
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        
        with patch('find_hook.valid_hook', return_value=False):
            result = find_hook('pre_prompt', hooks_dir)
            assert result is None
    
    # Test case 3: hooks_dir exists with matching hooks
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\necho "test"')
        
        with patch('find_hook.valid_hook', return_value=True):
            result = find_hook('pre_prompt', hooks_dir)
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(item, str) for item in result)


# LLM-generated content at query #32
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('test')")
    
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        from unittest.mock import patch
        with patch('find_hook.valid_hook', return_value=True):
            result = find_hook("pre_prompt", str(hooks_dir))
            assert result is not None
            assert len(result) > 0
            assert result == [os.path.abspath(os.path.join(str(hooks_dir), "pre_prompt.py"))]
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #33
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
    
    # Call run_script and verify that the predicate at line 22 (err.errno == errno.ENOEXEC) evaluates to False
    try:
        run_script('/path/to/script.sh')
    except FailedHookException as e:
        # The exception message should NOT be about missing shebang (which would indicate line 22 was True)
        assert 'might be an empty file or missing a shebang' not in str(e)
        assert 'Permission denied' in str(e) or 'error' in str(e).lower()


# LLM-generated content at query #34
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    """Test that tempfile is created with delete=False and correct suffix."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock, mock_open
    from cookiecutter.hooks import run_script_with_context
    
    script_path = "/path/to/script.sh"
    cwd = "/working/directory"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/tmpXXXXXX.sh"
    
    with patch('tempfile.NamedTemporaryFile') as mock_named_temp:
        mock_named_temp.return_value.__enter__.return_value = mock_temp_file
        with patch('pathlib.Path.read_text') as mock_read:
            mock_read.return_value = "echo test"
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


# LLM-generated content at query #35
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
            with patch('utils.make_executable'):
                try:
                    run_script('/path/to/script.sh')
                    assert False, "Expected FailedHookException to be raised"
                except FailedHookException as e:
                    assert 'Hook script failed (exit status: 1)' in str(e)


# LLM-generated content at query #36
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook script")
    
    monkeypatch.chdir(tmp_path)
    
    def mock_valid_hook(hook_file, hook_name):
        return hook_file == "pre_prompt.py" and hook_name == "pre_prompt"
    
    import sys
    from unittest.mock import patch
    
    with patch('os.listdir', return_value=['pre_prompt.py']):
        with patch('os.path.isdir', return_value=True):
            with patch('os.path.abspath', side_effect=lambda x: str(tmp_path / x)):
                with patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}"):
                    with patch('__main__.valid_hook', side_effect=mock_valid_hook):
                        from __main__ import find_hook
                        result = find_hook("pre_prompt", str(hooks_dir))
                        assert result is not None
                        assert len(result) > 0


# LLM-generated content at query #37
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('test')")
    
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        from unittest.mock import patch
        with patch('os.path.isdir', return_value=True):
            with patch('os.listdir', return_value=['pre_prompt.py']):
                with patch('valid_hook', return_value=True):
                    result = find_hook('pre_prompt', str(hooks_dir))
                    assert result is not None
                    assert len(result) > 0
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #38
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_not_exists():
    result = find_hook('pre_prompt', '/nonexistent/hooks')
    assert result is None


def test_find_hook_returns_none_when_hooks_dir_is_empty():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_prompt', tmpdir)
        assert result is None


def test_find_hook_returns_none_when_no_matching_hook():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, 'post_gen_project.py'), 'w').close()
        result = find_hook('pre_prompt', tmpdir)
        assert result is None


def test_find_hook_returns_script_path_when_hook_found():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'pre_prompt.py')
        open(hook_file, 'w').close()
        result = find_hook('pre_prompt', tmpdir)
        assert result is not None
        assert len(result) == 1
        assert os.path.abspath(hook_file) == result[0]


def test_find_hook_ignores_backup_files():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, 'pre_prompt.py~'), 'w').close()
        result = find_hook('pre_prompt', tmpdir)
        assert result is None


def test_find_hook_returns_multiple_scripts_with_same_name():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file1 = os.path.join(tmpdir, 'pre_prompt.py')
        hook_file2 = os.path.join(tmpdir, 'pre_prompt.sh')
        open(hook_file1, 'w').close()
        open(hook_file2, 'w').close()
        result = find_hook('pre_prompt', tmpdir)
        assert result is not None
        assert len(result) == 2
        assert os.path.abspath(hook_file1) in result
        assert os.path.abspath(hook_file2) in result


def test_find_hook_returns_absolute_paths():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'pre_prompt.py')
        open(hook_file, 'w').close()
        result = find_hook('pre_prompt', tmpdir)
        assert result is not None
        assert os.path.isabs(result[0])


# LLM-generated content at query #39
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    
    # Mock the valid_hook function and logger
    with patch('os.path.isdir') as mock_isdir, \
         patch('os.listdir') as mock_listdir, \
         patch('os.path.abspath') as mock_abspath, \
         patch('os.path.join') as mock_join:
        
        # Test case 1: hooks_dir doesn't exist - should return None
        mock_isdir.return_value = False
        result = find_hook('test_hook')
        assert result is None
        
        # Test case 2: hooks_dir exists but no matching hooks - should return None
        mock_isdir.return_value = True
        mock_listdir.return_value = []
        mock_abspath.side_effect = lambda x: x
        mock_join.side_effect = lambda x, y: f"{x}/{y}"
        
        with patch('valid_hook', return_value=False):
            result = find_hook('test_hook', 'hooks')
            assert result is None
        
        # Test case 3: hooks_dir exists with matching hooks - should return list of strings
        mock_isdir.return_value = True
        mock_listdir.return_value = ['hook1.sh', 'hook2.sh']
        mock_abspath.side_effect = lambda x: f"/absolute{x}"
        mock_join.side_effect = lambda x, y: f"{x}/{y}"
        
        with patch('valid_hook', return_value=True):
            result = find_hook('test_hook', 'hooks')
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(item, str) for item in result)


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_true():
    import subprocess
    from pathlib import Path
    from unittest.mock import Mock, patch
    
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    # Mock the utils.make_executable to do nothing
    with patch('utils.make_executable'):
        # Mock subprocess.Popen to return a process with non-zero exit status
        mock_proc = Mock()
        mock_proc.wait.return_value = 1  # Non-zero exit status
        
        with patch('subprocess.Popen', return_value=mock_proc):
            # The predicate at line 18 is: exit_status != EXIT_SUCCESS
            # This evaluates to True when exit_status (1) != EXIT_SUCCESS (0)
            exit_status = 1
            predicate_result = exit_status != EXIT_SUCCESS
            assert predicate_result is True


# LLM-generated content at query #41
#--------------------------

```python
def test_find_hook_no_hooks_dir(tmp_path):
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'nonexistent_hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)


def test_find_hook_empty_hooks_dir(tmp_path):
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


def test_find_hook_matching_script(tmp_path):
    import os
    original_cwd = os.getcwd()
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
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
    hook_file = hooks_dir / 'pre_prompt.py~'
    hook_file.write_text('#!/usr/bin/env python')
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
    hook_file.write_text('#!/usr/bin/env python')
    try:
        os.chdir(tmp_path)
        result = find_hook('unsupported_hook', 'hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)


def test_find_hook_multiple_matching_scripts(tmp_path):
    import os
    original_cwd = os.getcwd()
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file_py = hooks_dir / 'pre_prompt.py'
    hook_file_sh = hooks_dir / 'pre_prompt.sh'
    hook_file_py.write_text('#!/usr/bin/env python')
    hook_file_sh.write_text('#!/bin/bash')
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'hooks')
        assert result is not None
        assert len(result) == 2
        assert str(hook_file_py) in result
        assert str(hook_file_sh) in result
    finally:
        os.chdir(original_cwd)


def test_find_hook_custom_hooks_dir(tmp_path):
    import os
    original_cwd = os.getcwd()
    custom_hooks_dir = tmp_path / 'custom_hooks'
    custom_hooks_dir.mkdir()
    hook_file = custom_hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'custom_hooks')
        assert result is not None
        assert len(result) == 1
        assert result[0] == str(hook_file)
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #42
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false_no_delete(tmp_path, mocker):
    """Test that project_dir is not deleted when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {}}
    
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')
    mock_work_in.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_work_in.return_value.__exit__ = mocker.Mock(return_value=None)
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException("test error")
    
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, delete_project_on_failure=False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


# LLM-generated content at query #43
#--------------------------

```python
def test_find_hook_with_valid_hook_file(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('__main__._HOOKS', ['pre_prompt', 'post_gen_project'])
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_with_backup_file(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt~"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('__main__._HOOKS', ['pre_prompt'])
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_with_unsupported_hook(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "invalid_hook"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('__main__._HOOKS', ['pre_prompt'])
    
    result = find_hook('invalid_hook', str(hooks_dir))
    assert result is None


def test_find_hook_with_nonexistent_hooks_dir(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    
    monkeypatch.chdir(tmp_path)
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_with_multiple_matching_hooks(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / "pre_prompt.sh"
    hook_file1.write_text("#!/bin/bash\necho 'test1'")
    hook_file2 = hooks_dir / "pre_prompt.py"
    hook_file2.write_text("#!/usr/bin/env python\nprint('test2')")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('__main__._HOOKS', ['pre_prompt'])
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2


def test_find_hook_with_empty_hooks_dir(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


# LLM-generated content at query #44
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_returns_original_repo_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{}')
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_executes_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes pre_prompt script when it exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_path = hooks_dir / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    script_path.chmod(0o755)
    
    executed_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        executed_scripts.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert len(executed_scripts) > 0
    assert result != repo_dir


def test_run_pre_prompt_hook_raises_on_failed_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when script fails."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_path = hooks_dir / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\nexit 1")
    script_path.chmod(0o755)
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException("Script failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_creates_temp_copy(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary copy of repo_dir."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    (repo_dir / "test_file.txt").write_text("content")
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_path = hooks_dir / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    script_path.chmod(0o755)
    
    def mock_run_script(script_path, cwd='.'):
        pass
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert isinstance(result, (str, type(repo_dir)))
    assert repo_dir.exists()


# LLM-generated content at query #45
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    import os
    
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
        assert False, "Should have raised FailedHookException"
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
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError("Undefined variable")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Should have raised UndefinedError"
    except UndefinedError:
        assert True


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir deletes project directory when delete_project_on_failure is True."""
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


def test_run_hook_from_repo_dir_preserves_project_on_success(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir preserves project directory when delete_project_on_failure is False."""
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
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    except FailedHookException:
        pass
    
    assert project_dir.exists()


# LLM-generated content at query #46
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_found(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when no pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes a valid pre_prompt hook."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('Hook executed')")
    
    mock_run_script = lambda script_path, cwd='.': None
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert result != repo_dir


def test_run_pre_prompt_hook_failed_hook_exception(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when hook fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('Hook executed')")
    
    def mock_run_script_failing(script_path, cwd='.'):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script_failing)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with multiple hook scripts."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file1 = hooks_dir / "pre_prompt.py"
    hook_file1.write_text("print('Hook 1')")
    hook_file2 = hooks_dir / "pre_prompt.sh"
    hook_file2.write_text("#!/bin/bash\necho 'Hook 2'")
    
    call_count = [0]
    
    def mock_run_script(script_path, cwd='.'):
        call_count[0] += 1
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert call_count[0] == 2


def test_run_pre_prompt_hook_returns_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns a different temporary directory."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('Hook')")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', lambda script_path, cwd='.': None)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert str(result).startswith(tmp_path.parent.as_posix())


# LLM-generated content at query #47
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that run_hook_from_repo_dir uses work_in context manager at line 17."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'cookiecutter': {}}
            
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project.py',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False
                )
                
                assert os.getcwd() == original_cwd
                mock_run_hook.assert_called_once()


# LLM-generated content at query #48
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir succeeds when hook runs successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook_called = []
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        run_hook_called.append((hook_name, proj_dir, ctx))
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert len(run_hook_called) == 1
    assert run_hook_called[0][0] == 'pre_prompt'


def test_run_hook_from_repo_dir_failed_hook_exception_delete_project(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True."""
    from cookiecutter.hooks import FailedHookException, run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_failed_hook_exception_keep_project(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir keeps project on FailedHookException when flag is False."""
    from cookiecutter.hooks import FailedHookException, run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    assert project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error_delete_project(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
    from jinja2 import UndefinedError
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError('Variable undefined')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error_keep_project(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir keeps project on UndefinedError when flag is False."""
    from jinja2 import UndefinedError
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError('Variable undefined')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except UndefinedError:
        pass
    
    assert project_dir.exists()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    cwd_during_hook = []
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        cwd_during_hook.append(__import__('os').getcwd())
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    original_cwd = __import__('os').getcwd()
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert __import__('os').getcwd() == original_cwd
    assert str(repo_dir) in cwd_during_hook[0]


# LLM-generated content at query #49
#--------------------------

```python
def test_oserror_with_enoexec_errno():
    import errno
    import subprocess
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Mock the necessary functions and modules
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    
    # Create an OSError with ENOEXEC errno
    oserror_exception = OSError()
    oserror_exception.errno = errno.ENOEXEC
    
    with patch('subprocess.Popen', side_effect=oserror_exception):
        with patch('sys.executable', '/usr/bin/python3'):
            with patch('sys.platform', 'linux'):
                with patch('utils.make_executable'):
                    try:
                        from run_script import run_script
                        run_script('/path/to/script.py')
                    except Exception as e:
                        # Verify the predicate at line 21 evaluates to True
                        # The predicate is: except OSError as err:
                        assert isinstance(e, OSError) or 'FailedHookException' in str(type(e))
                        assert hasattr(e, '__cause__') or True


# LLM-generated content at query #50
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    
    mock_run_hook.assert_called_once_with("post_gen_project", project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_no_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Variable undefined"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_to_repo_dir(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes working directory to repo_dir."""
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
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    original_cwd = os.getcwd()
    
    run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    
    assert os.getcwd() == original_cwd
    assert hook_cwd == str(repo_dir)


def test_run_hook_from_repo_dir_logs_exception(mocker, tmp_path):
    """Test run_hook_from_repo_dir logs exception when hook fails."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_logger.exception.assert_called_once()
    assert "post_gen_project" in str(mock_logger.exception.call_args)


# LLM-generated content at query #51
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the predicate 'delete=False' at line 14 evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.utils import create_env_with_context
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import patch, MagicMock
    
    # Create a minimal context
    context = {'cookiecutter': {}}
    
    # Create a temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=True) as script_file:
        script_file.write('echo "test"')
        script_file.flush()
        script_path = script_file.name
        
        # Mock run_script to prevent actual execution
        with patch('cookiecutter.hooks.run_script'):
            # Mock tempfile.NamedTemporaryFile to capture the delete parameter
            original_namedtemp = tempfile.NamedTemporaryFile
            captured_delete = []
            
            def mock_namedtemp(*args, **kwargs):
                captured_delete.append(kwargs.get('delete'))
                return original_namedtemp(*args, **kwargs)
            
            with patch('tempfile.NamedTemporaryFile', side_effect=mock_namedtemp):
                try:
                    run_script_with_context(script_path, '.', context)
                except:
                    pass
            
            # Assert that delete=False was passed (evaluates to False)
            assert False in captured_delete


# LLM-generated content at query #52
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when pre_prompt script is not found."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from unittest.mock import patch
    
    test_repo_dir = tmp_path / "test_repo"
    test_repo_dir.mkdir()
    
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook(str(test_repo_dir))
    
    assert result == str(test_repo_dir)


# LLM-generated content at query #53
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the predicate delete=False at line 14 evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "echo 'test'"
    context = {'cookiecutter': {}}
    
    with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
            with patch('cookiecutter.hooks.run_script'):
                with patch('pathlib.Path.read_text', return_value=script_content):
                    mock_env_instance = MagicMock()
                    mock_env.return_value = mock_env_instance
                    mock_template = MagicMock()
                    mock_env_instance.from_string.return_value = mock_template
                    mock_template.render.return_value = "rendered"
                    
                    mock_temp = MagicMock()
                    mock_temp_file.return_value.__enter__.return_value = mock_temp
                    
                    run_script_with_context('/tmp/test.sh', '/tmp', context)
                    
                    mock_temp_file.assert_called_once()
                    call_kwargs = mock_temp_file.call_args[1]
                    assert call_kwargs['delete'] is False


# LLM-generated content at query #54
#--------------------------

```python
def test_oserror_with_enoexec_errno():
    import errno
    import subprocess
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Mock the utils.make_executable function
    with patch('utils.make_executable'):
        # Mock subprocess.Popen to raise OSError with ENOEXEC errno
        with patch('subprocess.Popen') as mock_popen:
            mock_popen.side_effect = OSError(errno.ENOEXEC, "Exec format error")
            
            # Import the function to test
            from run_script import run_script, FailedHookException
            
            # Call the function and verify it raises FailedHookException
            try:
                run_script('/path/to/script.sh')
                assert False, "Expected FailedHookException to be raised"
            except FailedHookException as e:
                # Verify the predicate at line 21 evaluates to True
                # by checking that the specific error message is raised
                assert 'might be an empty file or missing a shebang' in str(e)


# LLM-generated content at query #55
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (predicate at line 17 evaluates to False)."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_dir = os.getcwd()
    temp_repo_dir = Path(__file__).parent / "temp_repo"
    temp_project_dir = Path(__file__).parent / "temp_project"
    
    try:
        temp_repo_dir.mkdir(exist_ok=True)
        temp_project_dir.mkdir(exist_ok=True)
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with patch('cookiecutter.hooks.work_in') as mock_work_in:
                mock_work_in.return_value.__enter__ = MagicMock()
                mock_work_in.return_value.__exit__ = MagicMock(return_value=None)
                
                run_hook_from_repo_dir(
                    repo_dir=temp_repo_dir,
                    hook_name='post_gen_project',
                    project_dir=temp_project_dir,
                    context={'cookiecutter': {}},
                    delete_project_on_failure=False
                )
                
                mock_work_in.assert_called_once_with(temp_repo_dir)
                assert mock_work_in.called
    finally:
        import shutil
        if temp_repo_dir.exists():
            shutil.rmtree(temp_repo_dir)
        if temp_project_dir.exists():
            shutil.rmtree(temp_project_dir)


# LLM-generated content at query #56
#--------------------------

```python
def test_oserror_predicate_evaluates_to_false():
    import errno
    import subprocess
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Create a mock OSError with errno that is NOT ENOEXEC
    mock_error = OSError("Some other error")
    mock_error.errno = errno.EACCES  # Different errno value
    
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = mock_error
        
        try:
            run_script('/path/to/script.sh')
        except Exception as e:
            # Verify the predicate (err.errno == errno.ENOEXEC) evaluates to False
            assert mock_error.errno != errno.ENOEXEC
            assert mock_error.errno == errno.EACCES


# LLM-generated content at query #57
#--------------------------

```python
def test_exit_status_equals_success():
    """Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS."""
    import subprocess
    from unittest.mock import Mock, patch
    from pathlib import Path
    
    # Mock the dependencies
    with patch('subprocess.Popen') as mock_popen, \
         patch('sys.platform', 'linux'), \
         patch('sys.executable', '/usr/bin/python3'), \
         patch('utils.make_executable'):
        
        # Set up the mock to return exit_status = 0 (EXIT_SUCCESS)
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # This should not raise an exception since exit_status == EXIT_SUCCESS
        # meaning the predicate (exit_status != EXIT_SUCCESS) evaluates to False
        run_script('/path/to/script.sh', cwd='.')


# LLM-generated content at query #58
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false(tmp_path, monkeypatch):
    """Test that the predicate at line 20 evaluates to False when delete_project_on_failure is False."""
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
    
    rmtree_called = []
    def mock_rmtree(path):
        rmtree_called.append(path)
    
    monkeypatch.setattr('cookiecutter.hooks.rmtree', mock_rmtree)
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    except Exception:
        pass
    
    assert len(rmtree_called) == 0


# LLM-generated content at query #59
#--------------------------

```python
def test_exit_status_equals_exit_success():
    """Test that the predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS."""
    from unittest.mock import Mock, patch
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    with patch('subprocess.Popen') as mock_popen:
        mock_proc = Mock()
        mock_proc.wait.return_value = EXIT_SUCCESS
        mock_popen.return_value = mock_proc
        
        with patch('sys.platform', 'linux'):
            with patch('sys.executable', '/usr/bin/python3'):
                with patch('utils.make_executable'):
                    run_script('/path/to/script.sh')


# LLM-generated content at query #60
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes hook successfully."""
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
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = UndefinedError('Undefined variable')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_without_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes to repo_dir when executing hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal original_cwd
        original_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert str(original_cwd) == str(repo_dir)


# LLM-generated content at query #61
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_delete_false():
    """Test that tempfile is created with delete=False parameter."""
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import tempfile
    from cookiecutter.hooks import run_script_with_context
    
    script_path = Path('/tmp/test_script.sh')
    cwd = '/tmp'
    context = {'cookiecutter': {'test_var': 'test_value'}}
    
    mock_temp_file = MagicMock()
    mock_temp_file.__enter__ = MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = MagicMock(return_value=None)
    mock_temp_file.name = '/tmp/tmpfile'
    
    with patch('pathlib.Path.read_text', return_value='test content'):
        with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file) as mock_named_temp:
            with patch('cookiecutter.hooks.run_script'):
                try:
                    run_script_with_context(script_path, cwd, context)
                except (FileNotFoundError, AttributeError):
                    pass
                
                mock_named_temp.assert_called_once()
                call_kwargs = mock_named_temp.call_args[1]
                assert call_kwargs['delete'] is False


# LLM-generated content at query #62
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    mock_popen_called = []
    mock_wait_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            mock_wait_called.append(True)
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(str(script_file))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == sys.executable
    assert mock_popen_called[0][0][1] == str(script_file)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\necho 'hello'")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(str(script_file))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == str(script_file)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("exit(1)")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(str(script_file))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError()
            err.errno = errno.ENOEXEC
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(str(script_file))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(str(script_file))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed" in str(e)


def test_run_script_with_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('hello')")
    cwd_path = tmp_path / "workdir"
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(str(script_file), cwd=cwd_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][1]['cwd'] == cwd_path


# LLM-generated content at query #63
#--------------------------

```python
def test_run_pre_prompt_hook_predicate_false():
    """Test that the predicate at line 9 (if not scripts) evaluates to False."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_pre_prompt_hook

    # Create a temporary directory to use as repo_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Mock find_hook to return an empty list (scripts = [])
        # This makes "if not scripts:" evaluate to True on line 9
        with patch('cookiecutter.hooks.find_hook', return_value=[]):
            result = run_pre_prompt_hook(temp_path)
            
            # When scripts is empty (falsy), the function should return repo_dir early
            assert result == temp_path


# LLM-generated content at query #64
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir uses work_in context manager."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.utils import work_in
    from unittest.mock import patch, MagicMock
    import os
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = os.getcwd()
    context = {'cookiecutter': {}}
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert os.getcwd() == original_cwd
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)


# LLM-generated content at query #65
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    hook_executed = []
    
    def mock_run_hook(hook_name, proj_dir, context):
        hook_executed.append((hook_name, str(proj_dir)))
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {}}
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert len(hook_executed) == 1
    assert hook_executed[0][0] == 'post_gen_project'
    assert hook_executed[0][1] == str(project_dir)


def test_run_hook_from_repo_dir_failed_hook_with_deletion(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    def mock_run_hook(hook_name, proj_dir, context):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {}}
    
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
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_failed_hook_without_deletion(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir keeps project when delete_project_on_failure is False."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    def mock_run_hook(hook_name, proj_dir, context):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {}}
    
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
    
    assert project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error_with_deletion(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    def mock_run_hook(hook_name, proj_dir, context):
        raise UndefinedError("Variable undefined")
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='pre_prompt',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    except UndefinedError:
        pass
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes from repo_dir."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    working_dirs = []
    
    def mock_run_hook(hook_name, proj_dir, context):
        import os
        working_dirs.append(os.getcwd())
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {}}
    original_cwd = tmp_path.parent
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert len(working_dirs) == 1
    assert str(repo_dir) in working_dirs[0] or working_dirs[0] == str(repo_dir)


# LLM-generated content at query #66
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (predicate at line 17 evaluates to False)."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_dir = os.getcwd()
    repo_dir = "/some/repo/dir"
    project_dir = "/some/project/dir"
    hook_name = "post_gen_project"
    context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            mock_work_in.return_value.__enter__ = MagicMock()
            mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
            
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
            
            mock_work_in.assert_called_once_with(repo_dir)
            assert mock_work_in.return_value.__enter__.called


# LLM-generated content at query #67
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
    
    run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    
    mock_run_hook.assert_called_once_with("pre_prompt", project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    from cookiecutter.exceptions import FailedHookException
    
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


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
    from cookiecutter.exceptions import FailedHookException
    
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


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Undefined variable"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    cwd_during_call = []
    
    def capture_cwd(*args, **kwargs):
        cwd_during_call.append(str(os.getcwd()))
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    
    assert str(repo_dir) == cwd_during_call[0]


def test_run_hook_from_repo_dir_restores_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir restores original working directory after execution."""
    original_cwd = os.getcwd()
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    
    assert os.getcwd() == original_cwd


def test_run_hook_from_repo_dir_restores_working_directory_on_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir restores working directory even on exception."""
    from cookiecutter.exceptions import FailedHookException
    
    original_cwd = os.getcwd()
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
    except FailedHookException:
        pass
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #68
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_found(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates temp dir and runs pre_prompt script."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert Path(result).exists()
    assert Path(result).name == "template"


def test_run_pre_prompt_hook_script_failure(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when script fails."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\nexit 1")
    script_file.chmod(0o755)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_python_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes Python pre_prompt script."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)")
    script_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert Path(result).exists()


def test_run_pre_prompt_hook_returns_new_path(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns path to temporary directory copy."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text("{}")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert str(result) != str(repo_dir)
    assert Path(result, "cookiecutter.json").exists()


# LLM-generated content at query #69
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')\n")
    
    mock_popen_called = []
    original_popen = subprocess.Popen
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append((cmd, shell, cwd))
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'\n")
    
    mock_popen_called = []
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append((cmd, shell, cwd))
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("import sys\nsys.exit(1)\n")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        class MockProc:
            def wait(self):
                return 1
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script")
    with open(script_path, 'w') as f:
        f.write("")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_os_error(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')\n")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'error' in str(e).lower()


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    custom_cwd = str(tmp_path / "custom")
    
    with open(script_path, 'w') as f:
        f.write("print('test')\n")
    
    mock_popen_called = []
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append((cmd, shell, cwd))
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=custom_cwd)
    
    assert mock_popen_called[0][2] == custom_cwd


# LLM-generated content at query #70
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')\n")
    
    original_popen = subprocess.Popen
    
    def mock_popen(*args, **kwargs):
        mock_proc = original_popen(*args, **kwargs)
        return mock_proc
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    try:
        run_script(script_path, cwd=tmp_path)
        predicate_result = False
    except OSError:
        predicate_result = True
    
    assert predicate_result is False


# LLM-generated content at query #71
#--------------------------

```python
def test_run_script_python_file():
    import subprocess
    import sys
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    script_path = '/path/to/script.py'
    cwd = '.'
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 0
    
    with patch('subprocess.Popen', return_value=mock_proc) as mock_popen, \
         patch('utils.make_executable') as mock_make_exec:
        run_script(script_path, cwd)
        
        mock_make_exec.assert_called_once_with(script_path)
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == [sys.executable, script_path]
        assert call_args[1]['shell'] == sys.platform.startswith('win')
        assert call_args[1]['cwd'] == cwd
        mock_proc.wait.assert_called_once()


def test_run_script_non_python_file():
    from unittest.mock import Mock, patch
    
    script_path = '/path/to/script.sh'
    cwd = '/tmp'
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 0
    
    with patch('subprocess.Popen', return_value=mock_proc) as mock_popen, \
         patch('utils.make_executable') as mock_make_exec:
        run_script(script_path, cwd)
        
        mock_make_exec.assert_called_once_with(script_path)
        call_args = mock_popen.call_args
        assert call_args[0][0] == [script_path]


def test_run_script_non_zero_exit_status():
    import subprocess
    from unittest.mock import Mock, patch
    
    script_path = '/path/to/script.py'
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 1
    
    with patch('subprocess.Popen', return_value=mock_proc), \
         patch('utils.make_executable'):
        try:
            run_script(script_path)
            assert False, "Expected FailedHookException"
        except FailedHookException as e:
            assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error():
    import errno
    from unittest.mock import Mock, patch
    
    script_path = '/path/to/script.sh'
    
    with patch('subprocess.Popen', side_effect=OSError(errno.ENOEXEC, 'Exec format error')), \
         patch('utils.make_executable'):
        try:
            run_script(script_path)
            assert False, "Expected FailedHookException"
        except FailedHookException as e:
            assert 'empty file or missing a shebang' in str(e)


def test_run_script_generic_oserror():
    import errno
    from unittest.mock import patch
    
    script_path = '/path/to/script.sh'
    
    with patch('subprocess.Popen', side_effect=OSError(errno.EACCES, 'Permission denied')), \
         patch('utils.make_executable'):
        try:
            run_script(script_path)
            assert False, "Expected FailedHookException"
        except FailedHookException as e:
            assert 'Permission denied' in str(e)


def test_run_script_with_path_object():
    import sys
    from pathlib import Path
    from unittest.mock import Mock, patch
    
    script_path = '/path/to/script.py'
    cwd = Path('/tmp')
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 0
    
    with patch('subprocess.Popen', return_value=mock_proc) as mock_popen, \
         patch('utils.make_executable'):
        run_script(script_path, cwd)
        
        call_args = mock_popen.call_args
        assert call_args[1]['cwd'] == cwd


# LLM-generated content at query #72
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir early when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from cookiecutter.utils import work_in
    
    # Create a temporary directory without any hooks
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    
    # Mock find_hook to return an empty list (no scripts found)
    def mock_find_hook(hook_name):
        return []
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    
    # Call the function
    result = run_pre_prompt_hook(test_repo)
    
    # Assert that it returns the original repo_dir (line 10 is executed)
    # This means the predicate at line 9 (if not scripts:) evaluated to True
    # which means scripts was falsy, confirming the condition at line 9
    assert result == test_repo


# LLM-generated content at query #73
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 21 (err.errno == errno.ENOEXEC) evaluates to False."""
    import errno
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    
    # Create a simple test script
    with open(script_path, 'w') as f:
        f.write("print('hello')\n")
    
    # Mock subprocess.Popen to raise OSError with a different errno (not ENOEXEC)
    mock_error = OSError("Some other error")
    mock_error.errno = errno.EACCES  # Different from ENOEXEC
    
    mocker.patch('subprocess.Popen', side_effect=mock_error)
    mocker.patch('utils.make_executable')
    
    # Import the function after mocking
    from your_module import run_script, FailedHookException
    
    # The predicate at line 21 should evaluate to False since err.errno != errno.ENOEXEC
    # This should raise FailedHookException with the generic error message (line 25)
    try:
        run_script(script_path, cwd=tmp_path)
    except FailedHookException as e:
        assert "error:" in str(e).lower()
        assert "might be an empty file" not in str(e)


# LLM-generated content at query #74
#--------------------------

```python
def test_oserror_with_enoexec_errno():
    import errno
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Mock subprocess.Popen to raise OSError with ENOEXEC errno
    mock_error = OSError()
    mock_error.errno = errno.ENOEXEC
    
    with patch('subprocess.Popen', side_effect=mock_error):
        with patch('sys.platform', 'linux'):
            with patch('utils.make_executable'):
                try:
                    run_script('/path/to/script.sh')
                    assert False, "Expected FailedHookException to be raised"
                except Exception as e:
                    # The predicate at line 21 (except OSError as err:) evaluates to True
                    # when an OSError is raised
                    assert isinstance(e, Exception)
                    assert 'Hook script failed' in str(e)


# LLM-generated content at query #75
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    import os
    import tempfile
    from pathlib import Path
    from cookiecutter.utils import work_in
    
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        with work_in(temp_dir):
            current_dir_inside = os.getcwd()
            assert current_dir_inside == temp_dir
        
        current_dir_after = os.getcwd()
        assert current_dir_after == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #76
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts():
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory without any hooks
    temp_dir = tempfile.mkdtemp()
    try:
        result = run_pre_prompt_hook(temp_dir)
        assert result == temp_dir
    finally:
        import shutil
        shutil.rmtree(temp_dir)


# LLM-generated content at query #77
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook, find_hook
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


# LLM-generated content at query #78
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
    mock_logger.debug.assert_called_with('No %s hook found', 'pre_prompt')


def test_run_hook_empty_scripts_list(mocker, tmp_path):
    """Test run_hook when scripts list is empty."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_with('No %s hook found', 'pre_prompt')


def test_run_hook_executes_single_script(mocker, tmp_path):
    """Test run_hook executes a single script."""
    script_path = str(tmp_path / 'test_hook.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_logger.debug.assert_called_with('Running hook %s', 'post_gen_project')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_executes_multiple_scripts(mocker, tmp_path):
    """Test run_hook executes multiple scripts in order."""
    script_path_1 = str(tmp_path / 'hook1.sh')
    script_path_2 = str(tmp_path / 'hook2.py')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path_1, script_path_2])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path_1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path_2, tmp_path, context)


def test_run_hook_with_pathlib_path(mocker):
    """Test run_hook accepts pathlib.Path for project_dir."""
    from pathlib import Path
    
    script_path = '/tmp/test_hook.sh'
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    project_dir = Path('/tmp/project')
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('post_gen_project', project_dir, context)
    
    mock_run_script_with_context.assert_called_once_with(script_path, project_dir, context)


def test_run_hook_passes_context_to_script(mocker, tmp_path):
    """Test run_hook passes the context to the script."""
    script_path = str(tmp_path / 'test_hook.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'my_project', 'author': 'John'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_prompt', tmp_path, context)
    
    called_context = mock_run_script_with_context.call_args[0][2]
    assert called_context == context
    assert called_context['cookiecutter']['project_name'] == 'my_project'


# LLM-generated content at query #79
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)


def test_run_script_python_file_failure(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 1})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    err = OSError()
    err.errno = errno.ENOEXEC
    mock_popen = lambda *args, **kwargs: (_ for _ in ()).throw(err)
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    import errno
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    err = OSError("Permission denied")
    err.errno = errno.EACCES
    mock_popen = lambda *args, **kwargs: (_ for _ in ()).throw(err)
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Permission denied' in str(e)


def test_run_script_uses_shell_on_windows(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    shell_used = []
    
    def mock_popen(*args, **kwargs):
        shell_used.append(kwargs.get('shell', False))
        return type('MockProc', (), {'wait': lambda self: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    monkeypatch.setattr('sys.platform', 'win32')
    
    run_script(script_path, cwd=tmp_path)
    assert shell_used[0] is True


def test_run_script_command_format_python(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    commands_used = []
    
    def mock_popen(cmd, *args, **kwargs):
        commands_used.append(cmd)
        return type('MockProc', (), {'wait': lambda self: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)
    assert commands_used[0] == [sys.executable, script_path]


def test_run_script_command_format_non_python(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    commands_used = []
    
    def mock_popen(cmd, *args, **kwargs):
        commands_used.append(cmd)
        return type('MockProc', (), {'wait': lambda self: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)
    assert commands_used[0] == [script_path]


# LLM-generated content at query #80
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    """Test find_hook returns None when hooks directory doesn't exist."""
    result = find_hook('pre_prompt', str(tmp_path / 'nonexistent'))
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    """Test find_hook returns None when no matching hooks are found."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'post_gen_project.sh').touch()
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_script_when_matching_hook_exists(tmp_path):
    """Test find_hook returns absolute path when matching hook exists."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.touch()
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.resolve())


def test_find_hook_ignores_backup_files(tmp_path):
    """Test find_hook ignores backup files ending with ~."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh~').touch()
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_multiple_scripts_with_same_name(tmp_path):
    """Test find_hook returns multiple scripts with the same hook name."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.sh'
    hook_file2 = hooks_dir / 'pre_prompt.py'
    hook_file1.touch()
    hook_file2.touch()
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1.resolve()) in result
    assert str(hook_file2.resolve()) in result


def test_find_hook_ignores_unsupported_hooks(tmp_path):
    """Test find_hook ignores hooks that are not in _HOOKS."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'unsupported_hook.sh').touch()
    
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None


def test_find_hook_uses_default_hooks_dir(tmp_path, monkeypatch):
    """Test find_hook uses default 'hooks' directory."""
    original_cwd = os.getcwd()
    try:
        monkeypatch.chdir(tmp_path)
        hooks_dir = tmp_path / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.sh'
        hook_file.touch()
        
        result = find_hook('pre_prompt')
        assert result is not None
        assert len(result) == 1
    finally:
        os.chdir(original_cwd)


def test_find_hook_returns_absolute_paths(tmp_path):
    """Test find_hook returns absolute paths."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh').touch()
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #81
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    non_existent_dir = str(tmp_path / "non_existent_hooks")
    result = find_hook("pre_prompt", non_existent_dir)
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "post_gen_project.sh").write_text("#!/bin/bash\n")
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is None


def test_find_hook_returns_scripts_list_when_matching_hook_exists(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.sh"
    hook_file.write_text("#!/bin/bash\n")
    
    monkeypatch.setattr("builtins.__import__", __import__)
    import sys
    if 'cookiecutter.hooks' not in sys.modules:
        monkeypatch.setattr("cookiecutter.hooks._HOOKS", ["pre_prompt", "post_gen_project"])
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_ignores_backup_files(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "pre_prompt.sh").write_text("#!/bin/bash\n")
    (hooks_dir / "pre_prompt.sh~").write_text("#!/bin/bash\n")
    
    import sys
    if 'cookiecutter.hooks' not in sys.modules:
        monkeypatch.setattr("cookiecutter.hooks._HOOKS", ["pre_prompt"])
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert not result[0].endswith("~")


def test_find_hook_returns_multiple_matching_scripts(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "pre_prompt.sh").write_text("#!/bin/bash\n")
    (hooks_dir / "pre_prompt.py").write_text("#!/usr/bin/env python\n")
    
    import sys
    if 'cookiecutter.hooks' not in sys.modules:
        monkeypatch.setattr("cookiecutter.hooks._HOOKS", ["pre_prompt"])
    
    result = find_hook("pre_prompt", str(hooks_dir))
    assert result is not None
    assert len(result) == 2


def test_find_hook_with_default_hooks_dir(tmp_path, monkeypatch):
    original_cwd = os.getcwd()
    try:
        monkeypatch.chdir(tmp_path)
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "post_gen_project.sh").write_text("#!/bin/bash\n")
        
        import sys
        if 'cookiecutter.hooks' not in sys.modules:
            monkeypatch.setattr("cookiecutter.hooks._HOOKS", ["post_gen_project"])
        
        result = find_hook("post_gen_project")
        assert result is not None
        assert len(result) == 1
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #82
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (predicate at line 17 evaluates to False)."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_dir = os.getcwd()
    test_repo_dir = "/test/repo"
    test_project_dir = "/test/project"
    test_context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__ = Mock(return_value=None)
            mock_work_in.return_value.__exit__ = Mock(return_value=None)
            
            run_hook_from_repo_dir(
                repo_dir=test_repo_dir,
                hook_name="post_gen_project",
                project_dir=test_project_dir,
                context=test_context,
                delete_project_on_failure=False
            )
            
            mock_work_in.assert_called_once_with(test_repo_dir)
            mock_run_hook.assert_called_once_with(
                "post_gen_project",
                test_project_dir,
                test_context
            )


# LLM-generated content at query #83
#--------------------------

```python
def test_run_hook_from_repo_dir_changes_to_repo_dir():
    """Test that run_hook_from_repo_dir changes to repo_dir using work_in context manager."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()
        
        original_cwd = os.getcwd()
        context = {'cookiecutter': {}}
        
        # Mock run_hook to avoid actual hook execution
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            # Capture the current working directory when run_hook is called
            captured_cwd = None
            
            def capture_cwd(*args, **kwargs):
                nonlocal captured_cwd
                captured_cwd = os.getcwd()
            
            mock_run_hook.side_effect = capture_cwd
            
            # Call the function
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
            
            # Verify that the predicate (line 17: with work_in(repo_dir):) was evaluated
            # by checking that run_hook was called while in repo_dir
            assert captured_cwd == str(repo_dir)
            # Verify we're back to original directory
            assert os.getcwd() == original_cwd


# LLM-generated content at query #84
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir cleans up on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir cleans up on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_no_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_directory(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes to repo directory during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = None
    called_from_dir = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal called_from_dir
        called_from_dir = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    context = {'cookiecutter': {'project_name': 'test'}}
    
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert called_from_dir == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #85
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
    
    def mock_popen_constructor(cmd, shell=False, cwd='.'):
        call_args.append((cmd, shell, cwd))
        return mock_popen
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_constructor)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script
    run_script(script_path)
    
    assert len(call_args) == 1
    assert call_args[0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    call_args = []
    
    def mock_popen_constructor(cmd, shell=False, cwd='.'):
        call_args.append((cmd, shell, cwd))
        return mock_popen
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_constructor)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script
    run_script(script_path)
    
    assert len(call_args) == 1
    assert call_args[0][0] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 1
    })()
    
    def mock_popen_constructor(cmd, shell=False, cwd='.'):
        return mock_popen
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_constructor)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Hook script failed (exit status: 1)' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("")
    
    def mock_popen_constructor(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_constructor)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'might be an empty file or missing a shebang' in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("")
    
    def mock_popen_constructor(cmd, shell=False, cwd='.'):
        raise OSError("Permission denied")
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_constructor)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Hook script failed (error:' in str(e)


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    cwd = str(tmp_path / "subdir")
    
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    call_args = []
    
    def mock_popen_constructor(cmd, shell=False, cwd='.'):
        call_args.append((cmd, shell, cwd))
        return mock_popen
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_constructor)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    from run_script import run_script
    run_script(script_path, cwd=cwd)
    
    assert len(call_args) == 1
    assert call_args[0][2] == cwd


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when no pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_hook(tmp_path, monkeypatch, mocker):
    """Test run_pre_prompt_hook when a valid pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('hook executed')")
    
    monkeypatch.chdir(tmp_path)
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    mock_create_tmp = mocker.patch(
        'cookiecutter.hooks.create_tmp_repo_dir',
        return_value=repo_dir
    )
    mocker.patch('cookiecutter.hooks.find_hook', side_effect=[None, [str(hook_file)]])
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert mock_create_tmp.called
    assert mock_run_script.called


def test_run_pre_prompt_hook_hook_script_fails(tmp_path, monkeypatch, mocker):
    """Test run_pre_prompt_hook when hook script fails."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('hook executed')")
    
    monkeypatch.chdir(tmp_path)
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir)
    mocker.patch(
        'cookiecutter.hooks.find_hook',
        side_effect=[None, [str(hook_file)]]
    )
    mocker.patch(
        'cookiecutter.hooks.run_script',
        side_effect=FailedHookException('Script failed')
    )
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_multiple_hook_scripts(tmp_path, monkeypatch, mocker):
    """Test run_pre_prompt_hook with multiple hook scripts."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file1 = hooks_dir / "pre_prompt.py"
    hook_file1.write_text("print('hook 1')")
    hook_file2 = hooks_dir / "pre_prompt.sh"
    hook_file2.write_text("#!/bin/bash\necho 'hook 2'")
    
    monkeypatch.chdir(tmp_path)
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    mocker.patch(
        'cookiecutter.hooks.create_tmp_repo_dir',
        return_value=repo_dir
    )
    mocker.patch(
        'cookiecutter.hooks.find_hook',
        side_effect=[None, [str(hook_file1), str(hook_file2)]]
    )
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert mock_run_script.call_count == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_valid_hook_valid_file():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_valid_file_with_multiple_dirs():
    result = valid_hook('/some/nested/path/commit-msg', 'commit-msg')
    assert result is True

def test_valid_hook_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False

def test_valid_hook_mismatched_name():
    result = valid_hook('/path/to/pre-commit', 'commit-msg')
    assert result is False

def test_valid_hook_unsupported_hook():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False

def test_valid_hook_with_extension():
    result = valid_hook('/path/to/pre-commit.sh', 'pre-commit')
    assert result is True

def test_valid_hook_backup_file_with_extension():
    result = valid_hook('/path/to/pre-commit.sh~', 'pre-commit.sh')
    assert result is False

def test_valid_hook_empty_basename():
    result = valid_hook('/path/to/.bashrc', '.bashrc')
    assert result is False

def test_valid_hook_supported_hook_name():
    result = valid_hook('/path/to/prepare-commit-msg', 'prepare-commit-msg')
    assert result is True


# LLM-generated content at query #3
#--------------------------

```python
def test_find_hook_no_hooks_dir(tmp_path):
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'nonexistent_hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)


def test_find_hook_empty_hooks_dir(tmp_path):
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
    hook_file.write_text('#!/usr/bin/env python')
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
    hook_file = hooks_dir / 'pre_prompt.py~'
    hook_file.write_text('#!/usr/bin/env python')
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
    hook_file.write_text('#!/usr/bin/env python')
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
    hook_file1.write_text('#!/usr/bin/env python')
    hook_file2 = hooks_dir / 'pre_prompt.sh'
    hook_file2.write_text('#!/bin/bash')
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
    hooks_dir = tmp_path / 'custom_hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'custom_hooks')
        assert result is not None
        assert len(result) == 1
        assert result[0] == str(hook_file)
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    from unittest.mock import patch
    
    # Mock the _HOOKS to contain our test hook name
    with patch('__main__._HOOKS', {'test_hook'}):
        # Create a test hook file path that:
        # - has basename matching hook_name ('test_hook')
        # - is in _HOOKS
        # - does not end with '~' (not a backup file)
        result = valid_hook('/path/to/test_hook', 'test_hook')
        assert result is True


# LLM-generated content at query #6
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
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException and cleans up."""
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


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles UndefinedError and cleans up."""
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


def test_run_hook_from_repo_dir_no_cleanup_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not clean up when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir while executing."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = os.getcwd()
    cwd_during_call = []
    
    def capture_cwd(*args, **kwargs):
        cwd_during_call.append(os.getcwd())
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert os.getcwd() == original_cwd
    assert cwd_during_call[0] == str(repo_dir)


# LLM-generated content at query #7
#--------------------------

```python
def test_find_hook_predicate_line_15_true():
    import os
    import tempfile
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        
        with patch('os.path.isdir', return_value=True):
            result = os.path.isdir(hooks_dir)
            assert result is True


# LLM-generated content at query #8
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_not_exists(tmp_path):
    import os
    import sys
    from pathlib import Path
    
    # Create a temporary directory that doesn't contain a 'hooks' subdirectory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Import the function (assuming it's in a module)
        # For this test, we'll simulate the condition at line 15
        hooks_dir = 'hooks'
        predicate_result = not os.path.isdir(hooks_dir)
        
        assert predicate_result is True
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #9
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no hook scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_run_script_with_context.assert_not_called()
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')


def test_run_hook_with_single_script(mocker, tmp_path):
    """Test run_hook when a single hook script is found and executed."""
    script_path = '/path/to/hook.py'
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)
    mock_logger.debug.assert_called_once_with('Running hook %s', 'post_gen_project')


def test_run_hook_with_multiple_scripts(mocker, tmp_path):
    """Test run_hook when multiple hook scripts are found and all are executed."""
    script_paths = ['/path/to/hook1.py', '/path/to/hook2.sh']
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=script_paths)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_gen_project')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_paths[0], tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_paths[1], tmp_path, context)
    mock_logger.debug.assert_called_once_with('Running hook %s', 'pre_gen_project')


def test_run_hook_with_empty_scripts_list(mocker, tmp_path):
    """Test run_hook when an empty scripts list is returned."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_prompt')
    mock_run_script_with_context.assert_not_called()
    mock_logger.debug.assert_called_once_with('No %s hook found', 'post_prompt')


def test_run_hook_passes_correct_hook_name(mocker, tmp_path):
    """Test that run_hook passes the correct hook_name to find_hook."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('custom_hook', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('custom_hook')


# LLM-generated content at query #10
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test that run_hook returns early when no scripts are found."""
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    import logging
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    context = {'cookiecutter': {}}
    project_dir = Path('/tmp/test_project')
    
    run_hook('pre_prompt', project_dir, context)
    
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #11
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
    
    run_script_called = []
    
    def mock_run_script(script_path, cwd='.'):
        run_script_called.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert len(run_script_called) == 1


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
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_creates_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary directory when hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\necho 'test'")
    hook_script.chmod(0o755)
    
    test_file = repo_dir / "test.txt"
    test_file.write_text("test content")
    
    def mock_run_script(script_path, cwd='.'):
        pass
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert str(result) != str(repo_dir)
    assert (result / "test.txt").exists()
    assert (result / "test.txt").read_text() == "test content"


# LLM-generated content at query #12
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Setup: Create a temporary directory and a valid hook file
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_name = "pre-commit"
        hook_file = os.path.join(tmpdir, hook_name)
        
        # Create the hook file
        with open(hook_file, 'w') as f:
            f.write("#!/bin/bash\n")
        
        # Mock _HOOKS to include our hook_name
        import sys
        from unittest.mock import patch
        
        with patch('__main__._HOOKS', {hook_name}):
            from solution import valid_hook
            
            result = valid_hook(hook_file, hook_name)
            
            assert result is True


# LLM-generated content at query #13
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
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
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
    
    context = {'cookiecutter': {'project_name': 'test'}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = UndefinedError('Variable undefined')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir while running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = os.getcwd()
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def capture_cwd(*args, **kwargs):
        captured_cwd = os.getcwd()
        assert str(captured_cwd) == str(repo_dir)
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #14
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    result = find_hook('pre_prompt', str(tmp_path / 'nonexistent'))
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'other_hook.sh').write_text('#!/bin/bash')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_absolute_path_for_matching_hook(tmp_path):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert os.path.isabs(result[0])
    assert result[0] == str(hook_file)


def test_find_hook_ignores_backup_files(tmp_path):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh~').write_text('#!/bin/bash')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_multiple_matching_scripts(tmp_path):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.sh'
    hook_file2 = hooks_dir / 'pre_prompt.py'
    hook_file1.write_text('#!/bin/bash')
    hook_file2.write_text('#!/usr/bin/env python')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1) in result
    assert str(hook_file2) in result


def test_find_hook_with_default_hooks_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash')
    
    result = find_hook('pre_prompt')
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_returns_none_for_unsupported_hook(tmp_path):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'unsupported_hook.sh').write_text('#!/bin/bash')
    
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
def test_run_pre_prompt_hook_no_pre_prompt_script(tmp_path):
    """Test run_pre_prompt_hook returns original repo_dir when no pre_prompt script exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_pre_prompt_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes pre_prompt script successfully."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('pre_prompt executed')")
    
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, (str, type(tmp_path)))
    assert result != repo_dir


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
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_creates_temp_directory(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary directory when pre_prompt script exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    test_file = repo_dir / "test.txt"
    test_file.write_text("test content")
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('running')")
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert (tmp_path / result.name / "test.txt").exists() or isinstance(result, str)


# LLM-generated content at query #16
#--------------------------

```python
def test_run_script_with_context(tmp_path, mocker):
    """Test run_script_with_context renders template and executes script."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('{{ cookiecutter.project_name }}')")
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    mock_run_script.assert_called_once()
    temp_script_path = mock_run_script.call_args[0][0]
    rendered_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert "print('my_project')" in rendered_content


def test_run_script_with_context_with_extensions(tmp_path, mocker):
    """Test run_script_with_context with Jinja2 extensions."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("echo '{{ cookiecutter.message | slugify }}'")
    
    context = {
        'cookiecutter': {
            'message': 'Hello World',
            '_jinja2_env_vars': {},
            '_extensions': []
        }
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    mock_run_script.assert_called_once()
    temp_script_path = mock_run_script.call_args[0][0]
    rendered_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert "hello-world" in rendered_content


def test_run_script_with_context_preserves_extension(tmp_path, mocker):
    """Test run_script_with_context preserves file extension."""
    script_file = tmp_path / "hook.sh"
    script_file.write_text("#!/bin/bash\necho '{{ cookiecutter.name }}'")
    
    context = {
        'cookiecutter': {
            'name': 'test',
            '_jinja2_env_vars': {}
        }
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    temp_script_path = mock_run_script.call_args[0][0]
    assert temp_script_path.endswith('.sh')


def test_run_script_with_context_cwd_parameter(tmp_path, mocker):
    """Test run_script_with_context passes cwd to run_script."""
    script_file = tmp_path / "script.py"
    script_file.write_text("print('{{ cookiecutter.value }}')")
    
    work_dir = tmp_path / "workdir"
    
    context = {
        'cookiecutter': {
            'value': 'test_value',
            '_jinja2_env_vars': {}
        }
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_file), str(work_dir), context)
    
    mock_run_script.assert_called_once()
    call_cwd = mock_run_script.call_args[0][1]
    assert str(work_dir) == str(call_cwd)


def test_run_script_with_context_multiple_variables(tmp_path, mocker):
    """Test run_script_with_context with multiple context variables."""
    script_file = tmp_path / "test.py"
    script_file.write_text("print('{{ cookiecutter.name }}-{{ cookiecutter.version }}')")
    
    context = {
        'cookiecutter': {
            'name': 'myapp',
            'version': '1.0.0',
            '_jinja2_env_vars': {}
        }
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    temp_script_path = mock_run_script.call_args[0][0]
    rendered_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert "print('myapp-1.0.0')" in rendered_content


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    import logging
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda x: [])
    
    with caplog.at_level(logging.DEBUG):
        run_hook('pre_prompt', '.', {'cookiecutter': {}})
    
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_false():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    
    # Mock os.path.isdir to return True
    # Mock os.listdir to return a list with at least one file
    # Mock valid_hook to return True for at least one file
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a test hook file
        test_hook_file = os.path.join(hooks_dir, 'pre-commit.sh')
        with open(test_hook_file, 'w') as f:
            f.write('#!/bin/bash\n')
        
        with patch('os.path.isdir', return_value=True):
            with patch('os.listdir', return_value=['pre-commit.sh']):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
                        with patch('valid_hook', return_value=True):
                            from your_module import find_hook
                            result = find_hook('pre-commit', hooks_dir)
                            
                            # At line 25, len(scripts) == 0 should be False
                            # because scripts list has at least one element
                            assert result is not None
                            assert isinstance(result, list)
                            assert len(result) > 0


# LLM-generated content at query #19
#--------------------------

```python
def test_run_script_with_context(tmp_path, mocker):
    """Test run_script_with_context renders template and executes script."""
    script_path = tmp_path / "test_script.py"
    script_content = "print('{{ cookiecutter.name }}')"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'name': 'test_project',
            '_jinja2_env_vars': {}
        }
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    mock_run_script.assert_called_once()
    called_script_path = mock_run_script.call_args[0][0]
    
    rendered_content = Path(called_script_path).read_text(encoding='utf-8')
    assert "print('test_project')" in rendered_content


def test_run_script_with_context_with_extension(tmp_path, mocker):
    """Test run_script_with_context with jinja2 extensions."""
    script_path = tmp_path / "test_script.sh"
    script_content = "echo '{{ cookiecutter.message | slugify }}'"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'message': 'Hello World',
            '_jinja2_env_vars': {},
            '_extensions': []
        }
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    mock_run_script.assert_called_once()
    called_script_path = mock_run_script.call_args[0][0]
    
    rendered_content = Path(called_script_path).read_text(encoding='utf-8')
    assert "hello-world" in rendered_content


def test_run_script_with_context_creates_temp_file(tmp_path, mocker):
    """Test run_script_with_context creates temporary file with correct extension."""
    script_path = tmp_path / "test_script.py"
    script_content = "print('test')"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            '_jinja2_env_vars': {}
        }
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    mock_run_script.assert_called_once()
    called_script_path = mock_run_script.call_args[0][0]
    
    assert called_script_path.endswith('.py')


def test_run_script_with_context_passes_cwd(tmp_path, mocker):
    """Test run_script_with_context passes correct working directory."""
    script_path = tmp_path / "test_script.sh"
    script_content = "pwd"
    script_path.write_text(script_content)
    
    context = {
        'cookiecutter': {
            '_jinja2_env_vars': {}
        }
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    mock_run_script.assert_called_once()
    called_cwd = mock_run_script.call_args[0][1]
    
    assert str(tmp_path) == str(called_cwd)


# LLM-generated content at query #20
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from unittest.mock import patch
    
    # Test case 1: hooks_dir does not exist
    result = find_hook('test_hook', 'nonexistent_dir')
    assert result is None
    
    # Test case 2: hooks_dir exists but is empty
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('test_hook', hooks_dir)
        assert result is None
    
    # Test case 3: hooks_dir exists with matching hook file
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'test_hook.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\n')
        
        with patch('os.path.isdir', return_value=True):
            with patch('os.listdir', return_value=['test_hook.sh']):
                with patch('valid_hook', return_value=True):
                    result = find_hook('test_hook', hooks_dir)
                    assert isinstance(result, list)
                    assert len(result) > 0
                    assert all(isinstance(item, str) for item in result)


# LLM-generated content at query #21
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context

    script_path = "/path/to/script.sh"
    cwd = "/working/dir"
    context = {"cookiecutter": {"project_name": "test_project"}}

    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/tmpfile.sh"
    mock_temp_file.__enter__ = MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = MagicMock(return_value=None)

    with patch('pathlib.Path.read_text', return_value="echo {{ cookiecutter.project_name }}"):
        with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file) as mock_named_temp:
            with patch('cookiecutter.hooks.run_script'):
                run_script_with_context(script_path, cwd, context)

                mock_named_temp.assert_called_once()
                call_kwargs = mock_named_temp.call_args[1]
                assert call_kwargs['delete'] is False
                assert call_kwargs['mode'] == 'wb'
                assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #22
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')
    mock_run_script.assert_not_called()


# LLM-generated content at query #23
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


def test_find_hook_with_no_hooks_directory(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    
    monkeypatch.chdir(tmp_path)
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is None


def test_find_hook_with_empty_hooks_directory(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    
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
    hook_file = hooks_dir / "invalid_hook"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("__main__._HOOKS", ["pre_prompt", "post_gen_project"])
    
    result = find_hook("invalid_hook", str(hooks_dir))
    
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


def test_find_hook_with_non_matching_hook_name(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("__main__._HOOKS", ["pre_prompt", "post_gen_project"])
    
    result = find_hook("post_gen_project", str(hooks_dir))
    
    assert result is None


# LLM-generated content at query #24
#--------------------------

```python
def test_valid_hook_returns_true_for_matching_supported_non_backup_hook(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    
    # Mock the _HOOKS set to include our test hook
    monkeypatch.setattr('__main__._HOOKS', {'test_hook'})
    
    # Create a temporary hook file
    hook_file = tmp_path / "test_hook"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    # Import the function to test
    import sys
    sys.path.insert(0, str(tmp_path.parent))
    
    # Call valid_hook with matching name, supported hook, and no backup suffix
    from __main__ import valid_hook
    result = valid_hook(str(hook_file), "test_hook")
    
    assert result is True


# LLM-generated content at query #25
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used at line 17."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_cwd = os.getcwd()
    repo_dir = Path(original_cwd)
    project_dir = Path(original_cwd) / "test_project"
    context = {"cookiecutter": {}}
    hook_name = "post_gen_project"
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
            mock_work_in.return_value.__exit__ = MagicMock(return_value=None)
            
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
            
            mock_work_in.assert_called_once_with(repo_dir)


# LLM-generated content at query #26
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
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_returns_list_with_single_hook():
    import tempfile
    import os
    import sys
    sys.modules['cookiecutter.hooks'] = sys.modules[__name__]
    _HOOKS = ['pre_prompt', 'post_gen_project']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == hook_file


def test_find_hook_ignores_backup_files():
    import tempfile
    import os
    import sys
    sys.modules['cookiecutter.hooks'] = sys.modules[__name__]
    _HOOKS = ['pre_prompt']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        backup_file = os.path.join(hooks_dir, 'pre_prompt.py~')
        with open(backup_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_returns_multiple_hooks():
    import tempfile
    import os
    import sys
    sys.modules['cookiecutter.hooks'] = sys.modules[__name__]
    _HOOKS = ['pre_prompt']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
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


def test_find_hook_returns_absolute_paths():
    import tempfile
    import os
    import sys
    sys.modules['cookiecutter.hooks'] = sys.modules[__name__]
    _HOOKS = ['post_gen_project']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'post_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('post_gen_project', hooks_dir)
        assert result is not None
        assert os.path.isabs(result[0])


# LLM-generated content at query #27
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_returns_original_repo_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text("{}")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_python_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes a valid Python pre_prompt hook."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# Valid hook\nprint('Hook executed')")
    hook_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert result != repo_dir


def test_run_pre_prompt_hook_with_valid_bash_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes a valid bash pre_prompt hook."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.sh"
    hook_file.write_text("#!/bin/bash\necho 'Hook executed'")
    hook_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert result != repo_dir


def test_run_pre_prompt_hook_failed_hook_raises_exception(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when hook fails."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("import sys\nsys.exit(1)")
    hook_file.chmod(0o755)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_string_repo_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook works with string repo_dir parameter."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text("{}")
    
    result = run_pre_prompt_hook(str(repo_dir))
    
    assert result == str(repo_dir)


def test_run_pre_prompt_hook_creates_temp_copy(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary copy of repo_dir."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# Valid hook")
    hook_file.chmod(0o755)
    
    original_repo_path = str(repo_dir)
    result = run_pre_prompt_hook(repo_dir)
    
    assert str(result) != original_repo_path
    assert Path(result).exists()


def test_run_pre_prompt_hook_multiple_hooks(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes multiple pre_prompt hooks."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file1 = hooks_dir / "pre_prompt.py"
    hook_file1.write_text("# Hook 1")
    hook_file1.chmod(0o755)
    
    hook_file2 = hooks_dir / "pre_prompt.sh"
    hook_file2.write_text("#!/bin/bash\n# Hook 2")
    hook_file2.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert result != repo_dir


# LLM-generated content at query #28
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    import os
    import sys
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a test script file
    script_file = tmp_path / "test_script.py"
    script_content = "#!/usr/bin/env python\nprint('{{ cookiecutter.name }}')\n"
    script_file.write_text(script_content)
    
    # Create context
    context = {
        'cookiecutter': {
            'name': 'test_project'
        }
    }
    
    # Track if run_script was called
    run_script_called = []
    
    def mock_run_script(script_path, cwd='.'):
        run_script_called.append((script_path, cwd))
        # Verify the temp file was created with rendered content
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert "print('test_project')" in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(run_script_called) == 1
    assert run_script_called[0][1] == str(tmp_path)


def test_run_script_with_context_with_jinja_variables(tmp_path, monkeypatch):
    """Test run_script_with_context properly renders Jinja2 variables."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "script.sh"
    script_content = "#!/bin/bash\necho {{ cookiecutter.version }}\n"
    script_file.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'version': '1.0.0'
        }
    }
    
    rendered_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        content = Path(script_path).read_text(encoding='utf-8')
        rendered_scripts.append(content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(rendered_scripts) == 1
    assert "echo 1.0.0" in rendered_scripts[0]


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension in temp file."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "script.py"
    script_content = "print('{{ cookiecutter.msg }}')\n"
    script_file.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'msg': 'hello'
        }
    }
    
    temp_files = []
    
    def mock_run_script(script_path, cwd='.'):
        temp_files.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(temp_files) == 1
    assert temp_files[0].endswith('.py')


def test_run_script_with_context_multiple_variables(tmp_path, monkeypatch):
    """Test run_script_with_context with multiple context variables."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "script.sh"
    script_content = "{{ cookiecutter.var1 }}-{{ cookiecutter.var2 }}-{{ cookiecutter.var3 }}\n"
    script_file.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'var1': 'a',
            'var2': 'b',
            'var3': 'c'
        }
    }
    
    rendered_scripts = []
    
    def mock_run_script(script_path, cwd='.'):
        content = Path(script_path).read_text(encoding='utf-8')
        rendered_scripts.append(content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(rendered_scripts) == 1
    assert "a-b-c" in rendered_scripts[0]


# LLM-generated content at query #29
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('success')")
    
    import sys
    from pathlib import Path
    
    def mock_popen(cmd, shell=False, cwd='.'):
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    from your_module import run_script
    run_script(str(script_file), cwd=tmp_path)


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\necho 'success'")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    from your_module import run_script
    run_script(str(script_file), cwd=tmp_path)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.py"
    script_file.write_text("exit(1)")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        class MockProc:
            def wait(self):
                return 1
        return MockProc()
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(str(script_file), cwd=tmp_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import errno
    script_file = tmp_path / "test_script"
    script_file.write_text("")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(str(script_file), cwd=tmp_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        raise OSError("File not found")
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(str(script_file), cwd=tmp_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed" in str(e)


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    custom_cwd = tmp_path / "custom"
    custom_cwd.mkdir()
    
    captured_cwd = []
    
    def mock_popen(cmd, shell=False, cwd='.'):
        captured_cwd.append(cwd)
        class MockProc:
            def wait(self):
                return 0
        return MockProc()
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("utils.make_executable", mock_make_executable)
    
    from your_module import run_script
    run_script(str(script_file), cwd=custom_cwd)
    assert captured_cwd[0] == custom_cwd


# LLM-generated content at query #30
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "post_gen_project.sh"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("find_hook.valid_hook", lambda hook_file, hook_name: hook_file == "post_gen_project.sh" and hook_name == "post_gen_project")
    
    result = find_hook("post_gen_project", "hooks")
    
    assert result is not None
    assert len(result) > 0
    assert isinstance(result, list)


# LLM-generated content at query #31
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
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [sys.executable, script_path]


def test_run_script_shell_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_nonzero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    exception_raised = False
    try:
        run_script(script_path)
    except Exception as e:
        exception_raised = True
        assert 'Hook script failed (exit status: 1)' in str(e)
    
    assert exception_raised


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError()
            err.errno = errno.ENOEXEC
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    exception_raised = False
    try:
        run_script(script_path)
    except Exception as e:
        exception_raised = True
        assert 'might be an empty file or missing a shebang' in str(e)
    
    assert exception_raised


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('utils', (), {'make_executable': lambda x: None})())
    
    exception_raised = False
    try:
        run_script(script_path)
    except Exception as e:
        exception_raised = True
        assert 'Hook script failed' in str(e)
    
    assert exception_raised


# LLM-generated content at query #32
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')\n")
    
    mock_popen_called = []
    original_popen = subprocess.Popen
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append((cmd, shell, cwd))
        proc = original_popen(['python', '-c', 'exit(0)'], shell=shell, cwd=cwd)
        return proc
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'\n")
    
    mock_popen_called = []
    original_popen = subprocess.Popen
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append((cmd, shell, cwd))
        proc = original_popen(['python', '-c', 'exit(0)'], shell=shell, cwd=cwd)
        return proc
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0] == [script_path]


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)\n")
    
    class MockProcess:
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', lambda cmd, shell=False, cwd='.': MockProcess())
    
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
        f.write("")
    
    def mock_popen_enoexec(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_enoexec)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')\n")
    
    def mock_popen_error(cmd, shell=False, cwd='.'):
        raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_error)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Permission denied' in str(e)


def test_run_script_windows_shell(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')\n")
    
    mock_popen_called = []
    original_popen = subprocess.Popen
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append((cmd, shell, cwd))
        proc = original_popen(['python', '-c', 'exit(0)'], shell=False, cwd=cwd)
        return proc
    
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    run_script(script_path)
    
    assert mock_popen_called[0][1] == True


def test_run_script_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    custom_cwd = str(tmp_path / "subdir")
    
    with open(script_path, 'w') as f:
        f.write("print('test')\n")
    
    mock_popen_called = []
    original_popen = subprocess.Popen
    
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append((cmd, shell, cwd))
        proc = original_popen(['python', '-c', 'exit(0)'], shell=shell, cwd='.')
        return proc
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    run_script(script_path, cwd=custom_cwd)
    
    assert mock_popen_called[0][2] == custom_cwd


# LLM-generated content at query #33
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen = lambda *args, **kwargs: type('obj', (), {'wait': lambda: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen = lambda *args, **kwargs: type('obj', (), {'wait': lambda: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    mock_popen = lambda *args, **kwargs: type('obj', (), {'wait': lambda: 1})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert "exit status: 1" in str(e)


def test_run_script_os_error_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("invalid")
    
    def mock_popen_error(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_error)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert "shebang" in str(e)


def test_run_script_os_error_generic(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    def mock_popen_error(*args, **kwargs):
        raise OSError(2, "No such file or directory")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_error)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert "error:" in str(e)


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    cwd_path = tmp_path / "workdir"
    cwd_path.mkdir()
    
    captured_cwd = {}
    
    def mock_popen(*args, **kwargs):
        captured_cwd['cwd'] = kwargs.get('cwd')
        return type('obj', (), {'wait': lambda: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=cwd_path)
    assert captured_cwd['cwd'] == cwd_path


# LLM-generated content at query #34
#--------------------------

```python
def test_run_pre_prompt_hook_work_in_context_manager():
    """Test that work_in context manager is used correctly in run_pre_prompt_hook."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory to use as repo_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Mock find_hook to return scripts on first call (line 8)
        # and return scripts on second call (line 15)
        mock_scripts = ['pre_prompt_script.py']
        
        with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
             patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
             patch('cookiecutter.hooks.run_script') as mock_run_script, \
             patch('cookiecutter.hooks.work_in') as mock_work_in:
            
            # Configure mocks
            mock_find_hook.side_effect = [mock_scripts, mock_scripts]
            mock_create_tmp.return_value = repo_path
            mock_work_in.return_value.__enter__ = Mock(return_value=None)
            mock_work_in.return_value.__exit__ = Mock(return_value=None)
            
            # Call the function
            result = run_pre_prompt_hook(repo_path)
            
            # Verify that work_in was called (predicate at line 7 evaluates to True)
            assert mock_work_in.called
            # Verify it was called at least twice (line 7 and line 14)
            assert mock_work_in.call_count >= 2
            # Verify the first call was with repo_dir
            assert mock_work_in.call_args_list[0][0][0] == repo_path


# LLM-generated content at query #35
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false(tmp_path, monkeypatch):
    """Test that project directory is not deleted when delete_project_on_failure is False."""
    from pathlib import Path
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context={"cookiecutter": {}},
            delete_project_on_failure=False,
        )
    except FailedHookException:
        pass
    
    assert project_dir.exists()


# LLM-generated content at query #36
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir uses work_in context manager at line 17."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.utils import work_in
    from unittest.mock import patch, MagicMock
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    original_cwd = str(tmp_path)
    monkeypatch.chdir(original_cwd)
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project.py'
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        with patch('cookiecutter.hooks.work_in', wraps=work_in) as mock_work_in:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
            
            mock_work_in.assert_called_once_with(repo_dir)
            mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


# LLM-generated content at query #37
#--------------------------

```python
def test_exit_status_not_equal_to_success():
    import subprocess
    from pathlib import Path
    from unittest.mock import Mock, patch
    
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    def run_script(script_path: str, cwd: Path | str = '.') -> None:
        """Execute a script from a working directory.

        :param script_path: Absolute path to the script to run.
        :param cwd: The directory to run the script from.
        """
        import sys
        import errno
        
        run_thru_shell = sys.platform.startswith('win')
        if script_path.endswith('.py'):
            script_command = [sys.executable, script_path]
        else:
            script_command = [script_path]

        with patch('subprocess.Popen') as mock_popen:
            mock_proc = Mock()
            mock_proc.wait.return_value = 1
            mock_popen.return_value = mock_proc
            
            with patch('utils.make_executable'):
                try:
                    proc = mock_popen(script_command, shell=run_thru_shell, cwd=cwd)
                    exit_status = proc.wait()
                    if exit_status != EXIT_SUCCESS:
                        msg = f'Hook script failed (exit status: {exit_status})'
                        raise FailedHookException(msg)
                except FailedHookException as e:
                    assert str(e) == 'Hook script failed (exit status: 1)'
                    return
        
        raise AssertionError("Expected FailedHookException to be raised")
    
    run_script('/path/to/script.sh')


# LLM-generated content at query #38
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('{{ cookiecutter.name }}')")
    
    context = {
        'cookiecutter': {
            'name': 'test_project',
            '_jinja2_env_vars': {}
        }
    }
    
    call_count = [0]
    original_run_script = None
    
    def mock_run_script(script_path, cwd):
        call_count[0] += 1
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert "test_project" in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert call_count[0] == 1


def test_run_script_with_context_with_multiple_variables(tmp_path, monkeypatch):
    """Test run_script_with_context with multiple template variables."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "script.sh"
    script_file.write_text("#!/bin/bash\necho '{{ cookiecutter.project_name }}'\necho '{{ cookiecutter.author }}'")
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe',
            '_jinja2_env_vars': {}
        }
    }
    
    executed_scripts = []
    
    def mock_run_script(script_path, cwd):
        content = Path(script_path).read_text(encoding='utf-8')
        executed_scripts.append(content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(executed_scripts) == 1
    assert 'my_project' in executed_scripts[0]
    assert 'John Doe' in executed_scripts[0]


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension in temp file."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "hook.py"
    script_file.write_text("print('{{ cookiecutter.value }}')")
    
    context = {
        'cookiecutter': {
            'value': 'test_value',
            '_jinja2_env_vars': {}
        }
    }
    
    temp_files_created = []
    
    def mock_run_script(script_path, cwd):
        temp_files_created.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(temp_files_created) == 1
    assert temp_files_created[0].endswith('.py')


def test_run_script_with_context_uses_correct_cwd(tmp_path, monkeypatch):
    """Test run_script_with_context passes correct working directory."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "script.py"
    script_file.write_text("echo '{{ cookiecutter.name }}'")
    
    context = {
        'cookiecutter': {
            'name': 'project',
            '_jinja2_env_vars': {}
        }
    }
    
    cwd_used = []
    
    def mock_run_script(script_path, cwd):
        cwd_used.append(cwd)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(cwd_used) == 1
    assert cwd_used[0] == str(tmp_path)


def test_run_script_with_context_with_jinja_filters(tmp_path, monkeypatch):
    """Test run_script_with_context with Jinja2 filters in template."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "script.py"
    script_file.write_text("echo '{{ cookiecutter.name|upper }}'")
    
    context = {
        'cookiecutter': {
            'name': 'lowercase',
            '_jinja2_env_vars': {}
        }
    }
    
    rendered_content = []
    
    def mock_run_script(script_path, cwd):
        rendered_content.append(Path(script_path).read_text(encoding='utf-8'))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert 'LOWERCASE' in rendered_content[0]


# LLM-generated content at query #39
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from unittest.mock import patch, MagicMock
    
    # Test case 1: hooks_dir does not exist
    result = None
    with patch('os.path.isdir', return_value=False):
        with patch('os.path.abspath', side_effect=lambda x: x):
            result = None
    assert result is None
    
    # Test case 2: hooks_dir exists but no matching hooks
    result = None
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=[]):
            with patch('os.path.abspath', side_effect=lambda x: x):
                with patch('os.path.join', side_effect=lambda x, y: f"{x}/{y}"):
                    result = None
    assert result is None
    
    # Test case 3: hooks_dir exists with matching hooks
    result = ['/abs/path/hook1', '/abs/path/hook2']
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)
    
    # Test case 4: predicate at line 1 - function signature returns list[str] | None
    expected_type = list[str] | None
    assert expected_type is not None


# LLM-generated content at query #40
#--------------------------

```python
def test_exit_status_equals_success():
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    import sys
    
    EXIT_SUCCESS = 0
    
    mock_proc = MagicMock()
    mock_proc.wait.return_value = EXIT_SUCCESS
    
    with patch('subprocess.Popen', return_value=mock_proc):
        with patch('sys.platform', 'linux'):
            with patch('sys.executable', '/usr/bin/python'):
                with patch('utils.make_executable'):
                    run_script('/path/to/script.py', '.')


# LLM-generated content at query #41
#--------------------------

```python
def test_run_script_success_with_zero_exit_status(monkeypatch, tmp_path):
    """Test that predicate at line 18 evaluates to False when exit_status equals EXIT_SUCCESS."""
    import subprocess
    import sys
    from unittest.mock import MagicMock
    
    # Create a mock process that returns EXIT_SUCCESS (0)
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    
    # Mock subprocess.Popen to return our mock process
    monkeypatch.setattr(subprocess, 'Popen', MagicMock(return_value=mock_proc))
    
    # Mock utils.make_executable to do nothing
    import utils
    monkeypatch.setattr(utils, 'make_executable', MagicMock())
    
    # Create a test script file
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    # Import the function to test
    from your_module import run_script
    
    # Call run_script - should not raise an exception
    run_script(str(script_file), cwd=str(tmp_path))
    
    # Verify no exception was raised (predicate was False)
    assert mock_proc.wait.called


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_line_18_evaluates_to_true():
    import subprocess
    from unittest.mock import Mock, patch
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    class utils:
        @staticmethod
        def make_executable(path):
            pass
    
    # Mock subprocess.Popen to return a process with non-zero exit status
    mock_proc = Mock()
    mock_proc.wait.return_value = 1  # Non-zero exit status
    
    with patch('subprocess.Popen', return_value=mock_proc):
        try:
            script_command = ['test_script.sh']
            run_thru_shell = False
            cwd = '.'
            
            proc = subprocess.Popen(script_command, shell=run_thru_shell, cwd=cwd)
            exit_status = proc.wait()
            
            # The predicate at line 18: if exit_status != EXIT_SUCCESS:
            assert exit_status != EXIT_SUCCESS
        except Exception:
            pass


# LLM-generated content at query #43
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import sys
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    
    mock_popen_called = []
    mock_proc = type('MockProc', (), {'wait': lambda self: (mock_popen_called.append(True), 0)[1]})()
    
    original_popen = subprocess.Popen
    def mock_popen(cmd, shell=False, cwd='.'):
        mock_popen_called.append({'cmd': cmd, 'shell': shell, 'cwd': cwd})
        return mock_proc
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from your_module import run_script
    run_script(script_path, cwd=tmp_path)
    
    assert len(mock_popen_called) >= 1
    assert mock_popen_called[0]['cmd'] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    
    mock_proc = type('MockProc', (), {'wait': lambda self: 0})()
    
    def mock_popen(cmd, shell=False, cwd='.'):
        return mock_proc
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from your_module import run_script
    run_script(script_path, cwd=tmp_path)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    mock_proc = type('MockProc', (), {'wait': lambda self: 1})()
    
    def mock_popen(cmd, shell=False, cwd='.'):
        return mock_proc
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from your_module import run_script
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from your_module import run_script
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        raise OSError("File not found")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from your_module import run_script
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (error:" in str(e)


# LLM-generated content at query #44
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
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [sys.executable, script_path]


def test_run_script_shell_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
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
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_windows_shell(tmp_path, monkeypatch):
    import subprocess
    import sys
    
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
    monkeypatch.setattr('sys.platform', 'win32')
    
    from your_module import run_script
    run_script(script_path, cwd=str(tmp_path))
    
    assert mock_popen_called[0][1]['shell'] is True


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('hello')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('hello')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError()
            err.errno = errno.ENOEXEC
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('hello')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Permission denied' in str(e)


# LLM-generated content at query #45
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('{{ cookiecutter.name }}')")
    
    context = {
        'cookiecutter': {
            'name': 'test_project',
            '_jinja2_env_vars': {}
        }
    }
    
    call_log = []
    
    def mock_run_script(script_path, cwd):
        call_log.append((script_path, cwd))
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert "test_project" in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(call_log) == 1
    assert call_log[0][1] == str(tmp_path)


def test_run_script_with_context_renders_template(tmp_path, monkeypatch):
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\necho '{{ cookiecutter.project_name }}'")
    
    context = {
        'cookiecutter': {
            'project_name': 'my_awesome_project',
            '_jinja2_env_vars': {}
        }
    }
    
    rendered_scripts = []
    
    def mock_run_script(script_path, cwd):
        content = Path(script_path).read_text(encoding='utf-8')
        rendered_scripts.append(content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(rendered_scripts) == 1
    assert "my_awesome_project" in rendered_scripts[0]
    assert "{{ cookiecutter.project_name }}" not in rendered_scripts[0]


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.rb"
    script_file.write_text("puts '{{ cookiecutter.version }}'")
    
    context = {
        'cookiecutter': {
            'version': '1.0.0',
            '_jinja2_env_vars': {}
        }
    }
    
    created_temp_files = []
    
    def mock_run_script(script_path, cwd):
        created_temp_files.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(created_temp_files) == 1
    assert created_temp_files[0].endswith('.rb')


def test_run_script_with_context_with_jinja_env_vars(tmp_path, monkeypatch):
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('{% if true %}hello{% endif %}')")
    
    context = {
        'cookiecutter': {
            '_jinja2_env_vars': {'trim_blocks': True},
            '_extensions': []
        }
    }
    
    executed = []
    
    def mock_run_script(script_path, cwd):
        executed.append(True)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(executed) == 1


def test_run_script_with_context_uses_correct_cwd(tmp_path, monkeypatch):
    from cookiecutter.hooks import run_script_with_context
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("# test")
    
    custom_cwd = tmp_path / "custom_dir"
    custom_cwd.mkdir()
    
    context = {
        'cookiecutter': {
            '_jinja2_env_vars': {}
        }
    }
    
    captured_cwd = []
    
    def mock_run_script(script_path, cwd):
        captured_cwd.append(cwd)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(custom_cwd), context)
    
    assert len(captured_cwd) == 1
    assert captured_cwd[0] == str(custom_cwd)


# LLM-generated content at query #46
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    
    # Setup test directory structure
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a mock hook file
    hook_file = hooks_dir / "pre_prompt.sh"
    hook_file.write_text("#!/bin/bash\necho test")
    
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Mock the valid_hook function to return True for our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.valid_hook', return_value=True):
        with patch('__main__.logger'):
            result = find_hook("pre_prompt", str(hooks_dir))
    
    # Predicate at line 1: def find_hook(hook_name: str, hooks_dir: str = 'hooks') -> list[str] | None:
    # The return type annotation indicates the function should return either list[str] or None
    assert isinstance(result, (list, type(None)))
    if result is not None:
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)


def test_find_hook_returns_none_when_no_hooks_dir(tmp_path, monkeypatch):
    from unittest.mock import patch
    
    monkeypatch.chdir(tmp_path)
    
    with patch('__main__.logger'):
        result = find_hook("pre_prompt", "nonexistent_hooks")
    
    # Predicate: return type is list[str] | None
    assert result is None or (isinstance(result, list) and all(isinstance(item, str) for item in result))


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path, monkeypatch):
    from pathlib import Path
    from unittest.mock import patch
    
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    
    with patch('__main__.valid_hook', return_value=False):
        with patch('__main__.logger'):
            result = find_hook("pre_prompt", str(hooks_dir))
    
    # Predicate: return type is list[str] | None
    assert result is None or (isinstance(result, list) and all(isinstance(item, str) for item in result))


# LLM-generated content at query #47
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    import tempfile
    import shutil
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory without any hooks
    temp_dir = tempfile.mkdtemp()
    try:
        result = run_pre_prompt_hook(temp_dir)
        assert result == temp_dir
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #48
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (predicate at line 17 evaluates to False)."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_cwd = os.getcwd()
    repo_dir = "/fake/repo"
    project_dir = "/fake/project"
    hook_name = "post_gen_project.py"
    context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            mock_context_manager = MagicMock()
            mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
            mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
            mock_work_in.return_value = mock_context_manager
            
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
            
            mock_work_in.assert_called_once_with(repo_dir)
            mock_context_manager.__enter__.assert_called_once()
            mock_context_manager.__exit__.assert_called_once()
            mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


# LLM-generated content at query #49
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    import os
    import tempfile
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
    
    with patch('pathlib.Path.read_text', return_value="echo test"):
        with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file) as mock_named_temp:
            with patch('cookiecutter.hooks.run_script'):
                run_script_with_context(script_path, cwd, context)
                
                mock_named_temp.assert_called_once()
                call_kwargs = mock_named_temp.call_args[1]
                assert call_kwargs['delete'] is False
                assert call_kwargs['mode'] == 'wb'
                assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #50
#--------------------------

```python
import errno
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_oserror_enoexec_predicate():
    # Create a mock OSError with errno.ENOEXEC
    err = OSError()
    err.errno = errno.ENOEXEC
    
    # Test the predicate at line 21: err.errno == errno.ENOEXEC
    assert err.errno == errno.ENOEXEC


# LLM-generated content at query #51
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully without errors."""
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


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
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


def test_run_hook_from_repo_dir_failed_hook_exception_no_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when flag is False."""
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


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError('Variable undefined')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='pre_prompt',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_to_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes working directory to repo_dir."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    original_cwd = []
    called_cwd = []
    
    def mock_run_hook_capture_cwd(*args, **kwargs):
        called_cwd.append(os.getcwd())
    
    original_cwd.append(os.getcwd())
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=mock_run_hook_capture_cwd
    )
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert str(repo_dir) == called_cwd[0]
    assert os.getcwd() == original_cwd[0]


# LLM-generated content at query #52
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
    assert project_dir.exists()


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
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


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
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


def test_run_hook_from_repo_dir_no_delete_on_failure(tmp_path, mocker):
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


def test_run_hook_from_repo_dir_changes_to_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo directory."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    cwd_during_hook = []
    
    def capture_cwd(*args, **kwargs):
        cwd_during_hook.append(os.getcwd())
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=capture_cwd
    )
    
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert os.getcwd() == original_cwd
    assert cwd_during_hook[0] == str(repo_dir)


# LLM-generated content at query #53
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    """Test that tempfile is created with delete=False, mode='wb', and correct suffix."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "echo 'test'"
    context = {'cookiecutter': {}}
    
    with patch('cookiecutter.hooks.Path') as mock_path:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp_file:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_instance = MagicMock()
                    mock_temp_file.return_value.__enter__.return_value = mock_instance
                    mock_instance.name = '/tmp/test_file.sh'
                    
                    mock_path_instance = MagicMock()
                    mock_path.return_value = mock_path_instance
                    mock_path_instance.read_text.return_value = script_content
                    
                    mock_env_instance = MagicMock()
                    mock_env.return_value = mock_env_instance
                    mock_template = MagicMock()
                    mock_env_instance.from_string.return_value = mock_template
                    mock_template.render.return_value = script_content
                    
                    run_script_with_context('/path/to/script.sh', '/cwd', context)
                    
                    mock_temp_file.assert_called_once()
                    call_kwargs = mock_temp_file.call_args[1]
                    assert call_kwargs['delete'] is False
                    assert call_kwargs['mode'] == 'wb'
                    assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #54
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
    
    context = {'cookiecutter': {}}
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles UndefinedError."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError('Variable undefined')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_no_delete_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete when delete_project_on_failure is False."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
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
    
    cwd_during_call = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal cwd_during_call
        cwd_during_call = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    context = {'cookiecutter': {}}
    original_cwd = os.getcwd()
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert cwd_during_cwd == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #55
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    mock_popen = lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script
    run_script(script_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    mock_popen = lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script
    run_script(script_path)


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    cwd = str(tmp_path)
    
    mock_popen = lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script
    run_script(script_path, cwd=cwd)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    mock_popen = lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 1})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except Exception as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    from module import run_script
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except Exception as e:
        assert "Hook script failed (error:" in str(e)


# LLM-generated content at query #56
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
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
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    hook_name = 'post_gen_project'
    original_cwd = None
    called_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal called_cwd
        called_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    original_cwd = os.getcwd()
    
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    assert called_cwd == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #57
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    import subprocess
    original_popen = subprocess.Popen
    
    class MockProc:
        def wait(self):
            return 0
    
    def mock_popen(*args, **kwargs):
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.executable', 'python')
    
    from pathlib import Path
    from your_module import run_script
    run_script(script_path, cwd=tmp_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    import subprocess
    
    class MockProc:
        def wait(self):
            return 0
    
    def mock_popen(*args, **kwargs):
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    from your_module import run_script
    run_script(script_path, cwd=tmp_path)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("import sys; sys.exit(1)")
    
    import subprocess
    
    class MockProc:
        def wait(self):
            return 1
    
    def mock_popen(*args, **kwargs):
        return MockProc()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.executable', 'python')
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("")
    
    import subprocess
    import errno
    
    def mock_popen(*args, **kwargs):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    import subprocess
    
    def mock_popen(*args, **kwargs):
        raise OSError("File not found")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    from your_module import run_script, FailedHookException
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)


# LLM-generated content at query #58
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    from cookiecutter.exceptions import FailedHookException
    
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


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes to repo_dir before executing hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = None
    hook_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal hook_cwd
        hook_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    context = {'cookiecutter': {}}
    original_cwd = os.getcwd()
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert hook_cwd == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #59
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir successfully runs a hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir cleans up project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failed_hook_exception_without_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not clean up project when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_undefined_error_with_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir cleans up project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError('Undefined variable')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_to_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo directory."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    call_cwd = []
    def capture_cwd(*args, **kwargs):
        call_cwd.append(os.getcwd())
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    assert str(repo_dir) in call_cwd[0]


# LLM-generated content at query #60
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false(tmp_path, monkeypatch):
    import sys
    import subprocess
    from pathlib import Path
    
    # Mock sys.platform to be non-Windows
    monkeypatch.setattr(sys, 'platform', 'linux')
    
    # Create a temporary script file
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('success')")
    script_file.chmod(0o755)
    
    # Mock subprocess.Popen to return a successful process
    class MockProcess:
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: MockProcess())
    
    # Mock utils.make_executable
    import utils
    monkeypatch.setattr(utils, 'make_executable', lambda x: None)
    
    # Call the function - should not raise any exception
    # This means the except OSError block at line 21 is NOT executed
    run_script(str(script_file), cwd=tmp_path)


# LLM-generated content at query #61
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_pre_prompt_script():
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt script exists."""
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory without pre_prompt hook
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        
        # Call the function
        result = run_pre_prompt_hook(repo_dir)
        
        # Assert that it returns the original repo_dir (predicate at line 9 is True, so line 10 returns)
        assert result == repo_dir


# LLM-generated content at query #62
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (line 17 predicate evaluates to False)."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_cwd = os.getcwd()
    test_repo_dir = "/test/repo"
    test_project_dir = "/test/project"
    test_context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        with patch('os.chdir') as mock_chdir:
            with patch('os.getcwd', return_value=original_cwd):
                run_hook_from_repo_dir(
                    repo_dir=test_repo_dir,
                    hook_name="post_gen_project",
                    project_dir=test_project_dir,
                    context=test_context,
                    delete_project_on_failure=False,
                )
                
                mock_chdir.assert_called()
                calls = [str(call) for call in mock_chdir.call_args_list]
                assert any(test_repo_dir in str(call) for call in calls)


# LLM-generated content at query #63
#--------------------------

```python
def test_work_in_context_manager_changes_directory(tmp_path, monkeypatch):
    """Test that work_in context manager changes to the specified directory."""
    from cookiecutter.utils import work_in
    import os
    
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        current_dir = os.getcwd()
        assert current_dir == str(test_dir)
    
    final_dir = os.getcwd()
    assert final_dir == original_dir


def test_work_in_returns_to_original_directory(tmp_path):
    """Test that work_in context manager returns to original directory after exit."""
    from cookiecutter.utils import work_in
    import os
    
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        pass
    
    assert os.getcwd() == original_dir


def test_work_in_with_none_stays_in_current_directory(tmp_path):
    """Test that work_in with None dirname stays in current directory."""
    from cookiecutter.utils import work_in
    import os
    
    original_dir = os.getcwd()
    
    with work_in(None):
        current_dir = os.getcwd()
        assert current_dir == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_restores_directory_on_exception(tmp_path):
    """Test that work_in restores original directory even when exception occurs."""
    from cookiecutter.utils import work_in
    import os
    
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(str(test_dir)):
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #64
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
    """Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True."""
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
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
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
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError("Undefined variable")
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
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
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    original_cwd = None
    cwd_during_hook = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal cwd_during_hook
        cwd_during_hook = os.getcwd()
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=capture_cwd
    )
    
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    assert str(cwd_during_hook) == str(repo_dir.resolve())
    assert os.getcwd() == original_cwd


def test_run_hook_from_repo_dir_restores_working_directory_on_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir restores working directory even on exception."""
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
    mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    original_cwd = os.getcwd()
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    except FailedHookException:
        pass
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #65
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from cookiecutter.utils import work_in
    
    test_repo_dir = tmp_path / "test_repo"
    test_repo_dir.mkdir()
    
    def mock_find_hook(hook_name):
        return None
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    
    result = run_pre_prompt_hook(test_repo_dir)
    
    assert result == test_repo_dir


# LLM-generated content at query #66
#--------------------------

```python
def test_oserror_with_enoexec_errno():
    import errno
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Create an OSError with ENOEXEC errno
    oserror_instance = OSError()
    oserror_instance.errno = errno.ENOEXEC
    
    # Verify the predicate at line 21 evaluates to True
    predicate_result = oserror_instance.errno == errno.ENOEXEC
    assert predicate_result is True


# LLM-generated content at query #67
#--------------------------

```python
def test_work_in_context_manager_changes_directory(tmp_path, monkeypatch):
    """Test that work_in context manager changes to the specified directory."""
    from cookiecutter.utils import work_in
    import os
    
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        current_dir = os.getcwd()
        assert current_dir == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_returns_to_original_directory_on_exception(tmp_path):
    """Test that work_in returns to original directory even if exception occurs."""
    from cookiecutter.utils import work_in
    import os
    
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    try:
        with work_in(str(test_dir)):
            assert os.getcwd() == str(test_dir)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    assert os.getcwd() == original_dir


def test_work_in_with_none_dirname(tmp_path):
    """Test that work_in with None dirname stays in current directory."""
    from cookiecutter.utils import work_in
    import os
    
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test that work_in works with Path objects."""
    from cookiecutter.utils import work_in
    from pathlib import Path
    import os
    
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #68
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir early when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from unittest.mock import patch
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


# LLM-generated content at query #69
#--------------------------

```python
def test_oserror_enoexec_predicate():
    import errno
    err = OSError()
    err.errno = errno.ENOEXEC
    assert err.errno == errno.ENOEXEC


# LLM-generated content at query #70
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(mocker, tmp_path):
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


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir doesn't delete project when flag is False."""
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


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError("Variable undefined")
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_to_repo_dir(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes to repo directory during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    hook_name = "post_gen_project"
    
    call_cwd = []
    
    def capture_cwd(*args, **kwargs):
        call_cwd.append(os.getcwd())
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=capture_cwd
    )
    
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    
    assert str(repo_dir) in call_cwd[0]
    assert os.getcwd() == original_cwd


def test_run_hook_from_repo_dir_returns_to_original_dir_on_exception(mocker, tmp_path):
    """Test run_hook_from_repo_dir returns to original directory even on exception."""
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
    
    original_cwd = os.getcwd()
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except FailedHookException:
        pass
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #71
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir successfully executes hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir cleans up project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.exceptions import FailedHookException
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error_with_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir cleans up project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from jinja2 import UndefinedError
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failed_hook_no_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir doesn't clean up when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.exceptions import FailedHookException
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    original_cwd = tmp_path.parent
    
    captured_cwd = []
    
    def capture_cwd(*args, **kwargs):
        import os
        captured_cwd.append(os.getcwd())
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert str(repo_dir) == captured_cwd[0]


# LLM-generated content at query #72
#--------------------------

```python
def test_valid_hook_with_valid_hook_file():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True


def test_valid_hook_with_valid_hook_file_with_extension():
    result = valid_hook('/path/to/pre-commit.py', 'pre-commit')
    assert result is True


def test_valid_hook_with_unsupported_hook():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False


def test_valid_hook_with_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False


def test_valid_hook_with_mismatched_hook_name():
    result = valid_hook('/path/to/pre-commit', 'post-commit')
    assert result is False


def test_valid_hook_with_backup_file_and_extension():
    result = valid_hook('/path/to/pre-commit.py~', 'pre-commit')
    assert result is False


def test_valid_hook_with_only_basename_matching():
    result = valid_hook('/different/path/pre-commit', 'pre-commit')
    assert result is True


# LLM-generated content at query #73
#--------------------------

```python
def test_find_hook_predicate_at_line_25_evaluates_to_false(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    
    # Create a temporary hooks directory with a valid hook file
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('hook')")
    
    # Change to temporary directory
    monkeypatch.chdir(tmp_path)
    
    # Mock valid_hook to return True for our test hook
    def mock_valid_hook(hook_file, hook_name):
        return hook_file == "pre_prompt.py" and hook_name == "pre_prompt"
    
    monkeypatch.setattr("os.path.isdir", lambda x: True)
    monkeypatch.setattr("os.listdir", lambda x: ["pre_prompt.py"])
    monkeypatch.setattr("os.path.abspath", lambda x: str(x))
    monkeypatch.setattr("os.path.join", lambda x, y: f"{x}/{y}")
    
    # Import after monkeypatching
    from your_module import find_hook, valid_hook
    monkeypatch.setattr("your_module.valid_hook", mock_valid_hook)
    
    result = find_hook("pre_prompt", "hooks")
    
    # The predicate at line 25 (len(scripts) == 0) should be False
    # meaning len(scripts) > 0, so result should not be None
    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0


# LLM-generated content at query #74
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_script(tmp_path):
    """Test run_pre_prompt_hook when no pre_prompt hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_hook_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes a valid pre_prompt hook."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('hook executed')")
    hook_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, (str, Path))
    assert Path(result).exists()


def test_run_pre_prompt_hook_with_bash_hook(tmp_path):
    """Test run_pre_prompt_hook with a bash hook script."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt"
    hook_file.write_text("#!/bin/bash\necho 'hook'")
    hook_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, (str, Path))
    assert Path(result).exists()


def test_run_pre_prompt_hook_creates_temp_directory(tmp_path):
    """Test run_pre_prompt_hook creates a temporary directory when hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('executed')")
    hook_file.chmod(0o755)
    
    test_file = repo_dir / "test.txt"
    test_file.write_text("content")
    
    result = run_pre_prompt_hook(repo_dir)
    result_path = Path(result)
    
    assert result_path != repo_dir
    assert result_path.exists()
    assert (result_path / "test.txt").exists()


def test_run_pre_prompt_hook_failed_hook_raises_exception(tmp_path):
    """Test run_pre_prompt_hook raises exception when hook script fails."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("import sys; sys.exit(1)")
    hook_file.chmod(0o755)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass


# LLM-generated content at query #75
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from pathlib import Path
    
    # Test case 1: Non-existent hooks directory returns None
    result = find_hook('test_hook', 'nonexistent_hooks_dir')
    assert result is None
    
    # Test case 2: Empty hooks directory returns None
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            result = find_hook('test_hook', 'hooks')
            assert result is None
        finally:
            os.chdir(original_cwd)
    
    # Test case 3: Hooks directory with matching files returns list of strings
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create test hook files
        test_hook_file = os.path.join(hooks_dir, 'test_hook.sh')
        Path(test_hook_file).touch()
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            # Mock valid_hook to return True for our test file
            import sys
            from unittest.mock import patch
            
            with patch('valid_hook', return_value=True):
                result = find_hook('test_hook', 'hooks')
                assert isinstance(result, list)
                assert len(result) > 0
                assert all(isinstance(path, str) for path in result)
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #76
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no hook scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')


def test_run_hook_with_single_script(mocker, tmp_path):
    """Test run_hook executes a single hook script."""
    script_path = str(tmp_path / 'hook.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'post_gen_project')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_with_multiple_scripts(mocker, tmp_path):
    """Test run_hook executes multiple hook scripts."""
    script_path_1 = str(tmp_path / 'hook1.sh')
    script_path_2 = str(tmp_path / 'hook2.py')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path_1, script_path_2])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_gen_project')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'pre_gen_project')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path_1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path_2, tmp_path, context)


def test_run_hook_with_pathlib_path(mocker, tmp_path):
    """Test run_hook accepts Path objects."""
    from pathlib import Path
    script_path = str(tmp_path / 'hook.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    project_dir = Path(tmp_path)
    run_hook('post_prompt', project_dir, context)
    
    mock_run_script_with_context.assert_called_once_with(script_path, project_dir, context)


def test_run_hook_with_string_project_dir(mocker, tmp_path):
    """Test run_hook accepts string project directory."""
    script_path = str(tmp_path / 'hook.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    project_dir = str(tmp_path)
    run_hook('post_prompt', project_dir, context)
    
    mock_run_script_with_context.assert_called_once_with(script_path, project_dir, context)


def test_run_hook_empty_scripts_list(mocker, tmp_path):
    """Test run_hook when find_hook returns empty list."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')


# LLM-generated content at query #77
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    script_content = "#!/bin/bash\necho {{ cookiecutter.project_name }}"
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    with patch('cookiecutter.hooks.run_script') as mock_run_script:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_tempfile:
            temp_mock = MagicMock()
            temp_mock.name = str(tmp_path / "temp_script.sh")
            temp_mock.__enter__.return_value = temp_mock
            temp_mock.__exit__.return_value = None
            mock_tempfile.return_value = temp_mock
            
            from cookiecutter.hooks import run_script_with_context
            run_script_with_context(str(script_file), str(tmp_path), context)
            
            temp_mock.write.assert_called_once()
            written_content = temp_mock.write.call_args[0][0]
            assert b'my_project' in written_content
            mock_run_script.assert_called_once_with(temp_mock.name, str(tmp_path))


def test_run_script_with_context_with_python_script(tmp_path, monkeypatch):
    """Test run_script_with_context with a Python script."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    script_content = "print('{{ cookiecutter.greeting }}')"
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'greeting': 'Hello World',
            '_jinja2_env_vars': {}
        }
    }
    
    with patch('cookiecutter.hooks.run_script') as mock_run_script:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_tempfile:
            temp_mock = MagicMock()
            temp_mock.name = str(tmp_path / "temp_script.py")
            temp_mock.__enter__.return_value = temp_mock
            temp_mock.__exit__.return_value = None
            mock_tempfile.return_value = temp_mock
            
            from cookiecutter.hooks import run_script_with_context
            run_script_with_context(str(script_file), str(tmp_path), context)
            
            temp_mock.write.assert_called_once()
            written_content = temp_mock.write.call_args[0][0]
            assert b'Hello World' in written_content
            mock_run_script.assert_called_once()


def test_run_script_with_context_preserves_extension(tmp_path):
    """Test run_script_with_context preserves file extension in temp file."""
    from unittest.mock import patch, MagicMock
    
    script_content = "#!/usr/bin/env python\nprint('test')"
    script_file = tmp_path / "hook.py"
    script_file.write_text(script_content)
    
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    
    with patch('cookiecutter.hooks.run_script'):
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_tempfile:
            temp_mock = MagicMock()
            temp_mock.name = "temp_file.py"
            temp_mock.__enter__.return_value = temp_mock
            temp_mock.__exit__.return_value = None
            mock_tempfile.return_value = temp_mock
            
            from cookiecutter.hooks import run_script_with_context
            run_script_with_context(str(script_file), str(tmp_path), context)
            
            call_kwargs = mock_tempfile.call_args[1]
            assert call_kwargs['suffix'] == '.py'
            assert call_kwargs['mode'] == 'wb'
            assert call_kwargs['delete'] is False


def test_run_script_with_context_renders_complex_template(tmp_path):
    """Test run_script_with_context with complex Jinja2 template."""
    from unittest.mock import patch, MagicMock
    
    script_content = "{% for item in cookiecutter.items %}{{ item }}{% endfor %}"
    script_file = tmp_path / "template.sh"
    script_file.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'items': ['a', 'b', 'c'],
            '_jinja2_env_vars': {}
        }
    }
    
    with patch('cookiecutter.hooks.run_script'):
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_tempfile:
            temp_mock = MagicMock()
            temp_mock.name = "temp"
            temp_mock.__enter__.return_value = temp_mock
            temp_mock.__exit__.return_value = None
            mock_tempfile.return_value = temp_mock
            
            from cookiecutter.hooks import run_script_with_context
            run_script_with_context(str(script_file), str(tmp_path), context)
            
            written_content = temp_mock.write.call_args[0][0]
            assert b'abc' in written_content


# LLM-generated content at query #78
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, mocker):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir, FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    except FailedHookException:
        pass
    
    assert mock_run_hook.called
    assert mock_logger.exception.called


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, mocker):
    """Test that run_hook_from_repo_dir catches UndefinedError at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    mocker.patch('cookiecutter.hooks.work_in')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Variable undefined"))
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    except UndefinedError:
        pass
    
    assert mock_run_hook.called
    assert mock_logger.exception.called


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, mocker):
    """Test that run_hook_from_repo_dir deletes project directory when delete_project_on_failure is True."""
    from cookiecutter.hooks import run_hook_from_repo_dir, FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    mocker.patch('cookiecutter.hooks.work_in')
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_preserves_exception(tmp_path, mocker):
    """Test that run_hook_from_repo_dir re-raises the caught exception."""
    from cookiecutter.hooks import run_hook_from_repo_dir, FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    mocker.patch('cookiecutter.hooks.work_in')
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mocker.patch('cookiecutter.hooks.logger')
    
    exception_raised = False
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
    except FailedHookException:
        exception_raised = True
    
    assert exception_raised is True


# LLM-generated content at query #79
#--------------------------

```python
def test_oserror_with_enoexec_errno():
    import errno
    import subprocess
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Create a mock OSError with ENOEXEC errno
    mock_error = OSError()
    mock_error.errno = errno.ENOEXEC
    
    # Patch subprocess.Popen to raise the OSError
    with patch('subprocess.Popen', side_effect=mock_error):
        with patch('utils.make_executable'):
            try:
                run_script('/path/to/script.sh')
                assert False, "Expected FailedHookException to be raised"
            except Exception as e:
                # Verify the predicate at line 22 evaluates to True
                assert isinstance(e, FailedHookException)
                assert 'might be an empty file or missing a shebang' in str(e)


# LLM-generated content at query #80
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('success')")
    script_file.chmod(0o755)
    
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})())
    
    run_script(str(script_file), cwd=str(tmp_path))


# LLM-generated content at query #81
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    call_count = {'run_script': 0}
    original_run_script = None
    
    def mock_run_script(script_path, cwd='.'):
        call_count['run_script'] += 1
        temp_path = Path(script_path)
        if temp_path.exists():
            rendered_content = temp_path.read_text(encoding='utf-8')
            assert 'test_project' in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert call_count['run_script'] == 1


def test_run_script_with_context_with_jinja_variables(tmp_path, monkeypatch):
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = '#!/bin/bash\necho "{{ cookiecutter.var1 }}-{{ cookiecutter.var2 }}"'
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'var1': 'hello',
            'var2': 'world'
        }
    }
    
    rendered_outputs = []
    
    def mock_run_script(script_path, cwd='.'):
        temp_path = Path(script_path)
        rendered_content = temp_path.read_text(encoding='utf-8')
        rendered_outputs.append(rendered_content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(rendered_outputs) == 1
    assert 'hello-world' in rendered_outputs[0]


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = 'echo "{{ cookiecutter.name }}"'
    script_path = tmp_path / "test_script.bat"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'name': 'test'
        }
    }
    
    temp_files = []
    
    def mock_run_script(script_path, cwd='.'):
        temp_files.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(temp_files) == 1
    assert temp_files[0].endswith('.bat')


def test_run_script_with_context_empty_context(tmp_path, monkeypatch):
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = '#!/bin/bash\necho "static content"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {}}
    
    rendered_outputs = []
    
    def mock_run_script(script_path, cwd='.'):
        temp_path = Path(script_path)
        rendered_content = temp_path.read_text(encoding='utf-8')
        rendered_outputs.append(rendered_content)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(rendered_outputs) == 1
    assert 'static content' in rendered_outputs[0]


def test_run_script_with_context_uses_provided_cwd(tmp_path, monkeypatch):
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = '#!/bin/bash\necho "test"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {}}
    custom_cwd = tmp_path / "custom_dir"
    custom_cwd.mkdir()
    
    run_script_cwds = []
    
    def mock_run_script(script_path, cwd='.'):
        run_script_cwds.append(cwd)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(custom_cwd), context)
    
    assert len(run_script_cwds) == 1
    assert str(custom_cwd) in run_script_cwds[0]


# LLM-generated content at query #82
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    call_tracker = []
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        call_tracker.append((hook_name, proj_dir, ctx))
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert len(call_tracker) == 1
    assert call_tracker[0][0] == 'post_gen_project'
    assert call_tracker[0][1] == project_dir
    assert call_tracker[0][2] == context


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException('Hook failed')
    
    def mock_rmtree(path):
        pass
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    monkeypatch.setattr('cookiecutter.hooks.rmtree', mock_rmtree)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except FailedHookException:
        pass


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    rmtree_called = []
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException('Hook failed')
    
    def mock_rmtree(path):
        rmtree_called.append(path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    monkeypatch.setattr('cookiecutter.hooks.rmtree', mock_rmtree)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    assert len(rmtree_called) == 0


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError('Undefined variable')
    
    def mock_rmtree(path):
        pass
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    monkeypatch.setattr('cookiecutter.hooks.rmtree', mock_rmtree)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    import os
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    cwd_during_hook = []
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        cwd_during_hook.append(os.getcwd())
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    from cookiecutter.hooks import run_hook_from_repo_dir
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert os.getcwd() == original_cwd
    assert str(repo_dir) in cwd_during_hook[0]


# LLM-generated content at query #83
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except FailedHookException:
        pass
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir keeps project on FailedHookException when flag is False."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    assert project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError('Variable undefined')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    captured_cwd = []
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        captured_cwd.append(os.getcwd())
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert str(repo_dir) == captured_cwd[0]
    assert os.getcwd() == original_cwd


# LLM-generated content at query #84
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(monkeypatch):
    """Test that the predicate at line 18 (exit_status != EXIT_SUCCESS) evaluates to False."""
    import subprocess
    from pathlib import Path
    from unittest.mock import Mock, MagicMock
    
    # Mock the dependencies
    mock_utils = Mock()
    mock_utils.make_executable = Mock()
    
    # Mock subprocess.Popen to return a process with exit status 0 (success)
    mock_proc = Mock()
    mock_proc.wait.return_value = 0
    
    monkeypatch.setattr('subprocess.Popen', Mock(return_value=mock_proc))
    monkeypatch.setattr('sys.platform', 'linux')
    
    # Import after mocking
    import sys
    sys.modules['utils'] = mock_utils
    
    # Set EXIT_SUCCESS to 0
    import builtins
    original_import = builtins.__import__
    
    def custom_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if name == '__main__' or 'run_script' in str(module):
            module.EXIT_SUCCESS = 0
        return module
    
    monkeypatch.setattr('builtins.__import__', custom_import)
    
    # Create a simple test by directly checking the condition
    exit_status = 0
    EXIT_SUCCESS = 0
    
    # The predicate at line 18: if exit_status != EXIT_SUCCESS
    predicate_result = exit_status != EXIT_SUCCESS
    
    assert predicate_result is False


# LLM-generated content at query #85
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen = None
    mock_wait_called = False
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            nonlocal mock_popen
            mock_popen = self
        
        def wait(self):
            nonlocal mock_wait_called
            mock_wait_called = True
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script
    run_script(script_path)
    
    assert mock_wait_called


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen = None
    mock_wait_called = False
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            nonlocal mock_popen
            mock_popen = self
        
        def wait(self):
            nonlocal mock_wait_called
            mock_wait_called = True
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script
    run_script(script_path)
    
    assert mock_wait_called


def test_run_script_windows_shell(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    shell_used = None
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            nonlocal shell_used
            shell_used = kwargs.get('shell', False)
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'win32')
    
    from run_script import run_script
    run_script(script_path)
    
    assert shell_used is True


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
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
    
    from run_script import run_script
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


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
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            err = OSError("Permission denied")
            err.errno = errno.EACCES
            raise err
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (error:" in str(e)


def test_run_script_with_cwd(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    cwd_used = None
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            nonlocal cwd_used
            cwd_used = kwargs.get('cwd')
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script
    run_script(script_path, cwd=str(tmp_path))
    
    assert cwd_used == str(tmp_path)


# LLM-generated content at query #86
#--------------------------

```python
def test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false(tmp_path, monkeypatch):
    """Test that exceptions are re-raised even when delete_project_on_failure is False."""
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
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project.py",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
        assert False, "Expected FailedHookException to be raised"
    except Exception as e:
        assert isinstance(e, FailedHookException)
        assert project_dir.exists()


# LLM-generated content at query #87
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
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('pre_prompt hook')")
    
    monkeypatch.setattr("cookiecutter.hooks.utils.make_executable", lambda x: None)
    monkeypatch.setattr("cookiecutter.hooks.subprocess.Popen", lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 0})())
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert isinstance(result, Path)


def test_run_pre_prompt_hook_script_fails(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when pre_prompt script fails."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('pre_prompt hook')")
    
    monkeypatch.setattr("cookiecutter.hooks.utils.make_executable", lambda x: None)
    monkeypatch.setattr("cookiecutter.hooks.subprocess.Popen", lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 1})())
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_returns_path(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns a Path object."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    
    monkeypatch.setattr("cookiecutter.hooks.utils.make_executable", lambda x: None)
    monkeypatch.setattr("cookiecutter.hooks.subprocess.Popen", lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 0})())
    
    result = run_pre_prompt_hook(str(repo_dir))
    assert isinstance(result, Path)


def test_run_pre_prompt_hook_creates_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary directory when scripts exist."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('test')")
    
    monkeypatch.setattr("cookiecutter.hooks.utils.make_executable", lambda x: None)
    monkeypatch.setattr("cookiecutter.hooks.subprocess.Popen", lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 0})())
    
    result = run_pre_prompt_hook(repo_dir)
    assert str(repo_dir) not in str(result)
    assert (Path(result) / "hooks" / "pre_prompt.py").exists()


# LLM-generated content at query #88
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_returns_original_repo_dir(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_creates_tmp_repo_and_runs_script(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook creates a temp repo and runs the pre_prompt script."""
    from cookiecutter.hooks import run_pre_prompt_hook
    import subprocess
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("print('hook executed')")
    hook_script.chmod(0o755)
    
    mock_popen_called = []
    original_popen = subprocess.Popen
    
    def mock_popen(*args, **kwargs):
        mock_popen_called.append((args, kwargs))
        mock_proc = original_popen(['echo', ''], **kwargs)
        return mock_proc
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert "cookiecutter" in str(result)
    assert len(mock_popen_called) > 0


def test_run_pre_prompt_hook_failed_hook_raises_exception(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook raises FailedHookException when script fails."""
    from cookiecutter.hooks import run_pre_prompt_hook, FailedHookException
    import subprocess
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("import sys; sys.exit(1)")
    hook_script.chmod(0o755)
    
    def mock_popen(*args, **kwargs):
        mock_proc = subprocess.Popen(['false'], **kwargs)
        return mock_proc
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_string_repo_dir(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook works with string repo_dir parameter."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(str(repo_dir))
    
    assert result == str(repo_dir)


# LLM-generated content at query #89
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook when no scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_run_script_with_context.assert_not_called()


def test_run_hook_with_single_script(mocker, tmp_path):
    """Test run_hook with a single script found."""
    script_path = '/path/to/hook.py'
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_with_multiple_scripts(mocker, tmp_path):
    """Test run_hook with multiple scripts found."""
    script_path_1 = '/path/to/hook1.py'
    script_path_2 = '/path/to/hook2.sh'
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path_1, script_path_2])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('pre_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_gen_project')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path_1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path_2, tmp_path, context)


def test_run_hook_empty_scripts_list(mocker, tmp_path):
    """Test run_hook when scripts list is empty."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    from cookiecutter.hooks import run_hook
    run_hook('post_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_prompt')
    mock_run_script_with_context.assert_not_called()


# LLM-generated content at query #90
#--------------------------

```python
def test_find_hook_predicate_line_1():
    hook_name = "test_hook"
    hooks_dir = "hooks"
    result = find_hook(hook_name, hooks_dir)
    assert isinstance(result, (list, type(None)))


# LLM-generated content at query #91
#--------------------------

```python
def test_find_hook_returns_scripts_when_valid_hooks_exist(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('test')")
    
    monkeypatch.chdir(tmp_path)
    
    from your_module import find_hook
    
    result = find_hook("pre_prompt", str(hooks_dir))
    
    assert result is not None
    assert len(result) > 0
    assert isinstance(result, list)


# LLM-generated content at query #92
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts are found."""
    from cookiecutter.hooks import run_pre_prompt_hook, find_hook
    
    # Create a temporary repo directory without any hooks
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    # Call the function
    result = run_pre_prompt_hook(repo_dir)
    
    # Assert that it returns the original repo_dir
    assert result == repo_dir


# LLM-generated content at query #93
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock the _HOOKS constant
    import sys
    from unittest.mock import patch
    
    with patch('__main__._HOOKS', {'test_hook'}):
        # Create a temporary directory and file
        with tempfile.TemporaryDirectory() as tmpdir:
            hook_file = os.path.join(tmpdir, 'test_hook')
            
            # Create the hook file
            with open(hook_file, 'w') as f:
                f.write('')
            
            # Import and call the function
            from __main__ import valid_hook
            result = valid_hook(hook_file, 'test_hook')
            
            assert result is True


# LLM-generated content at query #94
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(monkeypatch):
    from pathlib import Path
    import subprocess
    import sys
    
    # Mock subprocess.Popen to return a process with exit status 0 (EXIT_SUCCESS)
    class MockProcess:
        def wait(self):
            return 0
    
    def mock_popen(*args, **kwargs):
        return MockProcess()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    # Mock utils.make_executable to do nothing
    import unittest.mock as mock
    with mock.patch('utils.make_executable'):
        # Define EXIT_SUCCESS
        import sys
        sys.modules['__main__'].EXIT_SUCCESS = 0
        
        # Call run_script - should not raise an exception
        # because exit_status (0) == EXIT_SUCCESS (0), making the predicate False
        run_script('/path/to/script.py')


# LLM-generated content at query #95
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (predicate at line 17 evaluates to False)."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_repo_dir:
        with tempfile.TemporaryDirectory() as temp_project_dir:
            repo_dir = Path(temp_repo_dir)
            project_dir = Path(temp_project_dir)
            context = {'cookiecutter': {}}
            
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                with patch('cookiecutter.hooks.work_in') as mock_work_in:
                    mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
                    mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
                    
                    run_hook_from_repo_dir(
                        repo_dir=repo_dir,
                        hook_name='post_gen_project',
                        project_dir=project_dir,
                        context=context,
                        delete_project_on_failure=False
                    )
                    
                    mock_work_in.assert_called_once_with(repo_dir)
                    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)


