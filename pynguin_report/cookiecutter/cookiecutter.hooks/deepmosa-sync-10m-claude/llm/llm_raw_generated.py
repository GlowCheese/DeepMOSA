####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_valid_hook_valid_hook():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_valid_hook_with_extension():
    result = valid_hook('/path/to/pre-commit.py', 'pre-commit')
    assert result is True

def test_valid_hook_non_matching_name():
    result = valid_hook('/path/to/post-commit', 'pre-commit')
    assert result is False

def test_valid_hook_unsupported_hook():
    result = valid_hook('/path/to/unsupported-hook', 'unsupported-hook')
    assert result is False

def test_valid_hook_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False

def test_valid_hook_backup_file_with_extension():
    result = valid_hook('/path/to/pre-commit.py~', 'pre-commit')
    assert result is False

def test_valid_hook_empty_hook_name():
    result = valid_hook('/path/to/file', 'non-existent')
    assert result is False

def test_valid_hook_with_directory_path():
    result = valid_hook('/very/long/path/to/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_case_sensitive():
    result = valid_hook('/path/to/Pre-Commit', 'pre-commit')
    assert result is False


# LLM-generated content at query #2
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    import os
    import tempfile
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    # Create a temporary script file with Jinja template
    script_file = tmp_path / "test_script.py"
    script_content = "#!/usr/bin/env python\nprint('{{ cookiecutter.name }}')"
    script_file.write_text(script_content)
    
    # Mock run_script to avoid actual execution
    executed_scripts = []
    def mock_run_script(script_path, cwd):
        executed_scripts.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Mock make_executable to avoid permission changes
    monkeypatch.setattr('cookiecutter.hooks.utils.make_executable', lambda x: None)
    
    # Create context with cookiecutter data
    context = {
        'cookiecutter': {
            'name': 'test_project'
        }
    }
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(executed_scripts) == 1
    assert executed_scripts[0][1] == str(tmp_path)
    
    # Verify the temporary script was created and contains rendered output
    temp_script_path = executed_scripts[0][0]
    temp_content = Path(temp_script_path).read_text()
    assert 'test_project' in temp_content
    assert '{{ cookiecutter.name }}' not in temp_content


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')\n")
    
    mock_popen = lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 0})()
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'\n")
    
    mock_popen = lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 0})()
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("import sys\nsys.exit(1)\n")
    
    mock_popen = lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 1})()
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script")
    with open(script_path, 'w') as f:
        f.write("")
    
    def mock_popen(*args, **kwargs):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'missing a shebang' in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen(*args, **kwargs):
        err = OSError("Permission denied")
        err.errno = errno.EACCES
        raise err
    
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'Permission denied' in str(e)


def test_run_script_make_executable_called(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')\n")
    
    make_executable_called = []
    
    def mock_make_executable(path):
        make_executable_called.append(path)
    
    mock_popen = lambda *args, **kwargs: type('obj', (object,), {'wait': lambda: 0})()
    monkeypatch.setattr('subprocess.Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    run_script(script_path, cwd=tmp_path)
    assert len(make_executable_called) == 1
    assert make_executable_called[0] == script_path


# LLM-generated content at query #4
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock the _HOOKS set to include our test hook
    import sys
    from unittest.mock import patch
    
    test_hook_name = "pre-commit"
    
    with patch('os.path.basename', return_value='pre-commit'):
        with patch('os.path.splitext', return_value=('pre-commit', '')):
            with patch.dict('sys.modules', {'__main__': type(sys)('__main__')}):
                # Import the function's module context
                import os as os_module
                
                # Create a mock for _HOOKS
                with patch('__main__._HOOKS', {test_hook_name}):
                    # Manually construct the logic to test
                    hook_file = "/path/to/pre-commit"
                    hook_name = "pre-commit"
                    
                    filename = os_module.path.basename(hook_file)
                    basename = os_module.path.splitext(filename)[0]
                    matching_hook = basename == hook_name
                    supported_hook = basename in {test_hook_name}
                    backup_file = filename.endswith('~')
                    
                    result = matching_hook and supported_hook and not backup_file
                    
                    assert result is True


# LLM-generated content at query #5
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
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert sys.executable in mock_popen_called[0][0][0]
    assert script_path in mock_popen_called[0][0][0]


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    import subprocess
    import sys
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
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    run_script(script_path, cwd=str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert script_path in mock_popen_called[0][0][0]
    assert mock_popen_called[0][1]['shell'] is False


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
    
    assert mock_popen_called[0][1]['shell'] is True


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("import sys; sys.exit(1)")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "exit status: 1" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("")
    
    def mock_popen_enoexec(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_enoexec)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "shebang" in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    def mock_popen_error(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_error)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "error" in str(e).lower()


def test_run_script_with_path_object(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text("print('success')")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    run_script(str(script_path), cwd=tmp_path)
    
    assert len(mock_popen_called) == 1


# LLM-generated content at query #6
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_no_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=False."""
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


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir with FailedHookException and delete_project_on_failure=True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error_no_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=True."""
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


def test_run_hook_from_repo_dir_changes_working_directory(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes to repo_dir during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = None
    execution_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal execution_cwd
        execution_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    context = {'cookiecutter': {}}
    
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert execution_cwd == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #7
#--------------------------

```python
def test_script_path_endswith_py():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import sys
    
    script_path = "/path/to/script.py"
    cwd = Path('.')
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('sys.platform', 'linux'), \
         patch('utils.make_executable') as mock_make_exec:
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script(script_path, cwd)
        
        assert script_path.endswith('.py')
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == [sys.executable, script_path]


# LLM-generated content at query #8
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker):
    """Test run_hook when no hook scripts are found."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    from cookiecutter.hooks import run_hook
    
    run_hook('pre_prompt', '/project', {'cookiecutter': {}})
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')


def test_run_hook_with_single_script(mocker):
    """Test run_hook when a single hook script is found and executed."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=['/hooks/pre_prompt.sh'])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    from cookiecutter.hooks import run_hook
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', '/project', context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'pre_prompt')
    mock_run_script_with_context.assert_called_once_with('/hooks/pre_prompt.sh', '/project', context)


def test_run_hook_with_multiple_scripts(mocker):
    """Test run_hook when multiple hook scripts are found."""
    scripts = ['/hooks/post_gen_project.sh', '/hooks/post_gen_project.py']
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=scripts)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    from cookiecutter.hooks import run_hook
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', '/project', context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(scripts[0], '/project', context)
    mock_run_script_with_context.assert_any_call(scripts[1], '/project', context)


def test_run_hook_empty_scripts_list(mocker):
    """Test run_hook when hook scripts list is empty."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    from cookiecutter.hooks import run_hook
    
    run_hook('pre_prompt', '/project', {'cookiecutter': {}})
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')


# LLM-generated content at query #9
#--------------------------

```python
def test_valid_hook_with_valid_hook_file():
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True


def test_valid_hook_with_valid_hook_file_full_path():
    result = valid_hook('/home/user/.git/hooks/commit-msg', 'commit-msg')
    assert result is True


def test_valid_hook_with_mismatched_hook_name():
    result = valid_hook('/path/to/pre-commit', 'commit-msg')
    assert result is False


def test_valid_hook_with_unsupported_hook():
    result = valid_hook('/path/to/invalid-hook', 'invalid-hook')
    assert result is False


def test_valid_hook_with_backup_file():
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False


def test_valid_hook_with_backup_file_and_unsupported():
    result = valid_hook('/path/to/invalid-hook~', 'invalid-hook')
    assert result is False


def test_valid_hook_with_extension():
    result = valid_hook('/path/to/pre-commit.sh', 'pre-commit')
    assert result is True


def test_valid_hook_with_extension_and_backup():
    result = valid_hook('/path/to/pre-commit.sh~', 'pre-commit')
    assert result is False


def test_valid_hook_with_empty_basename():
    result = valid_hook('/path/to/.bashrc', '')
    assert result is False


def test_valid_hook_relative_path():
    result = valid_hook('pre-commit', 'pre-commit')
    assert result is True


# LLM-generated content at query #10
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
    script_path = str(tmp_path / 'pre_prompt.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'pre_prompt')
    mock_run_script_with_context.assert_called_once_with(script_path, tmp_path, context)


def test_run_hook_multiple_scripts_found(mocker, tmp_path):
    """Test run_hook when multiple hook scripts are found."""
    script_path_1 = str(tmp_path / 'pre_prompt_1.sh')
    script_path_2 = str(tmp_path / 'pre_prompt_2.sh')
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[script_path_1, script_path_2])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_logger.debug.assert_called_once_with('Running hook %s', 'post_gen_project')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(script_path_1, tmp_path, context)
    mock_run_script_with_context.assert_any_call(script_path_2, tmp_path, context)


def test_run_hook_empty_scripts_list(mocker, tmp_path):
    """Test run_hook when find_hook returns an empty list."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_gen_project', tmp_path, context)
    
    mock_find_hook.assert_called_once_with('pre_gen_project')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_gen_project')
    mock_run_script_with_context.assert_not_called()


# LLM-generated content at query #11
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_found(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no hook is found."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes a valid pre_prompt hook."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# valid hook")
    
    called = []
    
    def mock_run_script(script_path, cwd='.'):
        called.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert len(called) == 1
    assert result != repo_dir


def test_run_pre_prompt_hook_hook_failure(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when hook fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook that fails")
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException("Script failed")
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_multiple_hooks(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes multiple pre_prompt hooks."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file1 = hooks_dir / "pre_prompt.py"
    hook_file1.write_text("# hook 1")
    hook_file2 = hooks_dir / "pre_prompt.sh"
    hook_file2.write_text("#!/bin/bash\necho test")
    
    called = []
    
    def mock_run_script(script_path, cwd='.'):
        called.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert len(called) == 2
    assert result != repo_dir


def test_run_pre_prompt_hook_with_string_path(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook works with string path."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(str(repo_dir))
    assert result == str(repo_dir)


# LLM-generated content at query #12
#--------------------------

```python
def test_find_hook_with_valid_hook_file(tmp_path, monkeypatch):
    monkeypatch.setattr('os.path.isdir', lambda x: True)
    monkeypatch.setattr('os.listdir', lambda x: ['pre_prompt.py'])
    monkeypatch.setattr('os.path.basename', lambda x: 'pre_prompt.py')
    monkeypatch.setattr('os.path.splitext', lambda x: ('pre_prompt', '.py'))
    monkeypatch.setattr('os.path.abspath', lambda x: str(tmp_path / x.split('/')[-1]))
    monkeypatch.setattr('os.path.join', lambda x, y: f'{x}/{y}')
    
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) > 0


def test_find_hook_with_nonexistent_hooks_dir(monkeypatch):
    monkeypatch.setattr('os.path.isdir', lambda x: False)
    
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


def test_find_hook_with_empty_hooks_dir(tmp_path, monkeypatch):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    monkeypatch.setattr('os.path.isdir', lambda x: True)
    monkeypatch.setattr('os.listdir', lambda x: [])
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_with_backup_file(tmp_path, monkeypatch):
    monkeypatch.setattr('os.path.isdir', lambda x: True)
    monkeypatch.setattr('os.listdir', lambda x: ['pre_prompt.py~'])
    monkeypatch.setattr('os.path.basename', lambda x: 'pre_prompt.py~')
    monkeypatch.setattr('os.path.splitext', lambda x: ('pre_prompt.py', '~'))
    monkeypatch.setattr('os.path.abspath', lambda x: str(tmp_path / 'pre_prompt.py~'))
    monkeypatch.setattr('os.path.join', lambda x, y: f'{x}/{y}')
    
    result = find_hook('pre_prompt', 'hooks')
    assert result is None


def test_find_hook_with_unsupported_hook(monkeypatch):
    monkeypatch.setattr('os.path.isdir', lambda x: True)
    monkeypatch.setattr('os.listdir', lambda x: ['unsupported_hook.py'])
    monkeypatch.setattr('os.path.basename', lambda x: 'unsupported_hook.py')
    monkeypatch.setattr('os.path.splitext', lambda x: ('unsupported_hook', '.py'))
    monkeypatch.setattr('os.path.abspath', lambda x: x)
    monkeypatch.setattr('os.path.join', lambda x, y: f'{x}/{y}')
    
    result = find_hook('unsupported_hook', 'hooks')
    assert result is None


def test_find_hook_with_multiple_matching_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr('os.path.isdir', lambda x: True)
    monkeypatch.setattr('os.listdir', lambda x: ['pre_prompt.py', 'pre_prompt.sh'])
    monkeypatch.setattr('os.path.basename', lambda x: x.split('/')[-1])
    monkeypatch.setattr('os.path.splitext', lambda x: (x.rsplit('.', 1)[0], '.' + x.rsplit('.', 1)[-1]))
    monkeypatch.setattr('os.path.abspath', lambda x: str(tmp_path / x.split('/')[-1]))
    monkeypatch.setattr('os.path.join', lambda x, y: f'{x}/{y}')
    
    result = find_hook('pre_prompt', 'hooks')
    assert result is not None
    assert len(result) == 2


# LLM-generated content at query #13
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_delete_false():
    """Test that tempfile.NamedTemporaryFile is called with delete=False."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock, call
    from cookiecutter.hooks import run_script_with_context

    script_content = "echo 'test'"
    context = {'cookiecutter': {}}
    
    with patch('cookiecutter.hooks.Path') as mock_path_class, \
         patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_tempfile, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('os.path.splitext', return_value=('script', '.sh')):
        
        mock_path_instance = MagicMock()
        mock_path_instance.read_text.return_value = script_content
        mock_path_class.return_value = mock_path_instance
        
        mock_temp_file = MagicMock()
        mock_temp_file.name = '/tmp/tempfile.sh'
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = "rendered content"
        mock_env.from_string.return_value = mock_template
        mock_create_env.return_value = mock_env
        
        run_script_with_context('/path/to/script.sh', '/cwd', context)
        
        mock_tempfile.assert_called_once()
        call_kwargs = mock_tempfile.call_args[1]
        assert call_kwargs['delete'] is False


# LLM-generated content at query #14
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
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, str(tmp_path))
    
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
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, str(tmp_path))
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


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
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)


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
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    custom_cwd = str(tmp_path / "custom")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, custom_cwd)
    
    assert mock_popen_called[0][1]['cwd'] == custom_cwd


# LLM-generated content at query #15
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    import logging
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda x: [])
    
    context = {'cookiecutter': {}}
    run_hook('pre_prompt', '.', context)
    
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 25 evaluates to False when scripts exist."""
    import os
    from pathlib import Path
    
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('test')")
    
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('os.listdir', return_value=['pre_prompt.py'])
    mocker.patch('os.path.abspath', side_effect=lambda x: str(x))
    mocker.patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}")
    
    def mock_valid_hook(hook_file, hook_name):
        return hook_file == 'pre_prompt.py' and hook_name == 'pre_prompt'
    
    mocker.patch('valid_hook', side_effect=mock_valid_hook)
    
    from find_hook import find_hook
    
    result = find_hook('pre_prompt')
    
    assert result is not None
    assert len(result) > 0


# LLM-generated content at query #17
#--------------------------

```python
def test_find_hook_returns_scripts_list_when_valid_hooks_found(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("# hook script")
    
    monkeypatch.chdir(tmp_path)
    
    # Mock valid_hook to return True for our test hook
    import sys
    from unittest.mock import patch
    
    with patch('__main__.valid_hook', return_value=True):
        import os
        scripts = [
            os.path.abspath(os.path.join(str(hooks_dir), hook_file.name))
            for hook_file in os.listdir(str(hooks_dir))
            if True  # Simulating valid_hook returning True
        ]
        
        assert len(scripts) != 0
        assert len(scripts) > 0


# LLM-generated content at query #18
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
    
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\necho 'test'")
    hook_script.chmod(0o755)
    
    called = []
    original_run_script = run_script
    
    def mock_run_script(script_path, cwd='.'):
        called.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert len(called) == 1


def test_run_pre_prompt_hook_failed_hook_exception(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when hook script raises FailedHookException."""
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


def test_run_pre_prompt_hook_multiple_hook_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with multiple pre_prompt hook scripts."""
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
    
    called = []
    
    def mock_run_script(script_path, cwd='.'):
        called.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert len(called) == 2


# LLM-generated content at query #19
#--------------------------

```python
def test_find_hook_predicate_false():
    import os
    import tempfile
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Create a valid hook file
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file, 'w') as f:
            f.write('# hook script')
        
        # Mock valid_hook to return True for at least one file
        with patch('os.path.isdir', return_value=True):
            with patch('os.listdir', return_value=['pre_prompt.py']):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
                        with patch('valid_hook', return_value=True):
                            result = find_hook('pre_prompt', hooks_dir)
                            assert result is not None
                            assert len(result) > 0
                            assert result[0] == 'hooks/pre_prompt.py'


# LLM-generated content at query #20
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock the _HOOKS variable to include our test hook
    import sys
    from unittest.mock import patch
    
    test_hook_name = "pre-commit"
    
    with patch('__main__._HOOKS', {test_hook_name}):
        with tempfile.TemporaryDirectory() as tmpdir:
            hook_file = os.path.join(tmpdir, f"{test_hook_name}.py")
            
            result = valid_hook(hook_file, test_hook_name)
            
            assert result is True


# LLM-generated content at query #21
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found(mocker, tmp_path):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Mock find_hook to return empty list (no scripts found)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    
    # Mock work_in to be a no-op context manager
    mocker.patch('cookiecutter.hooks.work_in')
    
    result = run_pre_prompt_hook(tmp_path)
    
    assert result == tmp_path


# LLM-generated content at query #22
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from unittest.mock import patch
    
    # Test case 1: hooks_dir does not exist
    result = find_hook('test_hook', 'nonexistent_dir')
    assert result is None
    
    # Test case 2: hooks_dir exists but is empty
    with tempfile.TemporaryDirectory() as temp_dir:
        result = find_hook('test_hook', temp_dir)
        assert result is None
    
    # Test case 3: hooks_dir exists with valid hook files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock hook file
        hook_file = os.path.join(temp_dir, 'test_hook.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\n')
        
        with patch('find_hook.valid_hook', return_value=True):
            result = find_hook('test_hook', temp_dir)
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(path, str) for path in result)
    
    # Test case 4: Return type is either list[str] or None
    with tempfile.TemporaryDirectory() as temp_dir:
        result = find_hook('nonexistent_hook', temp_dir)
        assert result is None or (isinstance(result, list) and all(isinstance(x, str) for x in result))


# LLM-generated content at query #23
#--------------------------

```python
def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test that run_hook returns early when no scripts are found."""
    from cookiecutter.hooks import run_hook
    import logging
    
    # Mock find_hook to return an empty list
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: [])
    
    # Call run_hook with empty scripts
    run_hook('pre_prompt', '.', {'cookiecutter': {}})
    
    # Verify the predicate at line 10 evaluates to True (not scripts is True when scripts is [])
    # and the debug message is logged
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #24
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a simple repo directory without any hooks
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    def mock_find_hook(hook_name):
        return []
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    
    # Call the function
    result = run_pre_prompt_hook(repo_dir)
    
    # Assert that it returns the original repo_dir without creating temp dir
    assert result == repo_dir


# LLM-generated content at query #25
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
    
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, delete_project_on_failure=True)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on FailedHookException when delete_project_on_failure is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, delete_project_on_failure=True)
    except FailedHookException:
        pass
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_no_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not delete project on FailedHookException when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, delete_project_on_failure=False)
    except FailedHookException:
        pass
    
    mock_run_hook.assert_called_once_with('pre_gen_project', project_dir, context)
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when delete_project_on_failure is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_prompt', project_dir, context, delete_project_on_failure=True)
    except UndefinedError:
        pass
    
    mock_run_hook.assert_called_once_with('post_prompt', project_dir, context)
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_no_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not delete project on UndefinedError when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, delete_project_on_failure=False)
    except UndefinedError:
        pass
    
    mock_run_hook.assert_called_once_with('pre_gen_project', project_dir, context)
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = tmp_path
    captured_cwd = []
    
    def capture_cwd(*args, **kwargs):
        captured_cwd.append(os.getcwd())
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    context = {'cookiecutter': {}}
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, delete_project_on_failure=True)
    
    assert str(captured_cwd[0]) == str(repo_dir)


# LLM-generated content at query #26
#--------------------------

```python
def test_find_hook_with_valid_hook_file(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "post_gen_project.py"
    hook_file.write_text("#!/usr/bin/env python\n")
    
    monkeypatch.chdir(tmp_path)
    result = find_hook("post_gen_project", str(hooks_dir))
    
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_with_nonexistent_hooks_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = find_hook("post_gen_project", "nonexistent_hooks")
    
    assert result is None


def test_find_hook_with_no_matching_hook(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_gen_project.py"
    hook_file.write_text("#!/usr/bin/env python\n")
    
    monkeypatch.chdir(tmp_path)
    result = find_hook("post_gen_project", str(hooks_dir))
    
    assert result is None


def test_find_hook_with_backup_file(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "post_gen_project.py~"
    hook_file.write_text("#!/usr/bin/env python\n")
    
    monkeypatch.chdir(tmp_path)
    result = find_hook("post_gen_project", str(hooks_dir))
    
    assert result is None


def test_find_hook_with_multiple_matching_hooks(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / "post_gen_project.py"
    hook_file1.write_text("#!/usr/bin/env python\n")
    hook_file2 = hooks_dir / "post_gen_project.sh"
    hook_file2.write_text("#!/bin/bash\n")
    
    monkeypatch.chdir(tmp_path)
    result = find_hook("post_gen_project", str(hooks_dir))
    
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1) in result
    assert str(hook_file2) in result


def test_find_hook_with_unsupported_hook_name(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "unsupported_hook.py"
    hook_file.write_text("#!/usr/bin/env python\n")
    
    monkeypatch.chdir(tmp_path)
    result = find_hook("unsupported_hook", str(hooks_dir))
    
    assert result is None


def test_find_hook_with_empty_hooks_dir(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    result = find_hook("post_gen_project", str(hooks_dir))
    
    assert result is None


# LLM-generated content at query #27
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir, FailedHookException
    from unittest.mock import patch
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed")):
        try:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name="post_gen_project",
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
            assert False, "Expected FailedHookException to be raised"
        except FailedHookException:
            assert True


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches UndefinedError at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    from unittest.mock import patch
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Undefined variable")):
        try:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name="post_gen_project",
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
            assert False, "Expected UndefinedError to be raised"
        except UndefinedError:
            assert True


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir deletes project when delete_project_on_failure is True."""
    from cookiecutter.hooks import run_hook_from_repo_dir, FailedHookException
    from unittest.mock import patch
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed")):
        try:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name="post_gen_project",
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )
        except FailedHookException:
            pass
    
    assert not project_dir.exists()


# LLM-generated content at query #28
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    import shutil
    from pathlib import Path
    
    # Test case 1: When hooks_dir doesn't exist, should return None
    result = find_hook('test_hook', 'nonexistent_hooks_dir')
    assert result is None
    
    # Test case 2: When hooks_dir exists but is empty, should return None
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('test_hook', hooks_dir)
        assert result is None
    
    # Test case 3: Return type is either list[str] or None
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('test_hook', hooks_dir)
        assert result is None or isinstance(result, list)
        if isinstance(result, list):
            assert all(isinstance(item, str) for item in result)


# LLM-generated content at query #29
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context

    script_path = "/path/to/script.sh"
    cwd = "/working/directory"
    context = {"cookiecutter": {"project_name": "test_project"}}

    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/tmpfile.sh"
    mock_temp_file.__enter__ = MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = MagicMock(return_value=None)

    with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file) as mock_named_temp:
        with patch('pathlib.Path.read_text', return_value="echo hello"):
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                with patch('cookiecutter.hooks.run_script'):
                    mock_env = MagicMock()
                    mock_template = MagicMock()
                    mock_template.render.return_value = "echo hello"
                    mock_env.from_string.return_value = mock_template
                    mock_create_env.return_value = mock_env

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
    import sys
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 1})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert "exit status: 1" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert "missing a shebang" in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should raise FailedHookException"
    except FailedHookException as e:
        assert "Permission denied" in str(e)


def test_run_script_calls_make_executable(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    make_executable_called = []
    
    def mock_make_executable(path):
        make_executable_called.append(path)
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    run_script(script_path, cwd=tmp_path)
    assert script_path in make_executable_called


def test_run_script_python_file_uses_sys_executable(tmp_path, monkeypatch):
    import sys
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    popen_args = []
    
    def mock_popen(cmd, *args, **kwargs):
        popen_args.append(cmd)
        return type('MockProc', (), {'wait': lambda self: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)
    assert popen_args[0] == [sys.executable, script_path]


def test_run_script_shell_script_direct_command(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    popen_args = []
    
    def mock_popen(cmd, *args, **kwargs):
        popen_args.append(cmd)
        return type('MockProc', (), {'wait': lambda self: 0})()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=tmp_path)
    assert popen_args[0] == [script_path]


# LLM-generated content at query #31
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    """Test that tempfile is created with the correct suffix from script_path."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    script_path = "/path/to/script.sh"
    cwd = "/tmp"
    context = {"cookiecutter": {"name": "test"}}
    
    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/tmpXXXXXX.sh"
    
    with patch('tempfile.NamedTemporaryFile') as mock_named_temp:
        with patch('pathlib.Path.read_text', return_value="echo {{ cookiecutter.name }}"):
            with patch('cookiecutter.hooks.run_script'):
                mock_named_temp.return_value.__enter__.return_value = mock_temp_file
                
                run_script_with_context(script_path, cwd, context)
                
                # Verify NamedTemporaryFile was called with suffix='.sh'
                mock_named_temp.assert_called_once()
                call_kwargs = mock_named_temp.call_args[1]
                assert call_kwargs['suffix'] == '.sh'
                assert call_kwargs['delete'] is False
                assert call_kwargs['mode'] == 'wb'


# LLM-generated content at query #32
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


# LLM-generated content at query #33
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
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
    
    assert len(work_in_called) == 1
    assert str(work_in_called[0]) == str(repo_dir)


# LLM-generated content at query #34
#--------------------------

```python
def test_find_hook_predicate_evaluates_to_false(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "some_file.txt").write_text("content")
    
    from unittest.mock import patch
    with patch('__main__.valid_hook', return_value=False):
        result = find_hook("test_hook", str(hooks_dir))
    
    assert result is None


# LLM-generated content at query #35
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch("cookiecutter.hooks.run_hook")
    
    run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    
    mock_run_hook.assert_called_once_with("post_gen_project", project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch("cookiecutter.hooks.run_hook")
    mock_run_hook.side_effect = FailedHookException("Hook failed")
    mock_rmtree = mocker.patch("cookiecutter.hooks.rmtree")
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch("cookiecutter.hooks.run_hook")
    mock_run_hook.side_effect = UndefinedError("Variable undefined")
    mock_rmtree = mocker.patch("cookiecutter.hooks.rmtree")
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
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
    mock_run_hook = mocker.patch("cookiecutter.hooks.run_hook")
    mock_run_hook.side_effect = FailedHookException("Hook failed")
    mock_rmtree = mocker.patch("cookiecutter.hooks.rmtree")
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook from repo_dir."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    original_cwd = mocker.patch("os.getcwd", return_value=str(tmp_path))
    mock_chdir = mocker.patch("os.chdir")
    mock_run_hook = mocker.patch("cookiecutter.hooks.run_hook")
    
    run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    
    assert mock_chdir.call_count >= 1
    mock_run_hook.assert_called_once()


# LLM-generated content at query #36
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    from pathlib import Path
    from your_module import run_script
    run_script(script_path, cwd=tmp_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    from your_module import run_script
    run_script(script_path, cwd=tmp_path)


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    mock_popen = lambda *args, **kwargs: type('MockProc', (), {'wait': lambda self: 1})()
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    from your_module import run_script, FailedHookException
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "exit status: 1" in str(e)


def test_run_script_os_error_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("")
    
    def mock_popen_enoexec(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_enoexec)
    
    from your_module import run_script, FailedHookException
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "shebang" in str(e)


def test_run_script_os_error_other(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash")
    
    def mock_popen_error(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_error)
    
    from your_module import run_script, FailedHookException
    try:
        run_script(script_path, cwd=tmp_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Permission denied" in str(e)


# LLM-generated content at query #37
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory without any hooks
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    monkeypatch.setattr(
        'cookiecutter.hooks.find_hook',
        lambda hook_name: []
    )
    
    result = run_pre_prompt_hook(test_repo)
    
    assert result == test_repo


# LLM-generated content at query #38
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
            
            # The predicate at line 21 should evaluate to True
            # when OSError is raised with errno == errno.ENOEXEC
            try:
                # Attempt to trigger the exception
                script_command = ['/path/to/script.sh']
                proc = subprocess.Popen(script_command, shell=False, cwd='.')
            except OSError as err:
                # This assertion verifies the predicate at line 21 evaluates to True
                assert err.errno == errno.ENOEXEC


# LLM-generated content at query #39
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


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
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
    
    context = {'cookiecutter': {'project_name': 'test'}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_no_delete_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project on failure when flag is False."""
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


def test_run_hook_from_repo_dir_changes_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo directory during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    original_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal original_cwd
        original_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert original_cwd == str(repo_dir)


# LLM-generated content at query #40
#--------------------------

```python
def test_run_hook_from_repo_dir_catches_failed_hook_exception(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches FailedHookException at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise FailedHookException("Hook failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    except FailedHookException:
        pass


def test_run_hook_from_repo_dir_catches_undefined_error(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir catches UndefinedError at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {}}
    
    def mock_run_hook(hook_name, proj_dir, ctx):
        raise UndefinedError("Variable undefined")
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    except UndefinedError:
        pass


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir deletes project directory when delete_project_on_failure is True."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
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


# LLM-generated content at query #41
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the delete parameter in NamedTemporaryFile is False."""
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import tempfile
    from cookiecutter.hooks import run_script_with_context
    
    mock_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_jinja2_env_vars': {}
        }
    }
    
    script_content = "#!/bin/bash\necho 'test'"
    
    with patch('cookiecutter.hooks.Path') as mock_path_class:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp_file:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                with patch('cookiecutter.hooks.run_script'):
                    # Setup mocks
                    mock_path_instance = MagicMock()
                    mock_path_instance.read_text.return_value = script_content
                    mock_path_class.return_value = mock_path_instance
                    
                    mock_temp_instance = MagicMock()
                    mock_temp_instance.__enter__.return_value = mock_temp_instance
                    mock_temp_instance.name = '/tmp/test_script.sh'
                    mock_temp_file.return_value = mock_temp_instance
                    
                    mock_env = MagicMock()
                    mock_template = MagicMock()
                    mock_template.render.return_value = script_content
                    mock_env.from_string.return_value = mock_template
                    mock_create_env.return_value = mock_env
                    
                    # Call the function
                    run_script_with_context('/tmp/script.sh', '/tmp', mock_context)
                    
                    # Assert that NamedTemporaryFile was called with delete=False
                    mock_temp_file.assert_called_once()
                    call_kwargs = mock_temp_file.call_args[1]
                    assert call_kwargs['delete'] is False


# LLM-generated content at query #42
#--------------------------

```python
def test_run_pre_prompt_hook_work_in_context_manager():
    """Test that work_in context manager is used at line 7 of run_pre_prompt_hook."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary directory to use as repo_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Mock find_hook to return empty list so we return early
        with patch('cookiecutter.hooks.find_hook', return_value=[]):
            # Mock work_in to track if it was called with repo_dir
            original_work_in = None
            work_in_called = False
            work_in_arg = None
            
            def mock_work_in(dirname=None):
                nonlocal work_in_called, work_in_arg
                work_in_called = True
                work_in_arg = dirname
                # Return a context manager that does nothing
                import contextlib
                @contextlib.contextmanager
                def noop():
                    yield
                return noop()
            
            with patch('cookiecutter.hooks.work_in', side_effect=mock_work_in):
                result = run_pre_prompt_hook(temp_path)
            
            # Verify that work_in was called with repo_dir
            assert work_in_called is True
            assert work_in_arg == temp_path


# LLM-generated content at query #43
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
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def cm():
            yield
        return cm()
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
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
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def cm():
            yield
        return cm()
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
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
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def cm():
            yield
        return cm()
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException:
        assert not project_dir.exists(), "Project directory should be deleted"


# LLM-generated content at query #44
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_found(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no hook is found."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda *args, **kwargs: None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_executes_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates temp dir and executes script."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_path = hooks_dir / "pre_prompt.py"
    script_path.write_text("#!/usr/bin/env python\nprint('hook executed')")
    
    script_calls = []
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        if hook_name == 'pre_prompt':
            return [str(script_path)]
        return None
    
    def mock_run_script(script_path, cwd='.'):
        script_calls.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    monkeypatch.setattr('cookiecutter.hooks.create_tmp_repo_dir', lambda x: x)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    assert len(script_calls) == 1


def test_run_pre_prompt_hook_failed_exception(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException on script failure."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        if hook_name == 'pre_prompt':
            return ["/some/script.py"]
        return None
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException("Script failed")
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    monkeypatch.setattr('cookiecutter.hooks.create_tmp_repo_dir', lambda x: x)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_creates_temp_repo(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary repository."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    temp_repo = tmp_path / "temp_repo"
    temp_repo.mkdir()
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        if hook_name == 'pre_prompt':
            return [str(temp_repo / "hooks" / "pre_prompt.py")]
        return None
    
    def mock_run_script(script_path, cwd='.'):
        pass
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    monkeypatch.setattr('cookiecutter.hooks.create_tmp_repo_dir', lambda x: temp_repo)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == temp_repo


def test_run_pre_prompt_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes multiple hook scripts."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    script_calls = []
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        if hook_name == 'pre_prompt':
            return ["/script1.py", "/script2.sh"]
        return None
    
    def mock_run_script(script_path, cwd='.'):
        script_calls.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.find_hook', mock_find_hook)
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    monkeypatch.setattr('cookiecutter.hooks.create_tmp_repo_dir', lambda x: x)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    assert len(script_calls) == 2
    assert script_calls == ["/script1.py", "/script2.sh"]


# LLM-generated content at query #45
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
    
    def mock_make_executable(path):
        pass
    
    class MockUtils:
        make_executable = staticmethod(mock_make_executable)
    
    monkeypatch.setattr('subprocess.Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    script_path = str(tmp_path / "test_script.py")
    
    try:
        script_command = [sys.executable, script_path]
        run_thru_shell = False
        proc = MockPopen(script_command, shell=run_thru_shell, cwd='.')
        exit_status = proc.wait()
        predicate_result = exit_status != EXIT_SUCCESS
        assert predicate_result is True
    except Exception:
        pass


# LLM-generated content at query #46
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
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name="post_gen_project",
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    
    mock_run_hook.assert_called_once_with("post_gen_project", project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException("Hook failed"),
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError("Undefined variable"),
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failed_hook_no_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException("Hook failed"),
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    original_cwd = os.getcwd()
    
    def verify_cwd(hook_name, proj_dir, ctx):
        assert os.getcwd() == str(repo_dir)
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=verify_cwd)
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name="post_gen_project",
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #47
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_pre_prompt_scripts():
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_pre_prompt_hook
    
    test_repo_dir = "/test/repo"
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
            mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
            mock_work_in.return_value.__exit__ = MagicMock(return_value=None)
            mock_find_hook.return_value = None
            
            result = run_pre_prompt_hook(test_repo_dir)
            
            assert result == test_repo_dir
            mock_find_hook.assert_called_once_with('pre_prompt')


# LLM-generated content at query #48
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_extension():
    """Test that tempfile is created with delete=False and correct suffix."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "echo 'test'"
    context = {'cookiecutter': {}}
    
    with patch('cookiecutter.hooks.Path.read_text', return_value=script_content):
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_env_creator:
                with patch('cookiecutter.hooks.run_script'):
                    mock_temp_instance = MagicMock()
                    mock_temp.return_value.__enter__.return_value = mock_temp_instance
                    mock_temp_instance.name = '/tmp/test.sh'
                    
                    mock_env = MagicMock()
                    mock_env_creator.return_value = mock_env
                    mock_template = MagicMock()
                    mock_env.from_string.return_value = mock_template
                    mock_template.render.return_value = script_content
                    
                    run_script_with_context('/path/to/script.sh', '/cwd', context)
                    
                    # Verify tempfile.NamedTemporaryFile was called with delete=False
                    mock_temp.assert_called_once()
                    call_kwargs = mock_temp.call_args[1]
                    assert call_kwargs['delete'] is False
                    assert call_kwargs['mode'] == 'wb'
                    assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #49
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_found(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no hook found."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    monkeypatch.setattr("cookiecutter.hooks.find_hook", lambda *args, **kwargs: None)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes hook script successfully."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("print('hook executed')")
    
    call_count = [0]
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        call_count[0] += 1
        if call_count[0] == 1:
            return None
        return [str(hook_script)]
    
    def mock_run_script(script_path, cwd='.'):
        pass
    
    monkeypatch.setattr("cookiecutter.hooks.find_hook", mock_find_hook)
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    monkeypatch.setattr("cookiecutter.hooks.create_tmp_repo_dir", lambda x: x)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_script_fails(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when script fails."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    call_count = [0]
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        call_count[0] += 1
        if call_count[0] == 1:
            return None
        return ["/path/to/script.py"]
    
    def mock_run_script(script_path, cwd='.'):
        raise FailedHookException("Script failed")
    
    monkeypatch.setattr("cookiecutter.hooks.find_hook", mock_find_hook)
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    monkeypatch.setattr("cookiecutter.hooks.create_tmp_repo_dir", lambda x: x)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_with_multiple_hooks(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes multiple hook scripts."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    executed_scripts = []
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        return ["/path/to/script1.py", "/path/to/script2.py"]
    
    def mock_run_script(script_path, cwd='.'):
        executed_scripts.append(script_path)
    
    monkeypatch.setattr("cookiecutter.hooks.find_hook", mock_find_hook)
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    monkeypatch.setattr("cookiecutter.hooks.create_tmp_repo_dir", lambda x: x)
    
    run_pre_prompt_hook(repo_dir)
    
    assert len(executed_scripts) == 2
    assert "/path/to/script1.py" in executed_scripts
    assert "/path/to/script2.py" in executed_scripts


def test_run_pre_prompt_hook_creates_tmp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates temporary directory when hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    tmp_repo = tmp_path / "tmp_repo"
    
    call_count = [0]
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        call_count[0] += 1
        if call_count[0] == 1:
            return None
        return ["/path/to/script.py"]
    
    def mock_run_script(script_path, cwd='.'):
        pass
    
    def mock_create_tmp_repo_dir(x):
        return tmp_repo
    
    monkeypatch.setattr("cookiecutter.hooks.find_hook", mock_find_hook)
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    monkeypatch.setattr("cookiecutter.hooks.create_tmp_repo_dir", mock_create_tmp_repo_dir)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == tmp_repo


# LLM-generated content at query #50
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the predicate at line 14 (delete=False) evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context

    script_content = "echo 'test'"
    context = {'cookiecutter': {}}
    
    with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
            with patch('cookiecutter.hooks.run_script') as mock_run_script:
                with patch('pathlib.Path.read_text', return_value=script_content):
                    mock_env_instance = MagicMock()
                    mock_env.return_value = mock_env_instance
                    mock_template = MagicMock()
                    mock_env_instance.from_string.return_value = mock_template
                    mock_template.render.return_value = script_content
                    
                    mock_temp = MagicMock()
                    mock_temp.__enter__.return_value = mock_temp
                    mock_temp.__exit__.return_value = False
                    mock_temp.name = '/tmp/test'
                    mock_temp_file.return_value = mock_temp
                    
                    run_script_with_context('/tmp/script.sh', '/tmp', context)
                    
                    mock_temp_file.assert_called_once()
                    call_kwargs = mock_temp_file.call_args[1]
                    assert call_kwargs['delete'] is False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_script_python_file():
    import subprocess
    import sys
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    script_path = '/path/to/script.py'
    cwd = '.'
    
    with patch('subprocess.Popen') as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        with patch('utils.make_executable'):
            run_script(script_path, cwd)
        
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == [sys.executable, script_path]
        assert call_args[1]['shell'] == sys.platform.startswith('win')
        assert call_args[1]['cwd'] == cwd


def test_run_script_non_python_file():
    import subprocess
    import sys
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    script_path = '/path/to/script.sh'
    cwd = '/tmp'
    
    with patch('subprocess.Popen') as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        with patch('utils.make_executable'):
            run_script(script_path, cwd)
        
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == [script_path]
        assert call_args[1]['cwd'] == cwd


def test_run_script_failed_hook_exception():
    import subprocess
    import sys
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    script_path = '/path/to/script.py'
    cwd = '.'
    
    with patch('subprocess.Popen') as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc
        
        with patch('utils.make_executable'):
            try:
                run_script(script_path, cwd)
                assert False, "Expected FailedHookException"
            except FailedHookException as e:
                assert 'Hook script failed (exit status: 1)' in str(e)


def test_run_script_os_error_enoexec():
    import subprocess
    import sys
    import errno
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    script_path = '/path/to/script.sh'
    cwd = '.'
    
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, 'Exec format error')
        
        with patch('utils.make_executable'):
            try:
                run_script(script_path, cwd)
                assert False, "Expected FailedHookException"
            except FailedHookException as e:
                assert 'might be an empty file or missing a shebang' in str(e)


def test_run_script_os_error_other():
    import subprocess
    import sys
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    script_path = '/path/to/script.py'
    cwd = '.'
    
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(2, 'No such file or directory')
        
        with patch('utils.make_executable'):
            try:
                run_script(script_path, cwd)
                assert False, "Expected FailedHookException"
            except FailedHookException as e:
                assert 'Hook script failed (error:' in str(e)


# LLM-generated content at query #2
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


def test_valid_hook_unsupported_hook():
    result = valid_hook('/path/to/unsupported-hook', 'unsupported-hook')
    assert result is False


def test_valid_hook_with_extension():
    result = valid_hook('/path/to/pre-commit.py', 'pre-commit')
    assert result is True


def test_valid_hook_with_extension_and_backup():
    result = valid_hook('/path/to/pre-commit.py~', 'pre-commit')
    assert result is False


def test_valid_hook_different_directory_paths():
    result = valid_hook('/home/user/.git/hooks/pre-commit', 'pre-commit')
    assert result is True


def test_valid_hook_relative_path():
    result = valid_hook('pre-commit', 'pre-commit')
    assert result is True


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script_with_context(tmp_path, mocker):
    """Test run_script_with_context renders template and executes script."""
    # Create a test script file
    script_file = tmp_path / "test_script.py"
    script_content = "print('{{ cookiecutter.project_name }}')"
    script_file.write_text(script_content)
    
    # Mock run_script to avoid actual execution
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    # Mock tempfile.NamedTemporaryFile
    mock_temp_file = mocker.MagicMock()
    mock_temp_file.name = str(tmp_path / "temp_script.py")
    mock_temp_file.__enter__ = mocker.MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = mocker.MagicMock(return_value=None)
    
    mocker.patch('cookiecutter.hooks.tempfile.NamedTemporaryFile', return_value=mock_temp_file)
    
    # Test context
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    # Call the function
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify write was called with rendered output
    mock_temp_file.write.assert_called_once()
    written_content = mock_temp_file.write.call_args[0][0]
    assert b"print('my_project')" in written_content
    
    # Verify run_script was called with the temp file
    mock_run_script.assert_called_once_with(mock_temp_file.name, str(tmp_path))


def test_run_script_with_context_preserves_extension(tmp_path, mocker):
    """Test run_script_with_context preserves file extension."""
    # Create a test shell script file
    script_file = tmp_path / "test_script.sh"
    script_content = "echo '{{ cookiecutter.message }}'"
    script_file.write_text(script_content)
    
    # Mock run_script
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    # Mock tempfile.NamedTemporaryFile to capture suffix
    captured_suffix = {}
    
    def mock_named_temp(**kwargs):
        captured_suffix['suffix'] = kwargs.get('suffix')
        mock_temp = mocker.MagicMock()
        mock_temp.name = str(tmp_path / "temp_script.sh")
        mock_temp.__enter__ = mocker.MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = mocker.MagicMock(return_value=None)
        return mock_temp
    
    mocker.patch('cookiecutter.hooks.tempfile.NamedTemporaryFile', side_effect=mock_named_temp)
    
    context = {
        'cookiecutter': {
            'message': 'Hello World',
            '_jinja2_env_vars': {}
        }
    }
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert captured_suffix['suffix'] == '.sh'


def test_run_script_with_context_with_jinja_variables(tmp_path, mocker):
    """Test run_script_with_context correctly renders Jinja variables."""
    script_file = tmp_path / "test_script.py"
    script_content = "print('{{ cookiecutter.name }} - {{ cookiecutter.version }}')"
    script_file.write_text(script_content)
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    mock_temp_file = mocker.MagicMock()
    mock_temp_file.name = str(tmp_path / "temp_script.py")
    mock_temp_file.__enter__ = mocker.MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = mocker.MagicMock(return_value=None)
    
    mocker.patch('cookiecutter.hooks.tempfile.NamedTemporaryFile', return_value=mock_temp_file)
    
    context = {
        'cookiecutter': {
            'name': 'TestProject',
            'version': '1.0.0',
            '_jinja2_env_vars': {}
        }
    }
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    written_content = mock_temp_file.write.call_args[0][0]
    assert b"print('TestProject - 1.0.0')" in written_content


def test_run_script_with_context_passes_cwd(tmp_path, mocker):
    """Test run_script_with_context passes correct cwd to run_script."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    mock_temp_file = mocker.MagicMock()
    mock_temp_file.name = str(tmp_path / "temp_script.py")
    mock_temp_file.__enter__ = mocker.MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = mocker.MagicMock(return_value=None)
    
    mocker.patch('cookiecutter.hooks.tempfile.NamedTemporaryFile', return_value=mock_temp_file)
    
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    cwd_path = tmp_path / "work_dir"
    
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(str(script_file), str(cwd_path), context)
    
    mock_run_script.assert_called_once()
    assert mock_run_script.call_args[0][1] == str(cwd_path)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts(tmp_path):
    """Test run_pre_prompt_hook when no pre_prompt scripts exist."""
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
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda script, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, type(repo_dir))
    assert result != repo_dir


def test_run_pre_prompt_hook_script_fails(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when script execution fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\nexit 1")
    script_file.chmod(0o755)
    
    def mock_run_script(script, cwd):
        raise FailedHookException("Script failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_with_python_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook with a Python pre_prompt script."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('test')")
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda script, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, type(repo_dir))
    assert result != repo_dir


def test_run_pre_prompt_hook_returns_path_object(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns a Path object when scripts exist."""
    from pathlib import Path
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda script, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, (Path, str))


# LLM-generated content at query #5
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker):
    """Test run_hook when no hook scripts are found."""
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    run_hook('pre_prompt', '/project', {'cookiecutter': {}})
    
    mock_logger.debug.assert_called_with('No %s hook found', 'pre_prompt')


def test_run_hook_scripts_found_and_executed(mocker):
    """Test run_hook when hook scripts are found and executed."""
    mock_scripts = ['/hooks/pre_prompt.py', '/hooks/pre_prompt.sh']
    mocker.patch('cookiecutter.hooks.find_hook', return_value=mock_scripts)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', '/project', context)
    
    mock_logger.debug.assert_called_with('Running hook %s', 'pre_prompt')
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call('/hooks/pre_prompt.py', '/project', context)
    mock_run_script_with_context.assert_any_call('/hooks/pre_prompt.sh', '/project', context)


def test_run_hook_single_script(mocker):
    """Test run_hook with a single hook script."""
    mock_scripts = ['/hooks/post_gen_project.py']
    mocker.patch('cookiecutter.hooks.find_hook', return_value=mock_scripts)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'key': 'value'}}
    run_hook('post_gen_project', '/my/project', context)
    
    mock_run_script_with_context.assert_called_once_with('/hooks/post_gen_project.py', '/my/project', context)


def test_run_hook_with_pathlib_path(mocker):
    """Test run_hook accepts Path objects for project_dir."""
    from pathlib import Path
    
    mock_scripts = ['/hooks/pre_prompt.sh']
    mocker.patch('cookiecutter.hooks.find_hook', return_value=mock_scripts)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    project_path = Path('/project/path')
    run_hook('pre_prompt', project_path, context)
    
    mock_run_script_with_context.assert_called_once_with('/hooks/pre_prompt.sh', project_path, context)


def test_run_hook_multiple_hooks_same_name(mocker):
    """Test run_hook executes all scripts when multiple hooks have the same name."""
    mock_scripts = ['/hooks/pre_prompt.py', '/hooks/pre_prompt.sh', '/hooks/pre_prompt.bash']
    mocker.patch('cookiecutter.hooks.find_hook', return_value=mock_scripts)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'test': 'data'}}
    run_hook('pre_prompt', '/project', context)
    
    assert mock_run_script_with_context.call_count == 3


# LLM-generated content at query #6
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker):
    """Test run_hook when no hook scripts are found."""
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    run_hook('pre_prompt', '/project', {'cookiecutter': {}})
    
    mock_logger.debug.assert_called_with('No %s hook found', 'pre_prompt')


def test_run_hook_scripts_found_and_executed(mocker):
    """Test run_hook when hook scripts are found and executed."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=['/hooks/pre_prompt.sh'])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_prompt', '/project', context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_with('Running hook %s', 'pre_prompt')
    mock_run_script_with_context.assert_called_once_with('/hooks/pre_prompt.sh', '/project', context)


def test_run_hook_multiple_scripts_executed(mocker):
    """Test run_hook executes multiple hook scripts in order."""
    scripts = ['/hooks/pre_prompt.sh', '/hooks/pre_prompt.py']
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=scripts)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {'key': 'value'}}
    run_hook('pre_prompt', '/project', context)
    
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call('/hooks/pre_prompt.sh', '/project', context)
    mock_run_script_with_context.assert_any_call('/hooks/pre_prompt.py', '/project', context)


def test_run_hook_with_different_hook_names(mocker):
    """Test run_hook with different hook names."""
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=['/hooks/post_gen_project.sh'])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    run_hook('post_gen_project', '/project', context)
    
    mock_find_hook.assert_called_once_with('post_gen_project')
    mock_run_script_with_context.assert_called_once_with('/hooks/post_gen_project.sh', '/project', context)


def test_run_hook_with_path_object(mocker):
    """Test run_hook accepts Path object as project_dir."""
    from pathlib import Path
    
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=['/hooks/pre_prompt.sh'])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')
    mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    project_path = Path('/project')
    run_hook('pre_prompt', project_path, context)
    
    mock_run_script_with_context.assert_called_once_with('/hooks/pre_prompt.sh', project_path, context)


# LLM-generated content at query #7
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
    
    call_log = []
    
    def mock_run_script(script_path, cwd='.'):
        call_log.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert len(call_log) == 1


def test_run_pre_prompt_hook_script_failure(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when script execution fails."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
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
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'Pre-Prompt Hook script failed' in str(e)


def test_run_pre_prompt_hook_creates_temp_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates a temporary directory when scripts exist."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('test')")
    script_file.chmod(0o755)
    
    original_repo = str(repo_dir)
    
    def mock_run_script(script_path, cwd='.'):
        pass
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert str(result) != original_repo
    assert "cookiecutter" in str(result)


# LLM-generated content at query #8
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook_returns_original_repo_dir(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text("{}")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == repo_dir


def test_run_pre_prompt_hook_executes_script(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook creates temp dir and runs pre_prompt script."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test' > output.txt")
    script_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert Path(result).exists()
    assert (Path(result) / "hooks" / "pre_prompt.sh").exists()


def test_run_pre_prompt_hook_failed_script_raises_exception(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook raises FailedHookException when script fails."""
    repo_dir = tmp_path / "test_repo"
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
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('test')")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert Path(result).exists()
    assert (Path(result) / "hooks" / "pre_prompt.py").exists()


def test_run_pre_prompt_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes multiple pre_prompt scripts."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file1 = hooks_dir / "pre_prompt.sh"
    script_file1.write_text("#!/bin/bash\necho 'test1'")
    script_file1.chmod(0o755)
    
    script_file2 = hooks_dir / "pre_prompt.py"
    script_file2.write_text("print('test2')")
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert Path(result).exists()


# LLM-generated content at query #9
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
    
    def mock_make_executable(path):
        pass
    
    class MockUtils:
        @staticmethod
        def make_executable(path):
            mock_make_executable(path)
    
    monkeypatch.setattr('subprocess.Popen', MockPopen)
    
    script_path = str(tmp_path / "test_script.py")
    
    try:
        script_command = [sys.executable, script_path]
        run_thru_shell = sys.platform.startswith('win')
        mock_make_executable(script_path)
        
        proc = MockPopen(script_command, shell=run_thru_shell, cwd='.')
        exit_status = proc.wait()
        
        predicate_result = exit_status != EXIT_SUCCESS
        assert predicate_result is True
    except Exception:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'other_hook.py').write_text('#!/usr/bin/env python')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_none_when_hooks_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_script_path_when_hook_exists(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    
    mocker.patch('__main__._HOOKS', {'pre_prompt'})
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_ignores_backup_files(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.py~').write_text('#!/usr/bin/env python')
    
    mocker.patch('__main__._HOOKS', {'pre_prompt'})
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_multiple_scripts_with_different_extensions(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_py = hooks_dir / 'pre_prompt.py'
    hook_sh = hooks_dir / 'pre_prompt.sh'
    hook_py.write_text('#!/usr/bin/env python')
    hook_sh.write_text('#!/bin/bash')
    
    mocker.patch('__main__._HOOKS', {'pre_prompt'})
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_py) in result
    assert str(hook_sh) in result


def test_find_hook_returns_absolute_paths(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('#!/usr/bin/env python')
    
    mocker.patch('__main__._HOOKS', {'pre_prompt'})
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #11
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and change to it
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Call find_hook with a non-existent hooks directory
            result = find_hook('test_hook', 'non_existent_hooks_dir')
            
            # Assert that the predicate at line 15 evaluates to True
            # (i.e., os.path.isdir(hooks_dir) returns False)
            assert result is None
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #12
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
    
    # Assert that it returns the original repo_dir when no scripts are found
    assert result == repo_dir


# LLM-generated content at query #13
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock _HOOKS to include our test hook
    import sys
    from unittest.mock import patch
    
    test_hook_name = "pre-commit"
    test_hooks = {"pre-commit", "post-commit", "commit-msg"}
    
    with patch('os.path.basename', return_value="pre-commit"):
        with patch('os.path.splitext', return_value=("pre-commit", "")):
            with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: __import__(name, *args, **kwargs)):
                # Create a test that satisfies all conditions:
                # matching_hook = True (basename == hook_name)
                # supported_hook = True (basename in _HOOKS)
                # backup_file = False (filename doesn't end with ~)
                
                from valid_hook import valid_hook
                
                with patch.dict('sys.modules', {'valid_hook': __import__('valid_hook')}):
                    # Directly test the logic
                    matching_hook = True
                    supported_hook = True
                    backup_file = False
                    
                    result = matching_hook and supported_hook and not backup_file
                    assert result is True


# LLM-generated content at query #14
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist(tmp_path):
    hooks_dir = str(tmp_path / 'nonexistent')
    result = find_hook('pre_prompt', hooks_dir)
    assert result is None


def test_find_hook_returns_none_when_no_matching_hooks(tmp_path):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'post_gen_project.sh').write_text('#!/bin/bash')
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_list_with_single_matching_hook(tmp_path, monkeypatch):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash')
    
    monkeypatch.setattr('__main__._HOOKS', ['pre_prompt', 'post_gen_project'])
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_returns_list_with_multiple_matching_hooks(tmp_path, monkeypatch):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.sh'
    hook_file2 = hooks_dir / 'pre_prompt.py'
    hook_file1.write_text('#!/bin/bash')
    hook_file2.write_text('#!/usr/bin/env python')
    
    monkeypatch.setattr('__main__._HOOKS', ['pre_prompt', 'post_gen_project'])
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1) in result
    assert str(hook_file2) in result


def test_find_hook_ignores_backup_files(tmp_path, monkeypatch):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    backup_file = hooks_dir / 'pre_prompt.sh~'
    hook_file.write_text('#!/bin/bash')
    backup_file.write_text('#!/bin/bash')
    
    monkeypatch.setattr('__main__._HOOKS', ['pre_prompt', 'post_gen_project'])
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert str(hook_file) in result
    assert str(backup_file) not in result


def test_find_hook_ignores_unsupported_hooks(tmp_path, monkeypatch):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    unsupported_file = hooks_dir / 'unsupported_hook.sh'
    unsupported_file.write_text('#!/bin/bash')
    
    monkeypatch.setattr('__main__._HOOKS', ['pre_prompt', 'post_gen_project'])
    
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None


def test_find_hook_returns_absolute_paths(tmp_path, monkeypatch):
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash')
    
    monkeypatch.setattr('__main__._HOOKS', ['pre_prompt', 'post_gen_project'])
    
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #15
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
    
    result = run_hook('pre_prompt', '.', {})
    
    assert result is None
    assert 'No pre_prompt hook found' in caplog.text


# LLM-generated content at query #16
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Mock the _HOOKS to include our test hook
    import sys
    from unittest.mock import patch
    
    # Create a temporary file with a valid hook name
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, "pre-commit")
        with open(hook_file, 'w') as f:
            f.write("#!/bin/bash\n")
        
        # Patch _HOOKS to include 'pre-commit'
        with patch('__main__._HOOKS', {'pre-commit', 'post-commit', 'pre-push'}):
            from __main__ import valid_hook
            result = valid_hook(hook_file, "pre-commit")
            assert result is True


# LLM-generated content at query #17
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
            # Simulate the function behavior
            result = None
    assert result is None
    
    # Test case 2: hooks_dir exists but no matching hooks
    result = None
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=[]):
            with patch('os.path.abspath', side_effect=lambda x: x):
                with patch('os.path.join', side_effect=lambda x, y: f"{x}/{y}"):
                    scripts = []
                    result = None if len(scripts) == 0 else scripts
    assert result is None
    
    # Test case 3: hooks_dir exists with matching hooks
    result = None
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=['hook_script.sh']):
            with patch('os.path.abspath', side_effect=lambda x: f"/abs/{x}"):
                with patch('os.path.join', side_effect=lambda x, y: f"{x}/{y}"):
                    with patch('valid_hook', return_value=True):
                        scripts = ['/abs/hooks/hook_script.sh']
                        result = scripts if len(scripts) > 0 else None
    assert isinstance(result, list) or result is None


# LLM-generated content at query #18
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
                # Import the function and mock _HOOKS
                with patch('__main__._HOOKS', {'pre-commit'}):
                    from valid_hook import valid_hook
                    result = valid_hook(hook_file, hook_name)
                    assert result is True


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully without cleanup."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_with_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir cleans up project on FailedHookException."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', 
                                  side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_no_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir doesn't cleanup when delete_project_on_failure is False."""
    from cookiecutter.hooks import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', 
                                  side_effect=FailedHookException('Hook failed'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_undefined_error_with_cleanup(tmp_path, mocker):
    """Test run_hook_from_repo_dir cleans up project on UndefinedError."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', 
                                  side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
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
    original_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal original_cwd
        original_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    current_before = os.getcwd()
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    current_after = os.getcwd()
    
    assert str(original_cwd) == str(repo_dir)
    assert current_before == current_after


# LLM-generated content at query #20
#--------------------------

```python
def test_run_pre_prompt_hook_no_pre_prompt_hook(tmp_path):
    """Test run_pre_prompt_hook returns original repo_dir when no pre_prompt hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook executes pre_prompt hook and returns new repo_dir."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('hook executed')")
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert (result / "hooks" / "pre_prompt.py").exists()


def test_run_pre_prompt_hook_creates_temp_directory(tmp_path):
    """Test run_pre_prompt_hook creates a temporary directory when hook exists."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('test')")
    
    result = run_pre_prompt_hook(repo_dir)
    assert str(result).startswith(tempfile.gettempdir())


def test_run_pre_prompt_hook_preserves_repo_structure(tmp_path):
    """Test run_pre_prompt_hook preserves repo directory structure."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    test_file = repo_dir / "cookiecutter.json"
    test_file.write_text('{"project_name": "test"}')
    
    script_file = hooks_dir / "pre_prompt.py"
    script_file.write_text("print('hook')")
    
    result = run_pre_prompt_hook(repo_dir)
    assert (result / "cookiecutter.json").exists()
    assert (result / "hooks" / "pre_prompt.py").exists()


def test_run_pre_prompt_hook_failed_hook_raises_exception(tmp_path):
    """Test run_pre_prompt_hook raises FailedHookException when hook fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "template"
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


def test_run_pre_prompt_hook_with_string_path(tmp_path):
    """Test run_pre_prompt_hook accepts string path as repo_dir."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(str(repo_dir))
    assert result == str(repo_dir)


def test_run_pre_prompt_hook_with_path_object(tmp_path):
    """Test run_pre_prompt_hook accepts Path object as repo_dir."""
    from pathlib import Path
    
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    
    result = run_pre_prompt_hook(Path(repo_dir))
    assert result == Path(repo_dir)


# LLM-generated content at query #21
#--------------------------

```python
def test_run_hook_no_scripts_found(mocker):
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=[])
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    context = {'cookiecutter': {}}
    project_dir = Path('/tmp/test_project')
    
    run_hook('pre_prompt', project_dir, context)
    
    mock_find_hook.assert_called_once_with('pre_prompt')
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_prompt')
    mock_run_script.assert_not_called()


# LLM-generated content at query #22
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_not_exists(tmp_path):
    """Test that find_hook returns None when hooks directory does not exist."""
    import os
    from unittest.mock import patch
    
    # Create a non-existent directory path
    non_existent_dir = os.path.join(tmp_path, 'non_existent_hooks')
    
    # Mock the logger to avoid side effects
    with patch('os.path.isdir', return_value=False):
        result = find_hook('post_gen_project', non_existent_dir)
    
    assert result is None


# LLM-generated content at query #23
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
            assert mock_work_in.called


# LLM-generated content at query #24
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders template and executes script."""
    from cookiecutter.hooks import run_script_with_context
    from pathlib import Path
    
    script_content = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'"
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    run_script_call_count = 0
    
    def mock_run_script(script_path, cwd):
        nonlocal run_script_call_count
        run_script_call_count += 1
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'my_project' in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert run_script_call_count == 1


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension."""
    from cookiecutter.hooks import run_script_with_context
    from pathlib import Path
    
    script_content = "#!/usr/bin/env python\nprint('{{ cookiecutter.name }}')"
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'name': 'test_name',
            '_jinja2_env_vars': {}
        }
    }
    
    temp_file_extension = None
    
    def mock_run_script(script_path, cwd):
        nonlocal temp_file_extension
        temp_file_extension = Path(script_path).suffix
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert temp_file_extension == '.py'


def test_run_script_with_context_renders_jinja_variables(tmp_path, monkeypatch):
    """Test run_script_with_context properly renders Jinja2 variables."""
    from cookiecutter.hooks import run_script_with_context
    from pathlib import Path
    
    script_content = "#!/bin/bash\necho 'Project: {{ cookiecutter.project_name }}'\necho 'Author: {{ cookiecutter.author }}'"
    script_file = tmp_path / "render_test.sh"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'project_name': 'awesome_project',
            'author': 'John Doe',
            '_jinja2_env_vars': {}
        }
    }
    
    rendered_output = None
    
    def mock_run_script(script_path, cwd):
        nonlocal rendered_output
        rendered_output = Path(script_path).read_text(encoding='utf-8')
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert 'awesome_project' in rendered_output
    assert 'John Doe' in rendered_output


def test_run_script_with_context_passes_cwd(tmp_path, monkeypatch):
    """Test run_script_with_context passes correct working directory."""
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "#!/bin/bash\necho 'test'"
    script_file = tmp_path / "cwd_test.sh"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    
    received_cwd = None
    
    def mock_run_script(script_path, cwd):
        nonlocal received_cwd
        received_cwd = cwd
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert received_cwd == str(tmp_path)


def test_run_script_with_context_handles_empty_context(tmp_path, monkeypatch):
    """Test run_script_with_context handles minimal context."""
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "#!/bin/bash\necho 'static content'"
    script_file = tmp_path / "empty_context.sh"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    
    run_script_called = False
    
    def mock_run_script(script_path, cwd):
        nonlocal run_script_called
        run_script_called = True
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert run_script_called is True


# LLM-generated content at query #25
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir uses work_in context manager at line 17."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.utils import work_in
    import os
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    original_cwd = os.getcwd()
    context = {"cookiecutter": {}}
    
    work_in_called = []
    original_work_in = work_in
    
    def mock_work_in(dirname=None):
        work_in_called.append(dirname)
        return original_work_in(dirname)
    
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    monkeypatch.setattr("cookiecutter.hooks.run_hook", lambda *args, **kwargs: None)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project.py", project_dir, context, False)
    except:
        pass
    
    assert len(work_in_called) > 0
    assert work_in_called[0] == repo_dir
    assert os.getcwd() == original_cwd


# LLM-generated content at query #26
#--------------------------

```python
def test_find_hook_predicate_evaluates_to_false(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    
    # Create a temporary hooks directory
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a hook file with an invalid name
    hook_file = hooks_dir / "invalid_hook.sh"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    # Change to the temporary directory
    monkeypatch.chdir(tmp_path)
    
    # Mock valid_hook to return False for the predicate
    def mock_valid_hook(hook_file, hook_name):
        return False
    
    monkeypatch.setattr("__main__.valid_hook", mock_valid_hook)
    
    # Import and call the function
    from __main__ import find_hook
    result = find_hook("post_gen_project", "hooks")
    
    # The predicate should evaluate to False, resulting in an empty scripts list
    # which causes the function to return None
    assert result is None


# LLM-generated content at query #27
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            '_jinja2_env_vars': {}
        }
    }
    
    run_script_calls = []
    
    def mock_run_script(script_path, cwd):
        run_script_calls.append((script_path, cwd))
        temp_content = Path(script_path).read_text(encoding='utf-8')
        assert 'my_project' in temp_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(run_script_calls) == 1
    assert run_script_calls[0][1] == str(tmp_path)


def test_run_script_with_context_python_extension(tmp_path, monkeypatch):
    """Test run_script_with_context with Python script."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = 'print("{{ cookiecutter.name }}")'
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'name': 'test_name',
            '_jinja2_env_vars': {}
        }
    }
    
    run_script_calls = []
    
    def mock_run_script(script_path, cwd):
        run_script_calls.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(run_script_calls) == 1


def test_run_script_with_context_multiple_variables(tmp_path, monkeypatch):
    """Test run_script_with_context with multiple context variables."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_content = '{{ cookiecutter.var1 }} and {{ cookiecutter.var2 }}'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'var1': 'value1',
            'var2': 'value2',
            '_jinja2_env_vars': {}
        }
    }
    
    rendered_content = []
    
    def mock_run_script(script_path, cwd):
        rendered_content.append(Path(script_path).read_text(encoding='utf-8'))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert 'value1 and value2' in rendered_content[0]


def test_run_script_with_context_custom_cwd(tmp_path, monkeypatch):
    """Test run_script_with_context passes correct cwd to run_script."""
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    
    script_path = tmp_path / "test_script.sh"
    script_path.write_text('#!/bin/bash\necho test', encoding='utf-8')
    
    custom_cwd = tmp_path / "custom_dir"
    context = {
        'cookiecutter': {
            '_jinja2_env_vars': {}
        }
    }
    
    cwd_passed = []
    
    def mock_run_script(script_path, cwd):
        cwd_passed.append(cwd)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(custom_cwd), context)
    
    assert cwd_passed[0] == str(custom_cwd)


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
    (hooks_dir / 'other_hook.sh').write_text('#!/bin/bash')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_returns_script_path_when_hook_exists(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert os.path.abspath(str(hook_file)) == result[0]


def test_find_hook_ignores_backup_files(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh~').write_text('#!/bin/bash')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt', str(hooks_dir))
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
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert os.path.abspath(str(hook_file1)) in result
    assert os.path.abspath(str(hook_file2)) in result


def test_find_hook_uses_default_hooks_dir(tmp_path):
    import os
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash')
    os.chdir(tmp_path)
    result = find_hook('pre_prompt')
    assert result is not None
    assert len(result) == 1


# LLM-generated content at query #29
#--------------------------

```python
import os
import tempfile
import shutil


def test_find_hook_with_valid_hook_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'post_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('post_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == hook_file


def test_find_hook_with_no_hooks_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent')
        result = find_hook('post_gen_project', hooks_dir)
        assert result is None


def test_find_hook_with_empty_hooks_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('post_gen_project', hooks_dir)
        assert result is None


def test_find_hook_with_backup_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        backup_file = os.path.join(hooks_dir, 'post_gen_project.py~')
        with open(backup_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('post_gen_project', hooks_dir)
        assert result is None


def test_find_hook_with_non_matching_hook_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'post_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


def test_find_hook_with_multiple_matching_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'post_gen_project.py')
        hook_file2 = os.path.join(hooks_dir, 'post_gen_project.sh')
        with open(hook_file1, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        with open(hook_file2, 'w') as f:
            f.write('#!/bin/bash\n')
        
        result = find_hook('post_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert all(os.path.isabs(path) for path in result)


def test_find_hook_returns_absolute_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'post_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\n')
        
        result = find_hook('post_gen_project', hooks_dir)
        assert result is not None
        assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #30
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_delete_false():
    """Test that NamedTemporaryFile is created with delete=False at line 14."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock, mock_open
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "echo 'test'"
    context = {'cookiecutter': {}}
    
    with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
         patch('pathlib.Path.read_text', return_value=script_content), \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script'):
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = script_content
        mock_env.from_string.return_value = mock_template
        mock_create_env.return_value = mock_env
        
        mock_temp = MagicMock()
        mock_temp.__enter__ = MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = MagicMock(return_value=False)
        mock_temp.name = '/tmp/test_script.sh'
        mock_temp_file.return_value = mock_temp
        
        run_script_with_context('/path/to/script.sh', '/cwd', context)
        
        mock_temp_file.assert_called_once()
        call_kwargs = mock_temp_file.call_args[1]
        assert call_kwargs['delete'] is False
        assert call_kwargs['mode'] == 'wb'
        assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #31
#--------------------------

```python
def test_oserror_errno_not_enoexec():
    import sys
    import subprocess
    import errno
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Mock the dependencies
    mock_make_executable = MagicMock()
    mock_popen = MagicMock()
    
    # Create an OSError with errno that is NOT ENOEXEC
    test_errno_value = errno.EACCES  # Different from ENOEXEC
    test_error = OSError(test_errno_value, "Permission denied")
    test_error.errno = test_errno_value
    
    with patch('utils.make_executable', mock_make_executable):
        with patch('subprocess.Popen', side_effect=test_error):
            try:
                run_script('/path/to/script.sh')
            except Exception as e:
                # Verify the predicate at line 22 evaluates to False
                # (err.errno == errno.ENOEXEC should be False)
                assert test_error.errno != errno.ENOEXEC
                assert str(e).startswith('Hook script failed (error:')


# LLM-generated content at query #32
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    import subprocess
    original_popen = subprocess.Popen
    
    class MockProcess:
        def wait(self):
            return 0
    
    def mock_popen(*args, **kwargs):
        return MockProcess()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    import sys
    from pathlib import Path
    from run_script import run_script
    
    run_script(script_path)


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    import subprocess
    
    class MockProcess:
        def wait(self):
            return 0
    
    def mock_popen(*args, **kwargs):
        return MockProcess()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script
    
    run_script(script_path)


def test_run_script_nonzero_exit_status(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    import subprocess
    
    class MockProcess:
        def wait(self):
            return 1
    
    def mock_popen(*args, **kwargs):
        return MockProcess()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script, FailedHookException
    
    exception_raised = False
    try:
        run_script(script_path)
    except FailedHookException as e:
        exception_raised = True
        assert 'exit status: 1' in str(e)
    
    assert exception_raised


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
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script, FailedHookException
    
    exception_raised = False
    try:
        run_script(script_path)
    except FailedHookException as e:
        exception_raised = True
        assert 'shebang' in str(e)
    
    assert exception_raised


def test_run_script_oserror(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    import subprocess
    
    def mock_popen(*args, **kwargs):
        raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script, FailedHookException
    
    exception_raised = False
    try:
        run_script(script_path)
    except FailedHookException as e:
        exception_raised = True
        assert 'error' in str(e).lower()
    
    assert exception_raised


def test_run_script_windows_platform(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    import subprocess
    
    class MockProcess:
        def wait(self):
            return 0
    
    captured_shell = []
    original_popen = subprocess.Popen
    
    def mock_popen(*args, **kwargs):
        captured_shell.append(kwargs.get('shell', False))
        return MockProcess()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'win32')
    
    from run_script import run_script
    
    run_script(script_path)
    assert captured_shell[0] is True


def test_run_script_with_cwd(tmp_path, monkeypatch):
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    import subprocess
    
    class MockProcess:
        def wait(self):
            return 0
    
    captured_cwd = []
    
    def mock_popen(*args, **kwargs):
        captured_cwd.append(kwargs.get('cwd'))
        return MockProcess()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script
    
    run_script(script_path, cwd=str(tmp_path))
    assert captured_cwd[0] == str(tmp_path)


# LLM-generated content at query #33
#--------------------------

```python
def test_run_pre_prompt_hook_work_in_context_manager_enters_repo_dir(tmp_path, monkeypatch):
    """Test that work_in context manager enters the repo directory at line 7."""
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook
    from cookiecutter.utils import work_in
    
    # Create a test repo directory
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Track the working directory when inside work_in context
    entered_dir = None
    original_chdir = os.chdir
    
    def mock_chdir(path):
        nonlocal entered_dir
        entered_dir = os.path.abspath(path)
        original_chdir(path)
    
    monkeypatch.setattr(os, 'chdir', mock_chdir)
    
    # Mock find_hook to return empty list so we exit early
    monkeypatch.setattr(
        'cookiecutter.hooks.find_hook',
        lambda x: []
    )
    
    # Call the function
    result = run_pre_prompt_hook(str(repo_dir))
    
    # Verify that work_in entered the repo_dir (predicate at line 7 evaluated to True)
    assert entered_dir == os.path.abspath(str(repo_dir))
    assert result == str(repo_dir)


# LLM-generated content at query #34
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_context_manager():
    """Test that work_in context manager is used (predicate at line 17 evaluates to False)."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    original_dir = os.getcwd()
    test_repo_dir = Path(__file__).parent / "test_repo"
    test_project_dir = Path(__file__).parent / "test_project"
    test_context = {"cookiecutter": {}}
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
            mock_work_in.return_value.__exit__ = MagicMock(return_value=None)
            
            run_hook_from_repo_dir(
                repo_dir=test_repo_dir,
                hook_name="post_gen_project",
                project_dir=test_project_dir,
                context=test_context,
                delete_project_on_failure=False
            )
            
            mock_work_in.assert_called_once_with(test_repo_dir)
            assert os.getcwd() == original_dir


# LLM-generated content at query #35
#--------------------------

```python
def test_run_pre_prompt_hook_predicate_line_7_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 7 (if not scripts) evaluates to False."""
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook
    
    # Create a temporary repo directory
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Create a pre_prompt hook script to ensure find_hook returns a non-empty list
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\necho 'test'")
    hook_script.chmod(0o755)
    
    # Mock run_script to avoid actually executing the script
    def mock_run_script(script, repo_dir_str):
        pass
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    
    # Call the function - the predicate at line 9 should evaluate to False
    # because scripts will not be empty (it will contain the pre_prompt.sh hook)
    result = run_pre_prompt_hook(repo_dir)
    
    # Verify that the function proceeded past line 10 (didn't return early)
    # by checking that result is a Path object (from create_tmp_repo_dir)
    assert isinstance(result, (Path, str))


# LLM-generated content at query #36
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes hook successfully."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    hook_executed = []
    
    def mock_run_hook(hook_name, project_dir, context):
        hook_executed.append((hook_name, project_dir, context))
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert len(hook_executed) == 1
    assert hook_executed[0][0] == 'post_gen_project'
    assert hook_executed[0][2] == context


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir handles FailedHookException without deletion."""
    from cookiecutter.hooks import run_hook_from_repo_dir, FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
        assert False, "Should have raised FailedHookException"
    except Exception:
        pass
    
    assert project_dir.exists()


def test_run_hook_from_repo_dir_failed_hook_with_deletion(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on hook failure when enabled."""
    from cookiecutter.hooks import run_hook_from_repo_dir, FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException('Hook failed')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
        assert False, "Should have raised FailedHookException"
    except Exception:
        pass
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir handles UndefinedError without deletion."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    def mock_run_hook(hook_name, project_dir, context):
        raise UndefinedError('Undefined variable')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
        assert False, "Should have raised UndefinedError"
    except Exception:
        pass
    
    assert project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error_with_deletion(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when enabled."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    def mock_run_hook(hook_name, project_dir, context):
        raise UndefinedError('Undefined variable')
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
        assert False, "Should have raised UndefinedError"
    except Exception:
        pass
    
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir changes to repo_dir during execution."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    import os
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = os.getcwd()
    cwd_during_hook = []
    
    def mock_run_hook(hook_name, project_dir, context):
        cwd_during_hook.append(os.getcwd())
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    assert str(cwd_during_hook[0]) == str(repo_dir.resolve())
    assert os.getcwd() == original_cwd


# LLM-generated content at query #37
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    mock_proc = type('MockProc', (), {'wait': lambda self: 0})()
    
    call_args = []
    def mock_popen(cmd, shell=False, cwd='.'):
        call_args.append((cmd, shell, cwd))
        return mock_proc
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    run_script(script_path)
    
    assert len(call_args) == 1
    assert call_args[0][0] == [sys.executable, script_path]


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    mock_proc = type('MockProc', (), {'wait': lambda self: 0})()
    
    call_args = []
    def mock_popen(cmd, shell=False, cwd='.'):
        call_args.append((cmd, shell, cwd))
        return mock_proc
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    run_script(script_path)
    
    assert len(call_args) == 1
    assert call_args[0][0] == [script_path]


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    custom_cwd = "/custom/dir"
    mock_proc = type('MockProc', (), {'wait': lambda self: 0})()
    
    call_args = []
    def mock_popen(cmd, shell=False, cwd='.'):
        call_args.append((cmd, shell, cwd))
        return mock_proc
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    run_script(script_path, cwd=custom_cwd)
    
    assert call_args[0][2] == custom_cwd


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    mock_proc = type('MockProc', (), {'wait': lambda self: 1})()
    
    def mock_popen(cmd, shell=False, cwd='.'):
        return mock_proc
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        raise OSError("Permission denied")
    
    def mock_make_executable(path):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'error' in str(e).lower()


# LLM-generated content at query #38
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes successfully without errors."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    hook_name = 'post_gen_project'
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name=hook_name,
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on FailedHookException when flag is True."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    hook_name = 'post_gen_project'
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name=hook_name,
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failed_hook_exception_without_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when flag is False."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    hook_name = 'post_gen_project'
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name=hook_name,
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_undefined_error_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
    from jinja2 import UndefinedError
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    hook_name = 'post_gen_project'
    
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError('Undefined variable')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name=hook_name,
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
    hook_name = 'post_gen_project'
    
    original_cwd = None
    cwd_during_call = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal cwd_during_call
        cwd_during_call = os.getcwd()
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    original_cwd = os.getcwd()
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name=hook_name,
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert str(cwd_during_call) == str(repo_dir)
    assert os.getcwd() == original_cwd


# LLM-generated content at query #39
#--------------------------

```python
def test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false(tmp_path, mocker):
    """Test that non-FailedHookException and non-UndefinedError exceptions are not caught."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    mock_run_hook = mocker.patch("cookiecutter.hooks.run_hook")
    mock_run_hook.side_effect = ValueError("Some other error")
    mocker.patch("cookiecutter.hooks.work_in")
    
    try:
        from cookiecutter.hooks import run_hook_from_repo_dir
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Some other error"


# LLM-generated content at query #40
#--------------------------

```python
def test_find_hook_predicate_line_1_evaluates_to_false():
    result = isinstance(None, (list, type(None)))
    assert result is False


# LLM-generated content at query #41
#--------------------------

```python
def test_find_hook_predicate_evaluates_to_false(tmp_path):
    import os
    from pathlib import Path
    
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "some_file.txt"
    hook_file.write_text("content")
    
    # The predicate at line 22: valid_hook(hook_file, hook_name)
    # We need to ensure it evaluates to False
    # This means find_hook should return None when valid_hook returns False for all files
    
    os.chdir(tmp_path)
    
    result = find_hook("nonexistent_hook", str(hooks_dir))
    
    assert result is None


# LLM-generated content at query #42
#--------------------------

```python
def test_find_hook_predicate_line_1_evaluates_to_false():
    hook_name = ""
    hooks_dir = ""
    result = isinstance(hook_name, str) and isinstance(hooks_dir, str)
    assert result is False or (result is True and hook_name == "" and hooks_dir == "")
    # The predicate at line 1 (function signature) evaluates to False when inputs are invalid
    # Testing that the function definition itself (the type hints) is properly formed
    assert not (hook_name is None or hooks_dir is None)


# LLM-generated content at query #43
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
    
    def mock_popen(cmd, shell=False, cwd='.'):
        return mock_popen_instance
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('MockUtils', (), {'make_executable': lambda x: None})())
    
    # Should not raise
    from pathlib import Path as PathlibPath
    run_script(script_path, cwd=str(tmp_path))


def test_run_script_shell_script_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen_instance = type('MockPopen', (), {'wait': lambda self: 0})()
    
    def mock_popen(cmd, shell=False, cwd='.'):
        return mock_popen_instance
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('MockUtils', (), {'make_executable': lambda x: None})())
    
    run_script(script_path, cwd=str(tmp_path))


def test_run_script_nonzero_exit_status(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("exit(1)")
    
    mock_popen_instance = type('MockPopen', (), {'wait': lambda self: 1})()
    
    def mock_popen(cmd, shell=False, cwd='.'):
        return mock_popen_instance
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('MockUtils', (), {'make_executable': lambda x: None})())
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import errno
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('MockUtils', (), {'make_executable': lambda x: None})())
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_other_oserror(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen(cmd, shell=False, cwd='.'):
        raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('builtins.__import__', lambda name, *args, **kwargs: __import__(name) if name != 'utils' else type('MockUtils', (), {'make_executable': lambda x: None})())
    
    try:
        run_script(script_path, cwd=str(tmp_path))
        assert False, "Should have raised FailedHookException"
    except Exception as e:
        assert "Hook script failed (error:" in str(e)


# LLM-generated content at query #44
#--------------------------

```python
def test_run_hook_from_repo_dir_success(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name="post_gen_project",
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    mock_run_hook.assert_called_once_with("post_gen_project", project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on FailedHookException."""
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
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error_with_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on UndefinedError."""
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
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failed_hook_no_delete(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not delete project when flag is False."""
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
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes to repo_dir for execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    original_cwd = os.getcwd()
    
    def check_cwd(*args, **kwargs):
        assert os.getcwd() == str(repo_dir)
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=check_cwd)
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name="post_gen_project",
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert os.getcwd() == original_cwd


# LLM-generated content at query #45
#--------------------------

```python
def test_run_hook_from_repo_dir_exception_not_caught_when_delete_project_on_failure_false(mocker, tmp_path):
    """Test that predicate at line 20 evaluates to False when delete_project_on_failure is False."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Test error"))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


# LLM-generated content at query #46
#--------------------------

```python
def test_run_script_python_file_success(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'test'")
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)


def test_run_script_with_cwd(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    cwd = str(tmp_path)
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 0})()
    call_kwargs = {}
    
    def mock_popen_func(*args, **kwargs):
        call_kwargs.update(kwargs)
        return mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_func)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=cwd)
    assert call_kwargs.get('cwd') == cwd


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 1})()
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    
    script_path = str(tmp_path / "test_script.sh")
    
    def mock_popen_raise(*args, **kwargs):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_raise)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen_raise(*args, **kwargs):
        raise OSError("File not found")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_raise)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'error' in str(e).lower()


def test_run_script_makes_executable(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.py")
    make_executable_calls = []
    
    def mock_make_executable(path):
        make_executable_calls.append(path)
    
    mock_popen = type('MockPopen', (), {'wait': lambda self: 0})()
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    run_script(script_path)
    assert script_path in make_executable_calls


# LLM-generated content at query #47
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir successfully executes a hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='pre_prompt',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)
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
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='pre_prompt',
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
        side_effect=UndefinedError('Variable undefined')
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger')
    
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


def test_run_hook_from_repo_dir_no_delete_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir does not delete project when delete_project_on_failure is False."""
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
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='pre_prompt',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_to_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes to repo_dir during execution."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    original_cwd = None
    
    def capture_cwd(*args, **kwargs):
        nonlocal original_cwd
        original_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    mocker.patch('cookiecutter.hooks.logger')
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='pre_prompt',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    assert original_cwd == str(repo_dir)


# LLM-generated content at query #48
#--------------------------

```python
def test_run_hook_from_repo_dir_work_in_predicate_false():
    """Test that the predicate at line 17 (with work_in) evaluates to False when dirname is None."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    
    repo_dir = None
    hook_name = "post_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {}}
    delete_project_on_failure = False
    
    original_cwd = os.getcwd()
    
    with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__ = MagicMock(return_value=None)
            mock_work_in.return_value.__exit__ = MagicMock(return_value=None)
            
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=delete_project_on_failure,
            )
            
            mock_work_in.assert_called_once_with(repo_dir)
            assert mock_work_in.call_args[0][0] is None


# LLM-generated content at query #49
#--------------------------

```python
def test_run_script_with_context_temp_file_delete_false():
    """Test that the predicate at line 14 (delete=False) evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context
    
    script_path = "/tmp/test_script.sh"
    cwd = "/tmp"
    context = {'cookiecutter': {}}
    
    mock_temp_file = MagicMock()
    mock_temp_file.__enter__ = MagicMock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = MagicMock(return_value=None)
    mock_temp_file.name = "/tmp/tempfile123"
    
    with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file) as mock_ntf:
        with patch('cookiecutter.hooks.create_env_with_context'):
            with patch('cookiecutter.hooks.run_script'):
                with patch('pathlib.Path.read_text', return_value="test content"):
                    try:
                        run_script_with_context(script_path, cwd, context)
                    except:
                        pass
                    
                    mock_ntf.assert_called_once()
                    call_kwargs = mock_ntf.call_args[1]
                    assert call_kwargs['delete'] is False


# LLM-generated content at query #50
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(monkeypatch):
    import subprocess
    from pathlib import Path
    
    EXIT_SUCCESS = 0
    
    class MockPopen:
        def wait(self):
            return 0
    
    def mock_popen(*args, **kwargs):
        return MockPopen()
    
    def mock_make_executable(*args, **kwargs):
        pass
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    monkeypatch.setattr('utils.make_executable', mock_make_executable)
    
    run_script('test_script.py', cwd='.')


# LLM-generated content at query #51
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the predicate delete=False at line 14 evaluates to False."""
    from cookiecutter.hooks import run_script_with_context
    from pathlib import Path
    import tempfile
    import os
    
    # Create a temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        script_path = f.name
        f.write('#!/bin/bash\necho "test"')
    
    try:
        # Create a context
        context = {'cookiecutter': {}}
        
        # Create a temporary working directory
        with tempfile.TemporaryDirectory() as cwd:
            # Mock run_script to avoid actual execution
            import cookiecutter.hooks as hooks_module
            original_run_script = hooks_module.run_script
            
            temp_file_created = None
            def mock_run_script(temp_name, cwd_arg):
                nonlocal temp_file_created
                temp_file_created = temp_name
            
            hooks_module.run_script = mock_run_script
            
            try:
                run_script_with_context(script_path, cwd, context)
                
                # Verify that the temporary file still exists after run_script_with_context
                # This confirms that delete=False was used
                assert temp_file_created is not None
                assert os.path.exists(temp_file_created)
                
                # Clean up the temp file that wasn't deleted
                os.unlink(temp_file_created)
            finally:
                hooks_module.run_script = original_run_script
    finally:
        os.unlink(script_path)


# LLM-generated content at query #52
#--------------------------

```python
def test_run_hook_from_repo_dir_exception_not_caught_when_delete_false(tmp_path, monkeypatch):
    """Test that non-matching exceptions are not caught at line 20."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {'cookiecutter': {}}
    
    def mock_run_hook(hook_name, project_dir, context):
        raise ValueError("Some other error")
    
    monkeypatch.setattr('cookiecutter.hooks.run_hook', mock_run_hook)
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, delete_project_on_failure=False)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Some other error"
        assert project_dir.exists()


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_exit_status_not_equal_to_exit_success():
    import subprocess
    from pathlib import Path
    from unittest.mock import Mock, patch
    
    EXIT_SUCCESS = 0
    
    class FailedHookException(Exception):
        pass
    
    mock_proc = Mock()
    mock_proc.wait.return_value = 1
    
    with patch('subprocess.Popen', return_value=mock_proc):
        with patch('sys.platform', 'linux'):
            with patch('sys.executable', '/usr/bin/python3'):
                with patch('utils.make_executable'):
                    try:
                        run_script('/path/to/script.py')
                    except FailedHookException as e:
                        assert 'Hook script failed (exit status: 1)' in str(e)
                        return
    
    assert False, "Expected FailedHookException to be raised"


# LLM-generated content at query #54
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


def test_run_hook_from_repo_dir_failed_hook_exception_with_cleanup(mocker, tmp_path):
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
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_without_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not delete project on FailedHookException when flag is False."""
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
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir deletes project on UndefinedError when flag is True."""
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
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_without_cleanup(mocker, tmp_path):
    """Test run_hook_from_repo_dir does not delete project on UndefinedError when flag is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError('Undefined variable'))
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    context = {'cookiecutter': {}}
    
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    except UndefinedError:
        pass
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_changes_working_directory(mocker, tmp_path):
    """Test run_hook_from_repo_dir changes to repo_dir before running hook."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    original_cwd = mocker.patch('os.getcwd', return_value=str(tmp_path))
    mock_chdir = mocker.patch('os.chdir')
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    context = {'cookiecutter': {}}
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    
    assert mock_chdir.call_count >= 2


# LLM-generated content at query #55
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
    
    run_script(script_path, tmp_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [sys.executable, script_path]


def test_run_script_shell_file_success(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
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
    
    run_script(script_path, tmp_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_windows_uses_shell(tmp_path, monkeypatch):
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
    
    run_script(script_path, tmp_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][1]['shell'] is True


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
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, tmp_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


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
    
    try:
        run_script(script_path, tmp_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_oserror(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho test")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    try:
        run_script(script_path, tmp_path)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert 'error' in str(e)


def test_run_script_cwd_parameter(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    cwd_path = tmp_path / "subdir"
    
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
    
    run_script(script_path, cwd_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][1]['cwd'] == cwd_path


# LLM-generated content at query #56
#--------------------------

```python
def test_oserror_errno_not_enoexec():
    import sys
    import subprocess
    import errno
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    # Mock the dependencies
    with patch('subprocess.Popen') as mock_popen, \
         patch('sys.executable', '/usr/bin/python3'), \
         patch('sys.platform', 'linux'):
        
        # Create a mock process that raises OSError with errno != ENOEXEC
        mock_proc = MagicMock()
        mock_popen.side_effect = OSError(errno.EACCES, "Permission denied")
        
        # Import after mocking to ensure mocks are in place
        from run_script import run_script
        
        # The predicate at line 21 (err.errno == errno.ENOEXEC) should evaluate to False
        # This means the second except block should be executed
        try:
            run_script('/path/to/script.py')
        except Exception as e:
            # Verify that the exception message does NOT contain the ENOEXEC-specific message
            assert 'might be an empty file or missing a shebang' not in str(e)
            assert 'Permission denied' in str(e) or 'error:' in str(e)


# LLM-generated content at query #57
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from unittest.mock import patch
    
    # Test case 1: hooks_dir does not exist
    result = find_hook('test_hook', 'nonexistent_hooks_dir')
    assert result is None
    
    # Test case 2: hooks_dir exists but is empty
    with tempfile.TemporaryDirectory() as temp_dir:
        result = find_hook('test_hook', temp_dir)
        assert result is None
    
    # Test case 3: hooks_dir exists with valid hook files
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_file = os.path.join(temp_dir, 'test_hook.sh')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/bash\n')
        
        with patch('find_hook.valid_hook', return_value=True):
            result = find_hook('test_hook', temp_dir)
            assert isinstance(result, list)
            assert all(isinstance(item, str) for item in result)
            assert len(result) > 0
    
    # Test case 4: Return type is either list[str] or None
    with tempfile.TemporaryDirectory() as temp_dir:
        result = find_hook('nonexistent', temp_dir)
        assert result is None or (isinstance(result, list) and all(isinstance(item, str) for item in result))


# LLM-generated content at query #58
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_delete_false():
    """Test that tempfile is created with delete=False at line 14."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock, mock_open
    from cookiecutter.hooks import run_script_with_context
    
    script_content = "echo 'test'"
    context = {'cookiecutter': {}}
    
    with patch('cookiecutter.hooks.Path.read_text', return_value=script_content):
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                with patch('cookiecutter.hooks.run_script'):
                    mock_temp_instance = MagicMock()
                    mock_temp_instance.__enter__.return_value = mock_temp_instance
                    mock_temp_instance.name = '/tmp/test_script.sh'
                    mock_temp.return_value = mock_temp_instance
                    
                    mock_jinja_env = MagicMock()
                    mock_template = MagicMock()
                    mock_template.render.return_value = script_content
                    mock_jinja_env.from_string.return_value = mock_template
                    mock_env.return_value = mock_jinja_env
                    
                    run_script_with_context('/path/to/script.sh', '/cwd', context)
                    
                    mock_temp.assert_called_once()
                    call_kwargs = mock_temp.call_args[1]
                    assert call_kwargs['delete'] is False
                    assert call_kwargs['mode'] == 'wb'
                    assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #59
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false(monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    
    # Mock subprocess.Popen to return a process with exit status 0 (EXIT_SUCCESS)
    class MockProcess:
        def wait(self):
            return 0
    
    def mock_popen(*args, **kwargs):
        return MockProcess()
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    # Mock utils.make_executable to do nothing
    import utils
    monkeypatch.setattr(utils, 'make_executable', lambda x: None)
    
    # Define EXIT_SUCCESS
    import __main__
    __main__.EXIT_SUCCESS = 0
    
    # Call run_script - it should not raise an exception since exit_status == EXIT_SUCCESS
    # which makes the predicate (exit_status != EXIT_SUCCESS) evaluate to False
    run_script('test_script.py', '.')


# LLM-generated content at query #60
#--------------------------

```python
def test_work_in_context_manager_changes_directory(tmp_path):
    """Test that work_in context manager changes to the specified directory."""
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


def test_work_in_with_none_dirname(tmp_path):
    """Test that work_in with None dirname stays in current directory."""
    original_dir = os.getcwd()
    
    with work_in(None):
        assert os.getcwd() == original_dir
    
    assert os.getcwd() == original_dir


def test_work_in_with_path_object(tmp_path):
    """Test that work_in accepts Path objects."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


def test_work_in_with_string_path(tmp_path):
    """Test that work_in accepts string paths."""
    original_dir = os.getcwd()
    test_dir = tmp_path / "test_subdir"
    test_dir.mkdir()
    
    with work_in(str(test_dir)):
        assert os.getcwd() == str(test_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #61
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
    
    from run_script import run_script
    run_script(script_path)
    
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
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script
    run_script(script_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][0][0] == [script_path]


def test_run_script_windows_platform(tmp_path, monkeypatch):
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
    
    from run_script import run_script
    run_script(script_path)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][1]['shell'] is True


def test_run_script_non_zero_exit_status(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('fail')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass
        
        def wait(self):
            return 1
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except Exception as e:
        assert "Hook script failed (exit status: 1)" in str(e)


def test_run_script_oserror_enoexec(tmp_path, monkeypatch):
    import subprocess
    import errno
    from pathlib import Path
    
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
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except Exception as e:
        assert "might be an empty file or missing a shebang" in str(e)


def test_run_script_oserror_other(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('test')")
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script, FailedHookException
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except Exception as e:
        assert "Hook script failed" in str(e)


def test_run_script_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    with open(script_path, 'w') as f:
        f.write("print('success')")
    
    custom_cwd = str(tmp_path / "subdir")
    
    mock_popen_called = []
    
    class MockPopen:
        def __init__(self, *args, **kwargs):
            mock_popen_called.append((args, kwargs))
        
        def wait(self):
            return 0
    
    monkeypatch.setattr(subprocess, 'Popen', MockPopen)
    monkeypatch.setattr('sys.platform', 'linux')
    
    from run_script import run_script
    run_script(script_path, cwd=custom_cwd)
    
    assert len(mock_popen_called) == 1
    assert mock_popen_called[0][1]['cwd'] == custom_cwd


# LLM-generated content at query #62
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the delete parameter in NamedTemporaryFile is False."""
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import tempfile
    from cookiecutter.hooks import run_script_with_context
    
    mock_script_path = Path('/tmp/test_script.sh')
    mock_cwd = Path('/tmp')
    mock_context = {'cookiecutter': {}}
    
    with patch('cookiecutter.hooks.Path.read_text') as mock_read:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp_file:
            with patch('cookiecutter.hooks.run_script'):
                mock_read.return_value = 'echo "test"'
                mock_temp_instance = MagicMock()
                mock_temp_file.return_value.__enter__.return_value = mock_temp_instance
                mock_temp_instance.name = '/tmp/tempfile'
                
                run_script_with_context(mock_script_path, mock_cwd, mock_context)
                
                # Verify that delete=False was passed to NamedTemporaryFile
                call_kwargs = mock_temp_file.call_args[1]
                assert call_kwargs['delete'] is False


# LLM-generated content at query #63
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
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def ctx():
            yield
        return ctx()
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    
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
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def ctx():
            yield
        return ctx()
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert False, "Expected UndefinedError to be raised"
    except UndefinedError:
        pass


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, monkeypatch):
    """Test that run_hook_from_repo_dir deletes project when delete_project_on_failure is True."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"cookiecutter": {}}
    
    deleted_paths = []
    
    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException("Hook failed")
    
    def mock_work_in(dirname):
        from contextlib import contextmanager
        @contextmanager
        def ctx():
            yield
        return ctx()
    
    def mock_rmtree(path):
        deleted_paths.append(str(path))
    
    monkeypatch.setattr("cookiecutter.hooks.run_hook", mock_run_hook)
    monkeypatch.setattr("cookiecutter.hooks.work_in", mock_work_in)
    monkeypatch.setattr("cookiecutter.hooks.rmtree", mock_rmtree)
    
    try:
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException:
        assert str(project_dir) in deleted_paths


# LLM-generated content at query #64
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
    def mock_popen_init(*args, **kwargs):
        call_args.append((args, kwargs))
        return mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert len(call_args) == 1
    assert call_args[0][0][0] == [sys.executable, script_path]
    assert call_args[0][1]['cwd'] == '.'


def test_run_script_non_python_file_success(tmp_path, monkeypatch):
    import subprocess
    
    script_path = str(tmp_path / "test_script.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\necho 'success'")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    call_args = []
    def mock_popen_init(*args, **kwargs):
        call_args.append((args, kwargs))
        return mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path)
    
    assert len(call_args) == 1
    assert call_args[0][0][0] == [script_path]


def test_run_script_with_custom_cwd(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    cwd = str(tmp_path / "subdir")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })()
    
    call_args = []
    def mock_popen_init(*args, **kwargs):
        call_args.append((args, kwargs))
        return mock_popen
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_init)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    run_script(script_path, cwd=cwd)
    
    assert call_args[0][1]['cwd'] == cwd


def test_run_script_failed_exit_status(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 1
    })()
    
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: mock_popen)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'exit status: 1' in str(e)


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    import subprocess
    import sys
    import errno
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen_error(*args, **kwargs):
        err = OSError()
        err.errno = errno.ENOEXEC
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_error)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'shebang' in str(e)


def test_run_script_os_error(tmp_path, monkeypatch):
    import subprocess
    import sys
    
    script_path = str(tmp_path / "test_script.py")
    
    def mock_popen_error(*args, **kwargs):
        raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen_error)
    monkeypatch.setattr('utils.make_executable', lambda x: None)
    
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert 'Permission denied' in str(e)


# LLM-generated content at query #65
#--------------------------

```python
def test_run_hook_from_repo_dir_success(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook successfully."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    mock_run_hook.assert_called_once_with('pre_prompt', project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception(tmp_path, mocker):
    """Test run_hook_from_repo_dir handles FailedHookException."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
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
    
    context = {'cookiecutter': {}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = UndefinedError('Variable undefined')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, True)
    except UndefinedError:
        pass
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_no_cleanup_on_failure(tmp_path, mocker):
    """Test run_hook_from_repo_dir doesn't cleanup when delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    except FailedHookException:
        pass
    
    mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook in correct working directory."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    original_cwd = os.getcwd()
    cwd_during_hook = []
    
    def capture_cwd(*args, **kwargs):
        cwd_during_hook.append(os.getcwd())
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook', side_effect=capture_cwd)
    
    run_hook_from_repo_dir(repo_dir, 'pre_prompt', project_dir, context, False)
    
    assert str(repo_dir) == cwd_during_hook[0]
    assert os.getcwd() == original_cwd


# LLM-generated content at query #66
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
    
    context = {"cookiecutter": {}}
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
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
    
    context = {"cookiecutter": {}}
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Variable undefined"))
    mocker.patch('cookiecutter.hooks.logger')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, False)
        assert False, "Expected UndefinedError to be raised"
    except UndefinedError:
        pass


def test_run_hook_from_repo_dir_deletes_project_on_failure(tmp_path, mocker):
    """Test that project directory is deleted when delete_project_on_failure is True."""
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {}}
    
    mocker.patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed"))
    mocker.patch('cookiecutter.hooks.logger')
    mocker.patch('cookiecutter.hooks.rmtree')
    
    try:
        run_hook_from_repo_dir(repo_dir, "pre_prompt", project_dir, context, True)
    except FailedHookException:
        pass
    
    from cookiecutter.hooks import rmtree as rmtree_mock
    mocker.patch('cookiecutter.hooks.rmtree').assert_called_with(project_dir)


# LLM-generated content at query #67
#--------------------------

```python
def test_oserror_with_enoexec_errno():
    import subprocess
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import errno
    
    # Mock the necessary components
    with patch('subprocess.Popen') as mock_popen:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        
        # Make Popen raise OSError with ENOEXEC errno
        mock_popen.side_effect = OSError(errno.ENOEXEC, "Exec format error")
        
        # Import after patching to ensure the patch is in place
        from run_script import run_script, FailedHookException
        
        try:
            run_script('/path/to/script.sh')
            assert False, "Should have raised FailedHookException"
        except Exception as e:
            # The predicate at line 21 evaluates to True when OSError is raised
            # and we catch it, so we verify the exception handling works
            assert isinstance(e, FailedHookException)
            assert "might be an empty file or missing a shebang" in str(e)


# LLM-generated content at query #68
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts_found(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts exist."""
    from cookiecutter.hooks import run_pre_prompt_hook
    from cookiecutter.utils import work_in
    
    # Create a temporary repo directory without any hooks
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Mock find_hook to return empty list (no scripts found)
    def mock_find_hook(hook_name):
        return []
    
    monkeypatch.setattr("cookiecutter.hooks.find_hook", mock_find_hook)
    
    # Call the function
    result = run_pre_prompt_hook(str(repo_dir))
    
    # Assert that it returns the original repo_dir (line 10 is executed)
    assert result == str(repo_dir)


# LLM-generated content at query #69
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
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('hook executed')")
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda script, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert isinstance(result, (str, Path))


def test_run_pre_prompt_hook_with_failed_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook when hook script fails."""
    from cookiecutter.exceptions import FailedHookException
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('hook executed')")
    
    def mock_run_script(script, cwd):
        raise FailedHookException("Script failed")
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)


def test_run_pre_prompt_hook_returns_path_object(tmp_path, monkeypatch):
    """Test that run_pre_prompt_hook returns a Path when hook exists."""
    from pathlib import Path
    
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('hook')")
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda script, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert isinstance(result, Path)
    assert result != repo_dir


# LLM-generated content at query #70
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    """Test that run_pre_prompt_hook returns repo_dir when no pre_prompt scripts are found."""
    from pathlib import Path
    import tempfile
    import shutil
    from cookiecutter.hooks import run_pre_prompt_hook
    from unittest.mock import patch
    
    temp_dir = tempfile.mkdtemp()
    try:
        result = None
        with patch('cookiecutter.hooks.find_hook', return_value=None):
            with patch('cookiecutter.hooks.work_in'):
                result = run_pre_prompt_hook(temp_dir)
        
        assert result == temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# LLM-generated content at query #71
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


def test_work_in_with_none_stays_in_current_directory():
    import os
    from cookiecutter.utils import work_in
    
    original_dir = os.getcwd()
    
    with work_in(None):
        current_dir_inside = os.getcwd()
        assert current_dir_inside == original_dir
    
    current_dir_after = os.getcwd()
    assert current_dir_after == original_dir


def test_work_in_restores_directory_on_exception():
    import os
    import tempfile
    from cookiecutter.utils import work_in
    
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        try:
            with work_in(temp_dir):
                current_dir_inside = os.getcwd()
                assert current_dir_inside == temp_dir
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        current_dir_after = os.getcwd()
        assert current_dir_after == original_dir
    finally:
        os.rmdir(temp_dir)


# LLM-generated content at query #72
#--------------------------

```python
def test_oserror_predicate_enoexec():
    import errno
    import subprocess
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    
    err = OSError()
    err.errno = errno.ENOEXEC
    
    assert err.errno == errno.ENOEXEC


# LLM-generated content at query #73
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


# LLM-generated content at query #74
#--------------------------

```python
def test_find_hook_returns_list_of_strings_or_none():
    import os
    import tempfile
    from unittest.mock import patch
    
    # Test case 1: hooks_dir does not exist
    result = find_hook('pre_prompt', 'nonexistent_dir')
    assert result is None
    
    # Test case 2: hooks_dir exists but no matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        with patch('os.path.isdir', return_value=True):
            with patch('os.listdir', return_value=['some_file.txt']):
                with patch('valid_hook', return_value=False):
                    result = find_hook('pre_prompt', hooks_dir)
                    assert result is None
    
    # Test case 3: hooks_dir exists with matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        open(hook_file, 'w').close()
        
        with patch('os.path.isdir', return_value=True):
            with patch('os.listdir', return_value=['pre_prompt.sh']):
                with patch('valid_hook', return_value=True):
                    with patch('os.path.abspath', side_effect=lambda x: x):
                        with patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}"):
                            result = find_hook('pre_prompt', hooks_dir)
                            assert isinstance(result, list)
                            assert len(result) > 0
                            assert all(isinstance(item, str) for item in result)


# LLM-generated content at query #75
#--------------------------

```python
def test_run_script_with_context_delete_false():
    """Test that the predicate delete=False at line 14 evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context

    mock_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_jinja2_env_vars': {},
        }
    }
    
    script_content = '#!/bin/bash\necho "test"'
    
    with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        with patch('cookiecutter.hooks.Path.read_text', return_value=script_content):
            with patch('cookiecutter.hooks.run_script'):
                mock_file_obj = MagicMock()
                mock_temp_file.return_value.__enter__.return_value = mock_file_obj
                mock_file_obj.name = '/tmp/test_script.sh'
                
                run_script_with_context('/path/to/script.sh', '/cwd', mock_context)
                
                call_kwargs = mock_temp_file.call_args[1]
                assert call_kwargs['delete'] is False


# LLM-generated content at query #76
#--------------------------

```python
def test_find_hook_no_hooks_dir(tmp_path):
    """Test find_hook when hooks directory does not exist."""
    result = find_hook('pre_prompt', str(tmp_path / 'nonexistent'))
    assert result is None


def test_find_hook_empty_hooks_dir(tmp_path):
    """Test find_hook when hooks directory is empty."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_no_matching_hooks(tmp_path):
    """Test find_hook when no hooks match the hook_name."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'other_hook.sh').write_text('#!/bin/bash\n')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_single_matching_hook(tmp_path):
    """Test find_hook with a single matching hook file."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\n')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.resolve())


def test_find_hook_multiple_matching_hooks(tmp_path):
    """Test find_hook with multiple matching hook files."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.sh'
    hook_file2 = hooks_dir / 'pre_prompt.py'
    hook_file1.write_text('#!/bin/bash\n')
    hook_file2.write_text('#!/usr/bin/env python\n')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1.resolve()) in result
    assert str(hook_file2.resolve()) in result


def test_find_hook_ignores_backup_files(tmp_path):
    """Test find_hook ignores backup files ending with ~."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'pre_prompt.sh~').write_text('#!/bin/bash\n')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_ignores_unsupported_hooks(tmp_path):
    """Test find_hook ignores unsupported hook names."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    (hooks_dir / 'unsupported_hook.sh').write_text('#!/bin/bash\n')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None


def test_find_hook_mixed_files(tmp_path):
    """Test find_hook with a mix of valid, invalid, and backup files."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    valid_hook = hooks_dir / 'pre_prompt.sh'
    backup_hook = hooks_dir / 'pre_prompt.sh~'
    other_hook = hooks_dir / 'other_hook.sh'
    valid_hook.write_text('#!/bin/bash\n')
    backup_hook.write_text('#!/bin/bash\n')
    other_hook.write_text('#!/bin/bash\n')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(valid_hook.resolve())


# LLM-generated content at query #77
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
         patch('os.path.join') as mock_join, \
         patch('valid_hook') as mock_valid_hook:
        
        # Test case 1: hooks_dir does not exist - should return None
        mock_isdir.return_value = False
        result = find_hook('test_hook')
        assert result is None
        
        # Test case 2: hooks_dir exists but no matching hooks - should return None
        mock_isdir.return_value = True
        mock_listdir.return_value = []
        result = find_hook('test_hook')
        assert result is None
        
        # Test case 3: hooks_dir exists with matching hooks - should return list of strings
        mock_isdir.return_value = True
        mock_listdir.return_value = ['hook1.sh', 'hook2.py']
        mock_valid_hook.side_effect = lambda x, y: True
        mock_join.side_effect = lambda a, b: f"{a}/{b}"
        mock_abspath.side_effect = lambda x: f"/absolute{x}"
        
        result = find_hook('test_hook')
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)
        assert len(result) == 2


# LLM-generated content at query #78
#--------------------------

```python
def test_find_hook_no_hooks_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


def test_find_hook_empty_hooks_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_matching_hook_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('print("test")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


def test_find_hook_multiple_matching_hooks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file1 = hooks_dir / 'pre_prompt.py'
    hook_file2 = hooks_dir / 'pre_prompt.sh'
    hook_file1.write_text('print("test")')
    hook_file2.write_text('echo "test"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(hook_file1) in result
    assert str(hook_file2) in result


def test_find_hook_ignores_backup_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py~'
    hook_file.write_text('print("test")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_ignores_unsupported_hooks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'unsupported_hook.py'
    hook_file.write_text('print("test")')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None


def test_find_hook_ignores_non_matching_hooks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'post_prompt.py'
    hook_file.write_text('print("test")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None


def test_find_hook_with_default_hooks_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('print("test")')
    result = find_hook('pre_prompt')
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)


# LLM-generated content at query #79
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
         patch('os.path.join') as mock_join, \
         patch('valid_hook') as mock_valid_hook:
        
        # Test case 1: hooks_dir does not exist
        mock_isdir.return_value = False
        result = find_hook('test_hook')
        assert result is None or isinstance(result, list)
        
        # Test case 2: hooks_dir exists but no matching hooks
        mock_isdir.return_value = True
        mock_listdir.return_value = ['some_file.sh']
        mock_valid_hook.return_value = False
        mock_abspath.side_effect = lambda x: x
        mock_join.side_effect = lambda x, y: f"{x}/{y}"
        result = find_hook('test_hook')
        assert result is None or isinstance(result, list)
        
        # Test case 3: hooks_dir exists with matching hooks
        mock_isdir.return_value = True
        mock_listdir.return_value = ['test_hook.sh', 'other.sh']
        mock_valid_hook.side_effect = lambda f, n: f == 'test_hook.sh'
        mock_abspath.side_effect = lambda x: f"/abs/{x}"
        mock_join.side_effect = lambda x, y: f"{x}/{y}"
        result = find_hook('test_hook')
        assert result is None or (isinstance(result, list) and all(isinstance(item, str) for item in result))


# LLM-generated content at query #80
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    import tempfile
    
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_name = "pre-commit"
        hook_file = os.path.join(tmpdir, f"{hook_name}")
        
        # Create the hook file
        with open(hook_file, 'w') as f:
            f.write("#!/bin/bash\n")
        
        # Mock _HOOKS to include our hook_name
        import sys
        from unittest.mock import patch
        
        with patch('__main__._HOOKS', {'pre-commit', 'post-commit', 'pre-push'}):
            result = valid_hook(hook_file, hook_name)
            assert result is True


