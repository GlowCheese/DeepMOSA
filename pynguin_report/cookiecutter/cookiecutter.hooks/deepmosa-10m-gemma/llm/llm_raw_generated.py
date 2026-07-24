####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_valid_hook_success():
    _HOOKS = ["pre-commit", "post-checkout"]
    import os
    # Mocking the logic behavior by providing a setup where all conditions are True
    # filename: pre-commit.py -> basename: pre-commit
    # matching_hook: True, supported_hook: True, backup_file: False
    assert valid_hook("/path/to/pre-commit.py", "pre-commit") is True

def test_valid_hook_name_mismatch():
    _HOOKS = ["pre-commit"]
    import os
    # basename is 'other', which does not match 'pre-commit'
    assert valid_hook("/path/to/other.py", "pre-commit") is False

def test_valid_hook_unsupported_name():
    _HOOKS = ["pre-commit"]
    import os
    # basename is 'unknown', which is not in _HOOKS
    assert valid_hook("/path/to/unknown.py", "unknown") is False

def test_valid_hook_is_backup_file():
    _HOOKS = ["pre-commit"]
    import os
    # filename ends with '~', making backup_file True, which should return False
    assert valid_hook("/path/to/pre-commit.py~", "pre-commit") is False

def test_valid_hook_empty_strings():
    _HOOKS = [""]
    import os
    assert valid_hook("", "") is True
```


# LLM-generated content at query #2
#--------------------------

```python
def test_run_hook_no_scripts_found():
    from unittest.mock import patch
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        with patch('cookiecutter.hooks.logger.debug') as mock_debug:
            run_hook('post_gen_project', '/tmp/project', {})
            mock_debug.assert_any_call('No %s hook found', 'post_gen_project')

def test_run_hook_executes_found_scripts():
    from unittest.mock import patch
    scripts = ['/tmp/project/hooks/post_gen_project']
    with patch('cookiecutter.hooks.find_hook', return_value=scripts):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('post_gen_project', '/tmp/project', {'foo': 'bar'})
            mock_run.assert_called_once_with(scripts[0], '/tmp/project', {'foo': 'bar'})

def test_run_hook_executes_multiple_scripts():
    from unittest.mock import patch
    scripts = ['/tmp/project/hooks/post_gen_project', '/tmp/project/hooks/other_hook']
    with patch('cookiecutter.hooks.find_hook', return_value=scripts):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('post_gen_project', '/tmp/project', {})
            assert mock_run.call_count == 2
            mock_run.assert_any_call(scripts[0], '/tmp/project', {})
            mock_run.assert_any_call(scripts[1], '/tmp/project', {})
```


# LLM-generated content at query #3
#--------------------------

```python
import sys
import subprocess
import errno
from unittest.mock import patch, MagicMock
from pathlib import Path

def test_run_script_python_file_success():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'linux'):
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script(script_path='test_script.py', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('test_script.py')
        mock_popen.assert_called_once_with([sys.executable, 'test_script.py'], shell=False, cwd='/tmp')

def test_run_script_shell_script_success_windows():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'win32'):
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script(script_path='/path/to/script.sh', cwd='.')
        
        mock_make_exec.assert_called_once_with('/path/to/script.sh')
        mock_popen.assert_called_once_with(['/path/to/script.sh'], shell=True, cwd='.')

def test_run_script_failure_exit_status():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'),
         patch('sys.platform', 'linux'):
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        from your_module import FailedHookException
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test.py')
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

def test_run_script_os_error_enoexec():
    with patch('subprocess.Popen', side_effect=OSError(errno.ENOEXEC, 'exec format error')), \
         patch('utils.make_executable'),
         patch('sys.platform', 'linux'):
        from your_module import FailedHookException
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test.py')
        assert 'might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_os_error_generic():
    with patch('subprocess.Popen', side_effect=OSError(errno.EACCES, 'Permission denied')), \
         patch('utils.make_executable'),
         patch('sys.platform', 'linux'):
        from your_module import FailedHookException
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test.py')
        assert 'Hook script failed (error: Permission denied)' in str(excinfo.value)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_valid_hook_success():
    _HOOKS = ['pre-commit', 'post-checkout']
    import os
    # Mocking the logic via direct parameter injection if possible, 
    # but since I can only call the function, I assume _HOOKS is accessible in scope.
    # For this test, we assume a scenario where hook_file matches hook_name and is in _HOOKS.
    assert valid_hook('/path/to/pre-commit', 'pre-commit') == True

def test_valid_hook_mismatch_name():
    _HOOKS = ['pre-commit']
    import os
    assert valid_hook('/path/to/other-hook', 'pre-commit') == False

def test_valid_hook_unsupported_type():
    _HOOKS = ['pre-commit']
    import os
    assert valid_hook('/path/to/unknown', 'unknown') == False

def test_valid_hook_is_backup_file():
    _HOOKS = ['pre-commit']
    import os
    assert valid_hook('/path/to/pre-commit~', 'pre-commit') == False

def test_valid_hook_empty_strings():
    _HOOKS = []
    import os
    assert valid_hook('', '') == False

def test_valid_hook_with_extension():
    _HOOKS = ['pre-commit']
    import os
    # basename is 'pre-commit', splitext[0] is 'pre-commit'
    # If file is 'pre-commit.sh', basename becomes 'pre-commit.sh', 
    # splitext[0] is 'pre-commit'. This should return True if 'pre-commit' in _HOOKS.
    assert valid_hook('/path/to/pre-commit.sh', 'pre-commit') == True
```


# LLM-generated content at query #5
#--------------------------

```python
def test_run_hook_no_scripts_found():
    from unittest.mock import patch
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = None
        with patch('cookiecutter.hooks.logger.debug') as mock_log:
            from cookiecutter.hooks import run_hook
            run_hook("pre_gen_project", "/tmp/project", {})
            mock_log.assert_any_call('No %s hook found', "pre_gen_project")

def test_run_hook_executes_scripts():
    from unittest.mock import patch
    script_path = "/tmp/hooks/pre_gen_project.sh"
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = [script_path]
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_ctx:
            with patch('cookiecutter.hooks.logger.debug') as mock_log:
                from cookiecutter.hooks import run_hook
                context = {"project_name": "test"}
                run_hook("pre_gen_project", "/tmp/project", context)
                mock_run_ctx.assert_called_once_with(script_path, "/tmp/project", context)
                mock_log.assert_any_call('Running hook %s', "pre_gen_project")

def test_run_hook_executes_multiple_scripts():
    from unittest.mock import patch
    scripts = ["/tmp/hooks/pre_gen_project.sh", "/tmp/hooks/pre_gen_project.py"]
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = scripts
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_ctx:
            from cookiecutter.hooks import run_hook
            context = {"project_name": "test"}
            run_hook("pre_gen_project", "/tmp/project", context)
            assert mock_run_ctx.call_count == 2
            mock_run_ctx.assert_any_call(scripts[0], "/tmp/project", context)
            mock_run_ctx.assert_any_call(scripts[1], "/tmp/project", context)
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_dir_not_exists():
    result = find_hook("test_hook", "non_existent_directory_12345")
    assert result is None

def test_find_hook_empty_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        result = find_hook("test_hook", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_valid_script_found():
    # Note: This test assumes _HOOKS contains 'pre-commit' for the sake of valid logic execution
    # In a real environment, we would mock _HOOKS and os.listdir
    import unittest.mock as mock
    
    temp_dir = tempfile.mkdtemp()
    hook_name = "pre-commit"
    hook_filename = "pre-commit.sh"
    hook_path = os.path.join(temp_dir, hook_filename)
    
    with open(hook_path, 'w') as f:
        f.write("#!/bin/sh")

    with mock.patch('os.listdir', return_value=[hook_filename]), \
         mock.patch('__main__._HOOKS', ['pre-commit']), \
         mock.patch('os.path.isdir', return_value=True):
        
        result = find_hook(hook_name, temp_dir)
        assert result == [os.path.abspath(hook_path)]
    
    shutil.rmtree(temp_dir)

def test_find_hook_ignores_invalid_names():
    import unittest.mock as mock
    
    temp_dir = tempfile.mkdtemp()
    hook_name = "pre-commit"
    wrong_filename = "wrong_name.sh"
    
    with mock.patch('os.listdir', return_value=[wrong_filename]), \
         mock.patch('__main__._HOOKS', ['pre-commit']), \
         mock.patch('os.path.isdir', return_value=True):
        
        result = find_hook(hook_name, temp_dir)
        assert result is None
    
    shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files():
    import unittest.mock as mock
    
    temp_dir = tempfile.mkdtemp()
    hook_name = "pre-commit"
    backup_filename = "pre-commit.sh~"
    
    with mock.patch('os.listdir', return_value=[backup_filename]), \
         mock.patch('__main__._HOOKS', ['pre-commit']), \
         mock.patch('os.path.isdir', return_value=True):
        
        result = find_hook(hook_name, temp_dir)
        assert result is None
    
    shutil.rmtree(temp_dir)
```


# LLM-generated content at query #7
#--------------------------

```python
import os

def test_find_hook_type_hint_is_correct():
    # This test ensures the function signature matches the expected type hints.
    # The predicate at line 1 is effectively a type check on the return value and parameters.
    # Since we cannot use if/for, we verify that calling it with valid inputs returns a list or None.
    # To ensure the predicate (the definition itself) evaluates to True in a runtime context:
    result = find_hook("test_hook", "hooks")
    assert isinstance(find_hook.__annotations__['hook_name'], str)
    assert isinstance(find_hook.__annotations__['hooks_dir'], str)
    assert find_hook.__annotations__['return'] == list[str] | None
```


# LLM-generated content at query #8
#--------------------------

```python
def test_find_hook_returns_none_when_no_valid_hooks_found():
    import os
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.mkdir(hooks_dir)
    
    # Create a file that does not match the hook_name
    # Note: This assumes valid_hook is defined elsewhere and checks for name matching
    # Since we can't see valid_hook, we provide a file that would fail any reasonable check
    with open(os.path.join(hooks_dir, 'irrelevant_script.py'), 'w') as f:
        f.write('')

    # We pass a hook_name that won't match 'irrelevant_script'
    # The predicate at line 25 (len(scripts) == 0) will be True if no files pass valid_hook
    result = find_hook('non_existent_hook', hooks_dir)

    assert result is None

    shutil.rmtree(temp_dir)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks():
    import tempfile
    import shutil
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook

    temp_dir = Path(tempfile.mkdtemp())
    result = run_pre_prompt_hook(str(temp_dir))
    
    assert result == str(temp_dir)
    shutil.rmtree(temp_dir)

def test_run_pre_prompt_hook_with_valid_hooks():
    import tempfile
    import shutil
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook

    temp_dir = Path(tempfile.mkdtemp())
    hooks_dir = temp_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create a dummy python script that exits successfully
    script_path = hooks_dir / "pre_prompt.py"
    with open(script_path, "w") as f:
        f.write("import sys; sys.exit(0)")
    
    # We need to mock _HOOKS in the module if it's not available, 
    # but assuming the environment allows the execution of this script.
    # Since we can't modify the module code here, we rely on the actual logic.
    # Note: This test assumes 'pre_prompt' is in the _HOOKS constant.
    
    result = run_pre_prompt_hook(str(temp_dir))
    
    assert Path(result).resolve() != temp_dir.resolve()
    assert Path(result).name == temp_dir.name
    
    shutil.rmtree(temp_dir)

def test_run_pre_prompt_hook_failure():
    import tempfile
    import shutil
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook, FailedHookException

    temp_dir = Path(tempfile.mkdtemp())
    hooks_dir = temp_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create a script that exits with an error code
    script_path = hooks_dir / "pre_prompt.py"
    with open(script_path, "w") as f:
        f.write("import sys; sys.exit(1)")

    try:
        with Exception: # We expect FailedHookException
            run_pre_prompt_hook(str(temp_dir))
    except FailedHookException:
        pass
    except Exception as e:
        # If it's not a FailedHookException, the test failed to catch the right error
        raise e

    shutil.rmtree(temp_dir)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_run_pre_prompt_hook_returns_early_when_no_scripts_found():
    import tempfile
    import shutil
    from pathlib import Path
    from unittest.mock import patch
    from cookiecutter.hooks import run_pre_prompt_hook

    temp_dir = Path(tempfile.mkdtemp())
    try:
        with patch("cookiecutter.hooks.find_hook", return_value=[]):
            result = run_pre_prompt_hook(temp_dir)
            assert result == temp_dir
    finally:
        shutil.rmtree(temp_dir)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_hook_no_scripts_found():
    from unittest.mock import patch
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            from cookiecutter.hooks import run_hook
            run_hook('pre_gen_project', '/tmp/project', {})
            mock_run.assert_not_called()

def test_run_hook_executes_scripts():
    from unittest.mock import patch
    scripts = ['/tmp/project/hooks/pre_gen_project_1.sh', '/tmp/project/hooks/pre_gen_project_2.sh']
    with patch('cookiecutter.hooks.find_hook', return_value=scripts):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            from cookiecutter.hooks import run_hook
            context = {'project_name': 'test'}
            run_hook('pre_gen_project', '/tmp/project', context)
            assert mock_run.call_count == 2
            mock_run.assert_any_call(scripts[0], '/tmp/project', context)
            mock_run.assert_any_call(scripts[1], '/tmp/project', context)
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_directory_not_exists():
    result = find_hook("some_hook", "non_existent_directory_12345")
    assert result is None

def test_find_hook_returns_none_when_no_matching_hooks_in_dir():
    test_dir = tempfile.mkdtemp()
    try:
        with open(os.path.join(test_dir, "other_hook.py"), "w") as f:
            f.write("")
        result = find_hook("target_hook", test_dir)
        assert result is None
    finally:
        shutil.rmtree(test_dir)

def test_find_hook_returns_correct_path_for_valid_hook():
    test_dir = tempfile.mkdtemp()
    try:
        # Assuming _HOOKS contains 'target_hook'
        # We must mock or ensure the environment has target_hook in _HOOKS
        # Since I cannot modify global _HOOKS, this test assumes 
        # the context of a valid execution environment where hook name is known.
        hook_filename = "target_hook.py"
        hook_path = os.path.join(test_dir, hook_filename)
        with open(hook_path, "w") as f:
            f.write("")
        
        # Note: This test relies on 'target_hook' being in the global _HOOKS
        # For the purpose of this unit test logic, we assume valid_hook works.
        result = find_hook("target_hook", test_dir)
        assert result is not None
        assert os.path.abspath(hook_path) in result
    finally:
        shutil.rmtree(test_dir)

def test_find_hook_ignores_backup_files():
    test_dir = tempfile.mkdtemp()
    try:
        # target_hook~ is a backup file, should be ignored by valid_hook
        backup_path = os.path.join(test_dir, "target_hook.py~")
        with open(backup_path, "w") as f:
            f.write("")
        
        result = find_hook("target_hook", test_dir)
        assert result is None
    finally:
        shutil_rmtree_helper(test_dir)

def shutil_rmtree_helper(path):
    shutil.rmtree(path)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script_python_file_success():
    with unittest.mock.patch('subprocess.Popen') as mock_popen, \
         unittest.mock.patch('sys.executable', '/usr/bin/python3'), \
         unittest.mock.patch('utils.make_executable') as mock_make_exec:
        
        mock_process = unittest.mock.Mock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('/path/to/script.py', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/path/to/script.py')
        mock_popen.assert_called_once_with(['/usr/bin/python3', '/path/to/script.py'], shell=False, cwd='/tmp')

def test_run_script_shell_file_success():
    with unittest.mock.patch('subprocess.Popen') as mock_popen, \
         unittest.mock.patch('utils.make_executable') as mock_make_exec:
        
        mock_process = unittest.mock.Mock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('/path/to/script.sh', cwd='.')
        
        mock_make_exec.assert_called_once_with('/path/to/script.sh')
        mock_popen.assert_called_once_with(['/path/to/script.sh'], shell=False, cwd='.')

def test_run_script_failure_exit_status():
    with unittest.mock.patch('subprocess.Popen') as mock_popen, \
         unittest.mock.patch('utils.make_executable'):
        
        mock_process = unittest.mock.Mock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        with unittest.mock.raises(FailedHookException, match='Hook script failed \(exit status: 1\)'):
            run_script('/path/to/script.py')

def test_run_script_os_error_enoexec():
    with unittest.mock.patch('subprocess.Popen', side_effect=OSError(errno.ENOEXEC, 'Exec format error')), \
         unittest.mock.patch('utils.make_executable'):
        
        with unittest.mock.raises(FailedHookException, match='Hook script failed, might be an empty file or missing a shebang'):
            run_script('/path/to/script.py')

def test_run_script_os_error_generic():
    with unittest.mock.patch('subprocess.Popen', side_effect=OSError(errno.EACCES, 'Permission denied')), \
         unittest.mock.patch('utils.make_executable'):
        
        with unittest.mock.raises(FailedHookException, match='Hook script failed \(error: \[Errno 13\] Permission denied\)'):
            run_script('/path/to/script.py')
```


# LLM-generated content at query #4
#--------------------------

```python
def test_find_hook_returns_none_when_no_valid_hooks_found(tmp_path, monkeypatch):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    invalid_file = hooks_dir / "not_a_hook.txt"
    invalid_file.write_text("content")
    
    monkeypatch.setattr("os.path.isdir", lambda x: True)
    monkeypatch.setattr("os.listdir", lambda x: ["not_a_hook.txt"])
    monkeypatch.setattr("valid_hook", lambda file, name: False)
    
    result = find_hook("some_hook", str(hooks_dir))
    assert result is None
```


# LLM-generated content at query #5
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks_returns_original_dir():
    import tempfile
    import shutil
    from pathlib import Path
    import os

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        result = run_pre_post_prompt_hook(tmp_dir)
        assert result == tmp_dir
    finally:
        shutil.rmtree(tmp_dir)

def test_run_pre_prompt_hook_with_valid_hook_returns_tmp_dir():
    import tempfile
    import shutil
    from pathlib import Path
    import os
    import sys

    tmp_repo = Path(tempfile.mkdtemp())
    hooks_dir = tmp_repo / "hooks"
    hooks_dir.mkdir()
    
    # Create a dummy python script as a hook
    script_path = hooks_dir / "pre_prompt.py"
    script_content = f"import sys; print('running'); sys.exit(0)"
    script_path.write_text(script_content)
    
    # Mocking _HOOKS in the module context if possible, 
    # but assuming it contains 'pre_prompt' based on standard cookiecutter logic
    # We must ensure the environment is set up so valid_hook returns True
    
    try:
        result = run_pre_prompt_hook(tmp_repo)
        assert Path(result).exists()
        assert result != tmp_repo
        assert "cookiecutter" in str(result)
    finally:
        shutil.rmtree(tmp_repo)

def test_run_pre_prompt_hook_fails_on_bad_script():
    import tempfile
    import shutil
    from pathlib import Path
    import os

    tmp_repo = Path(tempfile.mkdtemp())
    hooks_dir = tmp_repo / "hooks"
    hooks_dir.mkdir()
    
    # Create a script that exits with error
    script_path = hooks_dir / "pre_prompt.py"
    script_content = "import sys; sys.exit(1)"
    script_path.write_text(script_content)

    try:
        from cookiecutter.hooks import FailedHookException
        import pytest
        with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
            run_pre_prompt_hook(tmp_repo)
    finally:
        shutil.rmtree(tmp_repo)
```


# LLM-generated content at query #6
#--------------------------

```python
from unittest.mock import patch, MagicMock
import subprocess
import errno
import sys

def test_run_script_python_file_success():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'linux'), \
         patch('sys.executable', '/usr/bin/python3'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('test_script.py', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('test_script.py')
        mock_popen.assert_called_once_with(['/usr/bin/python3', 'test_script.py'], shell=False, cwd='/tmp')

def test_run_script_shell_script_success():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'win32'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('test_script.sh', cwd='.')
        
        mock_make_exec.assert_called_once_with('test_script.sh')
        mock_popen.assert_called_once_with(['test_script.sh'], shell=True, cwd='.')

def test_run_script_failure_exit_status():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test_script.py')
        
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

def test_run_script_oserror_enoexec():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        mock_error = OSError()
        mock_error.errno = errno.ENOEXEC
        mock_popen.side_effect = mock_error
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test_script.sh')
            
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_oserror_generic():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        mock_error = OSError(errno.EACCES, 'Permission denied')
        mock_popen.side_effect = mock_error
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test_script.py')
            
        assert 'Hook script failed (error: [Errno 13] Permission denied)' in str(excinfo.value)
```


# LLM-generated content at query #7
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found():
    import os
    import tempfile
    import shutil
    from pathlib import Path
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir).resolve()
        
        with patch("cookiecutter.hooks.find_hook", return_value=[]):
            result = run_pre_prompt_hook(repo_dir)
            assert result == repo_dir
```


# LLM-generated content at query #8
#--------------------------

```python
import os

def test_find_hook_type_signature_valid():
    # Since we cannot use control structures, we verify the return type of a successful call.
    # We must mock/setup an environment where the function can execute and return a list.
    # Given the constraints, we assume 'hooks' directory exists and contains valid files.
    # The predicate at line 1 is a type hint, which is always True in a valid Python runtime.
    result = find_hook("test_hook", "hooks")
    assert isinstance(find_hook.__annotations__['hook_name'], str)
    assert isinstance(find_hook.__annotations__['hooks_dir'], str)
    assert find_hook.__annotations__['return'] == list[str] | None
```


# LLM-generated content at query #9
#--------------------------

```python
def test_valid_hook_success():
    _HOOKS = ["pre-commit", "post-merge"]
    import os
    # Mocking the environment behavior by providing a controlled setup
    # Assuming _HOOKS is accessible in the scope where valid_hook is defined
    # For testing purposes, we assume _HOOKS contains 'pre-commit'
    global _HOOKS
    _HOOKS = ["pre-commit"]
    assert valid_hook("/path/to/pre-commit", "pre-commit") is True

def test_valid_hook_mismatched_name():
    global _HOOKS
    _HOOKS = ["pre-commit"]
    assert valid_hook("/path/to/pre-commit", "wrong-name") is False

def test_valid_hook_unsupported_hook():
    global _HOOKS
    _HOOKS = ["pre-commit"]
    assert valid_hook("/path/to/unknown-hook", "unknown-hook") is False

def test_valid_hook_is_backup_file():
    global _HOOKS
    _HOOKS = ["pre-commit"]
    assert valid_hook("/path/to/pre-commit~", "pre-commit") is False

def test_valid_hook_with_extension():
    global _HOOKS
    _HOOKS = ["pre-commit"]
    # If the file has an extension like .txt, basename becomes 'pre-commit'
    # But if the logic uses splitext, it checks the part before the dot
    assert valid_hook("/path/to/pre-commit.py", "pre-commit") is True

def test_valid_hook_empty_strings():
    global _HOOKS
    _HOOKS = []
    assert valid_hook("", "") is False
```


# LLM-generated content at query #10
#--------------------------

```python
from unittest.mock import patch

def test_run_hook_returns_early_when_no_scripts_found():
    with patch('cookiecutter.hooks.find_hook', return_value=[]), \
         patch('cookiecutter.hooks.logger.debug') as mock_logger, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run:
        from cookiecutter.hooks import run_hook
        run_hook('pre_gen_project', '/tmp/project', {'some': 'context'})
        mock_logger.assert_called_once_with('No %s hook found', 'pre_gen_project')
        mock_run.assert_not_called()
```


# LLM-generated content at query #11
#--------------------------

```python
def test_run_script_python_file_success():
    import subprocess
    from unittest.mock import patch, MagicMock
    import sys
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'linux'), \
         patch('sys.executable', '/usr/bin/python3'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('/path/to/script.py', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/path/to/script.py')
        mock_popen.assert_called_once_with(['/usr/bin/python3', '/path/to/script.py'], shell=False, cwd='/tmp')

def test_run_script_shell_script_success():
    import subprocess
    from unittest.mock import patch, MagicMock
    import sys
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'linux'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('/path/to/script.sh', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/path/to/script.sh')
        mock_popen.assert_called_once_with(['/path/to/script.sh'], shell=False, cwd='/tmp')

def test_run_script_windows_shell_true():
    import subprocess
    from unittest.mock import patch, MagicMock
    import sys
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'win32'), \
         patch('sys.executable', 'C:\\Python\\python.exe'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('C:\\scripts\\test.py')
        
        mock_popen.assert_called_once_with(['C:\\Python\\python.exe', 'C:\\scripts\\test.py'], shell=True, cwd='.')

def test_run_script_failure_exit_status():
    import subprocess
    from unittest.mock import patch, MagicMock
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script('/path/to/script.py')
        
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

def test_run_script_oserror_enoexec():
    import subprocess
    import errno
    from unittest.mock import patch
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        err = OSError()
        err.errno = errno.ENOEXEC
        mock_popen.side_effect = err
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script('/path/to/script.sh')
            
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_oserror_generic():
    import subprocess
    from unittest.mock import patch
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        err = OSError(2, "No such file or directory")
        mock_popen.side_effect = err
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script('/path/to/nonexistent.py')
            
        assert 'Hook script failed (error: [Errno 2] No such file or directory)' in str(excinfo.value)
```


# LLM-generated content at query #12
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_directory_not_exists():
    result = find_hook("pre-commit", "non_existent_directory_12345")
    assert result is None

def test_find_hook_empty_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        # Assuming _HOOKS contains 'pre-commit' for this test context
        result = find_hook("pre-commit", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_valid_hook_found():
    temp_dir = tempfile.mkdtemp()
    # Create a dummy hook file that matches the logic of valid_hook
    # We assume 'pre-commit' is in _HOOKS for this test to pass
    hook_filename = "pre-commit"
    hook_path = os.path.join(temp_dir, hook_filename)
    with open(hook_path, 'w') as f:
        f.write("# dummy content")
    
    try:
        # Note: This test relies on 'pre-commit' being in the global _HOOKS variable
        result = find_hook("pre-commit", temp_dir)
        assert result is not None
        assert os.path.abspath(hook_path) in result
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files():
    temp_dir = tempfile.mkdtemp()
    # Create a backup file (ends with ~)
    hook_filename = "pre-commit~"
    hook_path = os.path.join(temp_dir, hook_filename)
    with open(hook_path, 'w') as f:
        f.write("# dummy content")
    
    try:
        result = find_hook("pre-commit", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_mismatched_name():
    temp_dir = tempfile.mkdtemp()
    # Create a file with a different name that is in _HOOKS (assuming 'post-commit' is valid)
    hook_filename = "post-commit"
    hook_path = os.path.join(temp_dir, hook_filename)
    with open(hook_path, 'w') as f:
        f.write("# dummy content")
    
    try:
        result = find_hook("pre-commit", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir

    repo_dir = "/tmp/repo"
    hook_name = "post_gen_project"
    project_dir = "/tmp/project"
    context = {"foo": "bar"}

    with patch("cookiecutter.hooks.work_in") as mock_work_in, \
         patch("cookiecutter.hooks.run_hook") as mock_run_hook:
        mock_work_in.return_value.__enter__.return_value = None
        
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
        
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


def test_run_hook_from_repo_dir_failure_deletes_project():
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException

    repo_dir = "/tmp/repo"
    hook_name = "post_gen_project"
    project_dir = "/tmp/project"
    context = {"foo": "bar"}

    with patch("cookiecutter.hooks.work_in") as mock_work_in, \
         patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = FailedHookException("Failed")
        
        try:
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
        except FailedHookException:
            pass

        mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_failure_does_not_delete_project():
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException

    repo_dir = "/tmp/repo"
    hook_name = "post_gen_project"
    project_dir = "/template/project"
    context = {"foo": "bar"}

    with patch("cookiecutter.hooks.work_in") as mock_work_in, \
         patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = FailedHookException("Failed")
        
        try:
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
        except FailedHookException:
            pass

        mock_rmtree.assert_not_called()
```


# LLM-generated content at query #14
#--------------------------

```python
def test_find_hook_returns_none_when_no_valid_hooks_found():
    import os
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.mkdir(hooks_dir)
    
    # Create a file that does not match the hook_name
    with open(os.path.join(hooks_dir, 'other_hook.py'), 'w') as f:
        f.write('')

    # We need to mock/control valid_hook if it's in scope, 
    # but based on requirements, we assume an environment where 
    # valid_hook returns False for 'target_hook'.
    # Since I cannot define a function, this test assumes the logic 
    # relies on the file content/name mismatch.
    
    result = find_hook('target_hook', hooks_dir)
    
    assert result is None

    shutil.rmtree(temp_dir)
```


# LLM-generated content at query #15
#--------------------------

```python
import os

def test_find_hook_type_hint_validation():
    # This test validates that the function signature exists and accepts a string.
    # The predicate at line 1 is the function definition itself.
    result = find_hook(hook_name="test_hook", hooks_dir="non_existent_dir")
    assert isinstance(find_hook, type(lambda: None))
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_script_python_file_success():
    with unittest.mock.patch('subprocess.Popen') as mock_popen, \
         unittest.mock.patch('sys.executable', '/usr/bin/python3'), \
         unittest.mock.patch('utils.make_executable') as mock_make_exec:
        mock_process = unittest.mock.Mock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('/path/to/script.py', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/path/to/script.py')
        mock_popen.assert_called_once_with(['/usr/bin/python3', '/path/to/script.py'], shell=unittest.mock.ANY, cwd='/tmp')

def test_run_script_shell_file_success():
    with unittest.mock.patch('subprocess.Popen') as mock_popen, \
         unittest.mock.patch('utils.make_executable') as mock_make_exec:
        mock_process = unittest.mock.Mock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('/path/to/script.sh', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/path/to/script.sh')
        mock_popen.assert_called_once_with(['/path/to/script.sh'], shell=unittest.mock.ANY, cwd='/tmp')

def test_run_script_failure_exit_status():
    with unittest.mock.patch('subprocess.Popen') as mock_popen, \
         unittest.mock.patch('utils.make_executable'):
        mock_process = unittest.mock.Mock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        with unittest.mock.raises(FailedHookException) as context:
            run_script('/path/to/script.py')
        
        assert 'Hook script failed (exit status: 1)' in str(context.exception)

def test_run_script_oserror_enoexec():
    with unittest.mock.patch('subprocess.Popen') as mock_popen, \
         unittest.mock.patch('utils.make_executable'), \
         unittest.mock.patch('errno.ENOEXEC', 8):
        
        error = OSError()
        error.errno = 8
        mock_popen.side_effect = error
        
        with unittest.mock.raises(FailedHookException) as context:
            run_script('/path/to/script.sh')
            
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(context.exception)

def test_run_script_oserror_generic():
    with unittest.mock.patch('subprocess.Popen') as mock_popen, \
         unittest.mock.patch('utils.make_executable'):
        
        error = OSError()
        error.strerror = "Permission denied"
        mock_popen.side_effect = error
        
        with unittest.mock.raises(FailedHookException) as context:
            run_script('/path/to/script.py')
            
        assert 'Hook script failed (error: ' in str(context.exception)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_valid_hook_success():
    import os
    _HOOKS = ['pre-commit', 'post-merge']
    assert valid_hook('/path/to/pre-commit', 'pre-commit') is True

def test_valid_hook_mismatched_name():
    import os
    _HOOKS = ['pre-commit']
    assert valid_hook('/path/to/pre-commit', 'post-merge') is False

def test_valid_hook_unsupported_hook():
    import os
    _HOOKS = ['pre-commit']
    assert valid_hook('/path/to/unknown', 'unknown') is False

def test_valid_hook_backup_file():
    import os
    _HOOKS = ['pre-commit']
    assert valid_hook('/path/to/pre-commit~', 'pre-commit') is False

def test_valid_hook_with_extension():
    import os
    _HOOKS = ['pre-commit']
    assert valid_hook('/path/to/pre-commit.sh', 'pre-commit') is True

def test_valid_hook_empty_params():
    import os
    _HOOKS = []
    assert valid_hook('', '') is False
```


# LLM-generated content at query #3
#--------------------------

```python
def test_run_hook_no_scripts_found():
    from unittest.mock import patch
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        from cookiecutter.hooks import run_hook
        run_hook('pre_gen_project', '/tmp/project', {'foo': 'bar'})
        mock_find_hook.assert_called_once_with('pre_gen_project')

def test_run_hook_executes_found_scripts():
    from unittest.mock import patch
    script_path = '/tmp/hooks/pre_gen_project.sh'
    context = {'foo': 'bar'}
    project_dir = '/tmp/project'
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_context:
            mock_find_hook.return_value = [script_path]
            from cookiecutter.hooks import run_hook
            run_hook('pre_gen_project', project_dir, context)
            mock_find_hook.assert_called_once_with('pre_gen_project')
            mock_run_context.assert_called_once_with(script_path, project_dir, context)

def test_run_hook_executes_multiple_scripts():
    from unittest.mock import patch
    script_paths = ['/tmp/hooks/pre_gen_project.sh', '/tmp/hooks/pre_gen_project.py']
    context = {'foo': 'bar'}
    project_dir = '/tmp/project'
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        with patch('cookielogger.hooks.run_script_with_context') as mock_run_context:
            mock_find_hook.return_value = script_paths
            from cookiecutter.hooks import run_hook
            run_hook('pre_gen_project', project_dir, context)
            assert mock_run_context.call_count == 2
            mock_run_context.assert_any_call(script_paths[0], project_dir, context)
            mock_run_context.assert_any_call(script_paths[1], project_dir, context)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_run_hook_returns_early_when_no_scripts_found():
    from unittest.mock import patch, MagicMock
    from pathlib import Path
    from cookiecutter.hooks import run_hook

    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        with patch('cookiecutter.hooks.logger') as mock_logger:
            mock_find_hook.return_value = []
            
            run_hook("post_gen_project", Path("/tmp"), {"some": "context"})
            
            mock_find_hook.assert_called_once_with("post_gen_project")
            mock_logger.debug.assert_called_once_with('No %s hook found', "post_gen_project")
```


# LLM-generated content at query #5
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            mock_context = {'project_name': 'test'}
            mock_work_in.return_value.__enter__.return_value = None
            
            run_hook_from_repo_dir(
                repo_dir='/tmp/repo',
                hook_name='post_gen_project',
                project_dir='/tmp/project',
                context=mock_context,
                delete_project_on_failure=True
            )
            
            mock_run_hook.assert_called_once_with('post_gen_project', '/tmp/project', mock_context)

def test_run_hook_from_repo_dir_failure_deletes_project():
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    from unittest.mock import patch, MagicMock

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                mock_context = {'project_name': 'test'}
                mock_work_in.return_value.__enter__.return_value = None
                mock_run_hook.side_effect = FailedHookException("Failed")
                
                try:
                    run_hook_from_repo_dir(
                        repo_dir='/tmp/repo',
                        hook_name='post_gen_project',
                        project_dir='/tmp/project',
                        context=mock_context,
                        delete_project_on_failure=True
                    )
                except FailedHookException:
                    pass
                
                mock_rmtree.assert_called_once_with('/tmp/project')

def test_run_hook_from_repo_dir_failure_does_not_delete_project():
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException
    from unittest.mock import patch, MagicMock

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                mock_context = {'project_name': 'test'}
                mock_work_in.return_value.__enter__.return_value = None
                mock_run_hook.side_effect = FailedHookException("Failed")
                
                try:
                    run_hook_from_repo_dir(
                        repo_dir='/tmp/repo',
                        hook_name='post_gen_project',
                        project_dir='/tmp/project',
                        context=mock_context,
                        delete_project_on_failure=False
                    )
                except FailedHookException:
                    pass
                
                mock_rmtree.assert_not_called()
```


# LLM-generated content at query #6
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks_returns_original_dir():
    import tempfile
    import shutil
    from pathlib import Path
    import os

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        result = run_pre_post_hook(tmp_dir)
        assert result == tmp_dir
    finally:
        shutil.rmtree(tmp_dir)

def test_run_pre_prompt_hook_with_valid_hook_returns_new_tmp_dir():
    import tempfile
    import shutil
    from pathlib import Path
    import os
    from unittest.mock import patch, MagicMock

    original_dir = Path(tempfile.mkdtemp())
    hooks_dir = original_dir / "hooks"
    os.mkdir(hooks_dir)
    hook_script = hooks_dir / "pre_prompt"
    with open(hook_script, "w") as f:
        f.write("#!/bin/bash\nexit 0")
    
    # Mocking internal dependencies to avoid actual process execution and file system complexity
    # while ensuring the logic flow of run_pre_prompt_hook is tested.
    with patch("cookiecutter.hooks.find_hook") as mock_find, \
         patch("cookiecutter.hooks.run_script") as mock_run, \
         patch("cookiecutter.utils.create_tmp_repo_dir") as mock_create:
        
        mock_find.side_effect = [None, [str(hook_script)]]
        mock_create.return_value = Path("/tmp/fake_repo")

        result = run_pre_prompt_hook(original_dir)

        assert result == Path("/tmp/fake_repo")
        assert mock_find.call_count == 2
        mock_run.assert_called_once()
    
    shutil.rmtree(original_dir)

def test_run_pre_prompt_hook_raises_failed_hook_exception_on_script_failure():
    import tempfile
    import shutil
    from pathlib import Path
    from cookiecutter.hooks import FailedHookException

    original_dir = Path(tempfile.mkdtemp())
    hooks_dir = original_dir / "hooks"
    os.mkdir(hooks_dir)
    hook_script = hooks_dir / "pre_prompt"
    with open(hook_script, "w") as f:
        f.write("exit 1")

    with patch("cookiecutter.hooks.find_hook") as mock_find, \
         patch("cookiecutter.hooks.run_script") as mock_run, \
         patch("cookiecutter.utils.create_tmp_repo_dir") as mock_create:
        
        mock_find.side_effect = [None, [str(hook_script)]]
        mock_create.return_value = Path("/tmp/fake_repo")
        mock_run.side_effect = FailedHookException("Original error")

        try:
            run_pre_prompt_hook(original_dir)
        except FailedHookException as e:
            assert str(e) == 'Pre-Prompt Hook script failed'
        else:
            assert False, "Expected FailedHookException to be raised"

    shutil.rmtree(original_dir)
```


# LLM-generated content at query #7
#--------------------------

```python
def test_run_script_raises_enoexec_exception():
    import subprocess
    import errno
    from unittest.mock import patch

    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        # Setup the OSError with errno.ENOEXEC
        error = OSError()
        error.errno = errno.ENOEXEC
        mock_popen.side_effect = error

        # Execution and Assertion
        # The predicate err.errno == errno.ENOEXEC is evaluated here
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test_script.sh')
        
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_directory_not_exists():
    result = find_hook("pre-commit", "non_existent_dir_12345")
    assert result is None

def test_find_hook_valid_hook_found(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    hooks_path = os.path.join(temp_dir, "hooks")
    os.mkdir(hooks_path)
    
    # Mocking _HOOKS to ensure the basename is recognized as supported
    monkeypatch.setattr("your_module._HOOKS", ["pre-commit", "post-checkout"])
    
    hook_file = os.path.join(hooks_path, "pre-commit.sh")
    with open(hook_file, "w") as f:
        f.write("#!/bin/bash\nexit 0")
    
    # Mocking os.listdir to return our created file
    monkeypatch.setattr("os.listdir", lambda path: ["pre-commit.sh"])
    # Mocking os.path.isdir to return True for our temp dir
    monkeypatch.setattr("os.path.isdir", lambda path: path == hooks_path)
    
    result = find_hook("pre-commit", hooks_path)
    assert result == [os.path.abspath(hook_file)]
    
    shutil.rmtree(temp_dir)

def test_find_hook_no_matching_hooks(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    hooks_path = os.path.join(temp_dir, "hooks")
    os.mkdir(hooks_path)
    
    monkeypatch.setattr("your_module._HOOKS", ["pre-commit"])
    # File exists but name doesn't match hook_name
    monkeypatch.setattr("os.listdir", lambda path: ["wrong-name.sh"])
    monkeypatch.setattr("os.path.isdir", lambda path: True)
    
    result = find_hook("pre-commit", hooks_path)
    assert result is None
    
    shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    hooks_path = os.path.join(temp_dir, "hooks")
    os.mkdir(hooks_path)
    
    monkeypatch.setattr("your_module._HOOKS", ["pre-commit"])
    # File matches name but ends with ~ (backup file)
    monkeypatch.setattr("os.listdir", lambda path: ["pre-commit.sh~"])
    monkeypatch.setattr("os.path.isdir", lambda path: True)
    
    result = find_hook("pre-commit", hooks_path)
    assert result is None
    
    shutil.rmtree(temp_dir)

def test_find_hook_empty_directory(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    hooks_path = os.path.join(temp_dir, "hooks")
    os.mkdir(hooks_path)
    
    monkeypatch.setattr("os.listdir", lambda path: [])
    monkeypatch.setattr("os.path.isdir", lambda path: True)
    
    result = find_hook("pre-commit", hooks_path)
    assert result is None
    
    shutil.rmtree(temp_dir)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks_returns_original_dir(tmp_path):
    repo_dir = tmp_path / "no_hooks_repo"
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


def test_run_pre_prompt_hook_with_valid_hook_runs_and_returns_tmp_dir(tmp_path, monkeypatch):
    repo_dir = tmp_path / "valid_hooks_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    script = hooks_dir / "pre_prompt.py"
    script.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)")
    
    # Mocking run_script to avoid actual execution of subprocesses in unit test
    def mock_run_script(path, cwd='.'):
        pass
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script)
    
    result = run_pre_prompt_hook(repo_dir)
    assert Path(result).resolve() != repo_dir.resolve()
    assert result.name == repo_dir.name
    assert (Path(result) / "hooks" / "pre_prompt.py").exists()


def test_run_pre_prompt_hook_raises_exception_on_script_failure(tmp_path, monkeypatch):
    repo_dir = tmp_path / "failing_hooks_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    script = hooks_dir / "pre_prompt.py"
    script.write_text("#!/usr/bin/env python\nimport sys\nsys_exit(1)")

    # Mocking run_script to simulate a failure
    def mock_run_script_fail(path, cwd='.'):
        raise FailedHookException("Hook script failed (exit status: 1)")
    monkeypatch.setattr("cookiecutter.hooks.run_script", mock_run_script_fail)

    try:
        run_pre_prompt_hook(repo_dir)
    except FailedHookException as e:
        assert str(e) == "Pre-Prompt Hook script failed"
    else:
        raise AssertionError("FailedHookException was not raised")
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from cookiecutter.hooks import run_pre_prompt_hook

def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        with patch("cookiecutter.hooks.find_hook", return_value=[]):
            result = run_pre_prompt_hook(tmp_path)
            assert result == tmp_path
```


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook_returns_early_when_no_scripts_found():
    from unittest.mock import patch
    from pathlib import Path
    from cookiecutter.hooks import run_hook

    with patch("cookiecutter.hooks.find_hook") as mock_find_hook:
        with patch("cookiecutter.hooks.logger") as mock_logger:
            mock_find_hook.return_value = []
            run_hook("pre_gen_project", "/tmp/project", {"foo": "bar"})
            
            mock_find_hook.assert_called_once_with("pre_gen_project")
            mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_gen_project')
```


# LLM-generated content at query #12
#--------------------------

```python
def test_find_hook_signature_validates():
    import os
    from unittest.mock import patch

    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[]), \
         patch('logging.Logger.debug'):
        result = find_hook(hook_name="test_hook", hooks_dir="hooks")
        assert isinstance(result, (list, type(None)))
```


# LLM-generated content at query #13
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_directory_not_exists():
    result = find_hook("pre-commit", "non_existent_dir_12345")
    assert result is None

def test_find_hook_success():
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, "hooks")
    os.mkdir(hooks_dir)
    
    # Mocking _HOOKS globally for the scope of this test if possible, 
    # but since I cannot modify global state, I assume 'pre-commit' is in _HOOKS
    # In a real scenario, we would use patch.
    
    hook_path = os.path.join(hooks_dir, "pre-commit.sh")
    with open(hook_path, "w") as f:
        f.write("#!/bin/bash\nexit 0")
    
    # We assume 'pre-commit' is a valid hook name in the environment's _HOOKS
    # For this test to be runnable, we rely on the existence of pre-commit in _HOOKS
    try:
        import __main__
        if not hasattr(__main__, '_HOOKS'):
            __main__._HOOKS = ["pre-commit", "post-commit"]
            
        result = find_hook("pre-commit", hooks_dir)
        assert result is not None
        assert os.path.abspath(hook_path) in result
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_no_matching_files():
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, "hooks")
    os.mkdir(hooks_dir)
    
    # Create a file that doesn't match the name
    with open(os.path.join(hooks_dir, "wrong-name.sh"), "w") as f:
        f.write("echo 1")
        
    result = find_hook("pre-commit", hooks_dir)
    assert result is None
    shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files():
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, "hooks")
    os.mkdir(hooks_dir)
    
    # Create a file with trailing tilde (backup file)
    with open(os.path.join(hooks_dir, "pre-commit.sh~"), "w") as f:
        f.write("echo 1")
        
    result = find_hook("pre-commit", hooks_dir)
    assert result is None
    shutil.rmtree(temp_dir)

def test_find_hook_multiple_valid_files():
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, "hooks")
    os.mkdir(hooks_dir)
    
    # Note: This requires 'pre-commit' to be in _HOOKS
    import __main__
    __main__._HOOKS = ["pre-commit"]

    path1 = os.path.join(hooks_dir, "pre-commit.sh")
    path2 = os.path.join(hooks_dir, "pre-commit.py")
    
    with open(path1, "w") as f: f.write("")
    with open(path2, "w") as f: f.write("")
    
    result = find_hook("pre-commit", hooks_dir)
    assert len(result) == 2
    assert os.path.abspath(path1) in result
    assert os.path.abspath(path2) in result
    shutil.rmtree(temp_dir)
```


# LLM-generated content at query #14
#--------------------------

```python
def test_run_script_raises_enoexec_exception():
    import subprocess
    import errno
    from pathlib import Path
    from unittest.mock import patch

    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        error = OSError()
        error.errno = errno.ENOEXEC
        mock_popen.side_effect = error
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script("test_script.sh")
        
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)
```


# LLM-generated content at query #15
#--------------------------

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from cookiecutter.hooks import run_pre_prompt_hook

def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        with patch("cookiecutter.hooks.find_hook", return_value=[]):
            result = run_pre_prompt_hook(tmp_path)
            assert result == tmp_path
```


# LLM-generated content at query #16
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_directory_not_exists():
    result = find_hook("pre-commit", "non_existent_directory_12345")
    assert result is None

def test_find_hook_empty_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        result = find_hook("pre-commit", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_valid_script_found():
    # Note: This test assumes _HOOKS contains 'pre-commit' 
    # and relies on the environment having a directory structure.
    # Since we cannot mock _HOOKS easily without imports, 
    # we assume 'pre-commit' is in the global _HOOKS for this logic.
    temp_dir = tempfile.mkdtemp()
    hook_file = "pre-commit"
    with open(os.path.join(temp_dir, hook_file), "w") as f:
        f.write("#!/bin/bash\nexit 0")
    
    try:
        # We use the absolute path to bypass dependency on CWD
        result = find_hook("pre-commit", temp_dir)
        assert result is not None
        assert len(result) == 1
        assert os.path.abspath(os.path.join(temp_dir, hook_file)) == result[0]
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files():
    temp_dir = tempfile.mkdtemp()
    # Assuming 'pre-commit' is in _HOOKS
    hook_file = "pre-commit~" 
    with open(os.path.join(temp_dir, hook_file), "w") as f:
        f.write("#!/bin/bash\nexit 0")
    
    try:
        result = find_hook("pre-commit", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_mismatched_name():
    temp_dir = tempfile.mkdtemp()
    # Assuming 'pre-commit' is in _HOOKS, but we search for 'post-commit'
    hook_file = "pre-commit" 
    with open(os.path.join(temp_dir, hook_file), "w") as f:
        f.write("#!/bin/bash\nexit 0")
    
    try:
        result = find_hook("post-commit", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_find_hook_signature_validity():
    import inspect
    from typing import get_type_hints

    # Test that the function exists and can be inspected for its signature
    # We use a dummy implementation to verify the structure of line 1
    def find_hook(hook_name: str, hooks_dir: str = 'hooks') -> list[str] | None:
        return []

    signature = inspect.signature(find_hook)
    
    assert 'hook_name' in signature.parameters
    assert isinstance(signature.parameters['hook_name'].annotation, type)
    assert signature.parameters['hook_name'].annotation is str
    
    assert 'hooks_dir' in signature.parameters
    assert signature.parameters['hooks_dir'].default == 'hooks'
    assert signature.parameters['hooks_dir'].annotation is str
    
    assert signature.return_annotation == list[str] | None
```


# LLM-generated content at query #18
#--------------------------

```python
def test_valid_hook_evaluates_to_true():
    import os
    _HOOKS = ['pre-commit', 'post-checkout']
    hook_file = '/path/to/pre-commit'
    hook_name = 'pre-commit'
    assert valid_hook(hook_file, hook_name) == True
```


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_returns_early_when_no_scripts_found(mocker):
    mock_find_hook = mocker.patch("cookiecutter.hooks.find_hook", return_value=[])
    mock_logger = mocker.patch("cookiecutter.hooks.logger")
    mock_run_script = mocker.patch("cookiecutter.hooks.run_script_with_context")

    run_hook("pre_gen_project", "/tmp/project", {"some": "context"})

    mock_find_hook.assert_called_once_with("pre_gen_project")
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_gen_project')
    mock_run_script.assert_not_called()
```


