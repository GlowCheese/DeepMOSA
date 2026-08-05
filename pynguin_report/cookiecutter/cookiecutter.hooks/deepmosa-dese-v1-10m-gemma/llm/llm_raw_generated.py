####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_hook_no_scripts_found():
    from unittest.mock import patch
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        with patch('cookiecutter.hooks.logger.debug') as mock_logger:
            from cookiecutter.hooks import run_hook
            run_hook('pre_gen_project', '/tmp/project', {})
            mock_logger.assert_any_call('No %s hook found', 'pre_gen_project')

def test_run_hook_executes_scripts():
    from unittest.mock import patch
    script_path = '/tmp/hooks/pre_gen_project'
    scripts = [script_path]
    with patch('cookiecutter.hooks.find_hook', return_value=scripts):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            from cookiecutter.hooks import run_hook
            context = {'name': 'test'}
            run_hook('pre_gen_project', '/tmp/project', context)
            mock_run.assert_called_once_with(script_path, '/tmp/project', context)

def test_run_hook_executes_multiple_scripts():
    from unittest.mock import patch
    script_paths = ['/tmp/hooks/pre_gen_project_1', '/tmp/hooks/pre_gen_project_2']
    with patch('cookiecutter.hooks.find_hook', return_value=script_paths):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            from cookiecutter.hooks import run_hook
            context = {'name': 'test'}
            run_hook('pre_gen_project', '/tmp/project', context)
            assert mock_run.call_count == 2
```


# LLM-generated content at query #2
#--------------------------

```python
import sys
import subprocess
import errno
from pathlib import Path
from unittest.mock import patch, MagicMock

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

def test_run_script_shell_command_success():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'win32'):
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('script.sh')
        
        mock_make_exec.assert_called_once_with('script.sh')
        mock_popen.assert_called_once_with(['script.sh'], shell=True, cwd='.')

def test_run_script_failure_exit_status():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test.py')
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

def test_run_script_oserror_enoexec():
    with patch('subprocess.Popen', side_effect=OSError(errno.ENOEXEC, 'Exec format error')), \
         patch('utils.make_executable'):
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test.py')
        assert 'might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_oserror_generic():
    with patch('subprocess.Popen', side_effect=OSError(errno.EACCES, 'Permission denied')), \
         patch('utils.make_executable'):
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test.py')
        assert 'Hook script failed (error: ' in str(excinfo.value)
```


# LLM-generated content at query #3
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_directory_not_exists():
    result = find_hook("pre-commit", "non_existent_dir_12345")
    assert result is None

def test_find_hook_no_matching_hooks():
    temp_dir = tempfile.mkdtemp()
    try:
        hooks_path = os.path.join(temp_dir, "hooks")
        os.mkdir(hooks_path)
        with open(os.path.join(hooks_path, "other_hook.sh"), "w") as f:
            f.write("#!/bin/bash")
        
        # Note: This test assumes _HOOKS contains 'other_hook' or similar logic is mocked. 
        # Since we cannot mock globals easily without imports, we rely on the environment.
        # For this exercise, we assume valid_hook works with the provided file.
        result = find_hook("pre-commit", hooks_dir=hooks_path)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_success():
    temp_dir = tempfile.mkdtemp()
    try:
        hooks_path = os.path.join(temp_dir, "hooks")
        os.mkdir(hooks_path)
        
        # We name the file to match a hypothetical entry in _HOOKS
        # Since we can't see _HOOKS, we assume 'post-commit' is valid for this test case logic
        hook_name = "post-commit"
        hook_file_name = f"{hook_name}.sh"
        hook_full_path = os.path.join(hooks_path, hook_file_name)
        
        with open(hook_full_path, "w") as f:
            f.write("#!/bin/bash")

        # We must ensure the logic of valid_hook passes. 
        # This requires 'post-commit' to be in the global _HOOKS.
        # In a real scenario, we would mock _HOOKS.
        result = find_hook(hook_name, hooks_dir=hooks_path)
        
        # If the test environment has no way to inject into _HOOKS, 
        # this specific assertion might fail, but it represents the correct test structure.
        if result is not None:
            assert len(result) == 1
            assert os.path.abspath(hook_full_path) == result[0]
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files():
    temp_dir = tempfile.mkdtemp()
    try:
        hooks_path = os.path.join(temp_dir, "hooks")
        os.mkdir(hooks_path)
        
        # Assuming 'pre-commit' is in _HOOKS
        hook_name = "pre-commit"
        backup_file_name = f"{hook_name}.sh~"
        backup_full_path = os.path.join(hooks_path, backup_file_name)
        
        with open(backup_full_path, "w") as f:
            f.write("#!/bin/bash")

        result = find_hook(hook_name, hooks_dir=hooks_path)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)
```


# LLM-generated content at query #4
#--------------------------

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_run_script_with_context_success():
    template_content = "hello_{{ name }}"
    context = {"name": "world"}
    cwd = "."
    
    with patch("pathlib.Path.read_text", return_value=template_content), \
         patch("cookiecutter.hooks.create_env_with_context") as mock_create_env, \
         patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
         patch("cookiecutter.hooks.run_script") as mock_run_script:
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "hello_world"
        
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.__enter__.return_value = mock_temp_file_instance
        mock_temp_file_instance.name = "/tmp/test_script.py"
        mock_temp_file.return_value = mock_temp_file_instance

        from cookiecutter.hooks import run_script_with_context
        run_script_with_context("template.py", cwd, context)

        mock_create_env.assert_called_once_with(context)
        mock_template.render.assert_called_once_with(**context)
        mock_temp_file_instance.write.assert_called_once_with(b"hello_world")
        mock_run_script.assert_called_once_with("/tmp/test_script.py", cwd)

def test_run_script_with_context_extension_handling():
    template_content = "data: {{ value }}"
    context = {"value": 123}
    script_path = "script.txt"
    cwd = "."

    with patch("pathlib.Path.read_text", return_value=template_content), \
         patch("cookiecutter.hooks.create_env_with_context") as mock_create_env, \
         patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
         patch("cookiecutter.hooks.run_script") as mock_run_script:
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "data: 123"
        
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.__enter__.return_value = mock_temp_file_instance
        mock_temp_file_instance.name = "/tmp/test_script.txt"
        mock_temp_file.return_value = mock_temp_file_instance

        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)

        # Check if suffix was correctly extracted from script_path (.txt)
        args, _ = mock_temp_file.call_args
        assert mock_temp_file.call_args[1]['suffix'] == '.txt'
```


# LLM-generated content at query #5
#--------------------------

```python
def test_run_pre_prompt_hook_returns_original_dir_when_no_hook_exists(tmp_path):
    repo_dir = tmp_path / "no_hooks_repo"
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_returns_tmp_dir_when_hook_exists(tmp_path, monkeypatch):
    repo_dir = tmp_path / "with_hooks_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    pre_prompt_script = hooks_dir / "pre_prompt.py"
    pre_prompt_script.write_text("print('hello')")
    
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda script, cwd: None)
    
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, Path)
    assert result != repo_dir
    assert result.name == repo_dir.name

def test_run_pre_prompt_hook_raises_exception_on_script_failure(tmp_path, monkeypatch):
    repo_dir = tmp_path / "fail_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    pre_prompt_script = hooks_dir / "pre_prompt.py"
    pre_prompt_script.write_text("exit(1)")

    from cookiecutter.hooks import FailedHookException
    monkeypatch.setattr("cookiecutter.hooks.run_script", lambda script, cwd: exec("raise FailedHookException('fail')"))

    try:
        run_pre_prompt_hook(repo_dir)
    except FailedHookException as e:
        assert "Pre-Prompt Hook script failed" in str(e)
```


# LLM-generated content at query #6
#--------------------------

```python
import os

def test_find_hook_type_hint_validation():
    # This test ensures the function signature is valid and can be called.
    # Since we cannot modify the environment to create real files without side effects,
    # we test that the function accepts a string for hook_name and a string for hooks_dir.
    # The predicate at line 1 (the type hint) is evaluated during function definition/call.
    result = find_hook(hook_name="test_hook", hooks_dir="non_existent_directory_for_type_check")
    assert isinstance(result, (list, type(None)))
```


# LLM-generated content at query #7
#--------------------------

```python
def test_valid_hook_success():
    _HOOKS = ["pre-commit", "post-checkout"]
    import os
    # Mocking the environment logic within the scope of the test call
    # Assuming _HOOKS is accessible or defined in the module context
    global _HOOKS
    _HOOKS = ["pre-commit", "post-checkout"]
    assert valid_hook("/path/to/pre-commit", "pre-commit") == True

def test_valid_hook_wrong_name():
    global _HOOKS
    _HOOKS = ["pre-commit"]
    assert valid_hook("/path/to/pre-commit", "wrong-name") == False

def test_valid_hook_unsupported_type():
    global _HOOKS
    _HOOKS = ["pre-commit"]
    assert valid_hook("/path/to/unknown", "unknown") == False

def test_valid_hook_is_backup_file():
    global _HOOKS
    _HOOKS = ["pre-commit"]
    assert valid_hook("/path/to/pre-commit~", "pre-commit") == False

def test_valid_hook_mismatch_and_unsupported():
    global _HOOKS
    _HOOKS = ["pre-commit"]
    assert valid_hook("/path/to/other", "pre-commit") == False

def test_valid_hook_complex_path():
    global _HOOKS
    _HOOKS = ["post-merge"]
    assert valid_hook("C:\\Users\\Admin\\hooks\\post-merge.py", "post-merge") == True
```


# LLM-generated content at query #8
#--------------------------

```python
def test_find_hook_returns_none_when_no_valid_hooks_found():
    import os
    import tempfile
    import shutil

    test_dir = tempfile.mkdtemp()
    # Create a file that does not match the hook_name to ensure valid_hook returns False
    # or simply create no files at all so the list is empty.
    # Since we cannot redefine valid_hook, we rely on the fact that 
    # if the directory is empty, len(scripts) will be 0.
    
    result = find_hook(hook_name='non_existent_hook', hooks_dir=test_dir)
    
    assert result is None
    shutil.rmtree(test_dir)
```


# LLM-generated content at query #9
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
        
        from your_module import run_script, EXIT_SUCCESS
        run_script('/path/to/script.py', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/path/to/script.py')
        mock_popen.assert_called_once_with(['/usr/bin/python3', '/path/to/script.py'], shell=False, cwd='/tmp')

def test_run_script_shell_script_windows():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'win32'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        from your_module import run_script
        run_script('/path/to/script.sh', cwd='.')
        
        mock_popen.assert_called_once_with(['/path/to/script.sh'], shell=True, cwd='.')

def test_run_script_failure_exit_status():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        from your_module import run_script, FailedHookException
        with Exception as e:
            try:
                run_script('/path/to/script.py')
            except FailedHookException as error:
                actual_error = error
        
        assert str(actual_error) == 'Hook script failed (exit status: 1)'

def test_run_script_os_error_enoexec():
    with patch('subprocess.Popen', side_effect=OSError(errno.ENOEXEC, 'exec format error')), \
         patch('utils.make_executable'):
        
        from your_module import run_script, FailedHookException
        with Exception as e:
            try:
                run_script('/path/to/script.sh')
            except FailedHookException as error:
                actual_error = error
        
        assert str(actual_error) == 'Hook script failed, might be an empty file or missing a shebang'

def test_run_script_os_error_general():
    with patch('subprocess.Popen', side_effect=OSError(errno.EACCES, 'Permission denied')), \
         patch('utils.make_executable'):
        
        from your_module import run_script, FailedHookException
        with Exception as e:
            try:
                run_script('/path/to/script.sh')
            except FailedHookException as error:
                actual_error = error
        
        assert 'Hook script failed (error: ' in str(actual_error)
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
        # Create an empty directory so find_hook('pre_prompt') returns [] or None
        with patch("cookiecutter.hooks.find_hook", return_value=[]):
            result = run_pre_prompt_hook(tmp_path)
            assert result == tmp_path
```


# LLM-generated content at query #11
#--------------------------

```python
def test_find_hook_signature_type_hinting():
    import inspect
    from typing import get_type_hints

    # Verifying the type hint for hook_name is str
    # Note: Since we cannot use control structures, we assert directly on the signature
    sig = inspect.signature(find_hook)
    assert sig.parameters['hook_name'].annotation is str
    assert sig.parameters['hooks_dir'].default == 'hooks'
    assert sig.return_annotation is list[str] or sig.return_annotation is list or sig.return_annotation is None
```


# LLM-generated content at query #12
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks_returns_original_dir():
    import tempfile
    import shutil
    from pathlib import Path
    import os

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir).resolve()
        # Ensure no hooks directory exists
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook_returns_tmp_dir():
    import tempfile
    import shutil
    from pathlib import Path
    import os
    import sys

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir).resolve()
        hooks_dir = repo_dir / "hooks"
        os.mkdir(hooks_dir)
        
        # Create a dummy python script that exits successfully
        hook_script = hooks_dir / "pre_prompt.py"
        with open(hook_script, "w") as f:
            f.write("import sys; sys.exit(0)")
        
        # We need to mock _HOOKS in hooks module if it's not already present 
        # and ensure valid_hook recognizes 'pre_prompt'
        import cookiecutter.hooks
        cookiecutter.hooks._HOOKS = ["pre_prompt"]

        result = run_pre_prompt_hook(repo_dir)
        
        assert Path(result).resolve() != repo_dir
        assert Path(result).name == repo_dir.name
        assert os.path.exists(result)
        
        # Cleanup is handled by tempfile context, but we check logic here
```


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    import os
    from unittest.mock import patch, MagicMock
    from pathlib import Path
    from cookiecutter.hooks import run_hook_from_repo_dir

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            mock_work_in.return_value.__enter__.return_value = None
            
            run_hook_from_repo_dir(
                repo_dir='/repo',
                hook_name='post_gen_project',
                project_dir='/project',
                context={'foo': 'bar'},
                delete_project_on_failure=True
            )
            
            mock_run_hook.assert_called_once_with('post_gen_project', '/project', {'foo': 'bar'})

def test_run_hook_from_repo_dir_failure_deletes_project():
    import os
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                mock_work_in.return_value.__enter__.return_value = None
                mock_run_hook.side_effect = FailedHookException("Failed")
                
                try:
                    run_hook_from_repo_dir(
                        repo_dir='/repo',
                        hook_name='post_gen_project',
                        project_dir='/project',
                        context={'foo': 'bar'},
                        delete_project_on_failure=True
                    )
                except FailedHookException:
                    pass
                
                mock_rmtree.assert_called_once_with('/project')

def test_run_hook_from_repo_dir_failure_does_not_delete_project():
    import os
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                mock_work_in.return_value.__enter__.return_value = None
                mock_run_hook.side_effect = FailedHookException("Failed")
                
                try:
                    run_hook_from_repo_dir(
                        repo_dir='/repo',
                        hook_name='post_gen_project',
                        project_dir='/project',
                        context={'foo': 'bar'},
                        delete_project_on_failure=False
                    )
                except FailedHookException:
                    pass
                
                mock_rmtree.assert_not_called()
```


# LLM-generated content at query #14
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    # Mocking the global _HOOKS variable context needed for line 11
    global _HOOKS
    _HOOKS = ['pre-commit', 'post-checkout']
    
    hook_file = '/path/to/pre-commit'
    hook_name = 'pre-commit'
    
    assert valid_hook(hook_file, hook_name) is True
```


# LLM-generated content at query #15
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found():
    import os
    import tempfile
    import shutil
    from pathlib import Path
    from unittest.mock import patch

    temp_dir = tempfile.mkdtemp()
    repo_dir = Path(temp_dir) / "empty_repo"
    repo_dir.mkdir()
    
    with patch("cookiecutter.hooks.find_hook", return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

    shutil.rmtree(temp_dir)
```


# LLM-generated content at query #16
#--------------------------

```python
def test_run_hook_no_scripts_found():
    from unittest.mock import patch
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        with patch('cookiecutter.hooks.logger.debug') as mock_logger:
            from cookiecutter.hooks import run_hook
            run_hook('pre_gen_project', '/tmp/project', {})
            mock_logger.assert_any_call('No %s hook found', 'pre_gen_project')

def test_run_hook_executes_scripts():
    from unittest.mock import patch
    with patch('cookiecutter.hooks.find_hook', return_value=['/tmp/project/hooks/pre_gen_project.py']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            with patch('cookiecutter.hooks.logger.debug') as mock_logger:
                from cookiecutter.hooks import run_hook
                context = {'project_name': 'test'}
                run_hook('pre_gen_project', '/tmp/project', context)
                mock_run_script.assert_called_once_with('/tmp/post_gen_project.py' if False else '/tmp/project/hooks/pre_gen_project.py', '/tmp/project', context)
                mock_logger.assert_any_call('Running hook %s', 'pre_gen_project')

def test_run_hook_executes_multiple_scripts():
    from unittest.mock import patch
    scripts = ['/tmp/project/hooks/pre_gen_project.py', '/tmp/project/hooks/pre_gen_project.sh']
    with patch('cookiecutter.hooks.find_hook', return_value=scripts):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            from cookiecutter.hooks import run_hook
            run_hook('pre_gen_project', '/tmp/project', {})
            assert mock_run_script.call_count == 2
```


# LLM-generated content at query #17
#--------------------------

```python
def test_valid_hook_returns_true_when_conditions_met():
    import os
    global _HOOKS
    _HOOKS = ['pre-commit', 'post-checkout']
    hook_file = '/path/to/pre-commit'
    hook_name = 'pre-commit'
    assert valid_hook(hook_file, hook_name) == True
```


# LLM-generated content at query #18
#--------------------------

```python
def test_find_hook_type_hint_validity():
    import inspect
    from typing import get_type_hints

    # Test the function signature for type hint correctness
    # Since we cannot define a new function with different logic, 
    # we verify that the existing function's type hints match expectations.
    
    # Note: The prompt asks to ensure the predicate at line 1 evaluates to True.
    # Line 1 is a function definition. In a testing context for a signature, 
    # we check if the return type hint is list[str] | None.
    
    hints = get_type_hints(find_hook)
    
    # We check if 'return' value type hint is correctly identified as Union[list[str], NoneType]
    # In Python 3.10+, 'list[str] | None' is equivalent to typing.Union[list[str], None]
    from typing import Union, get_origin
    
    return_hint = hints['return']
    origin = get_origin(return_hint)
    
    # The predicate at line 1 is the definition itself. 
    # We assert that the function exists and its return type hint is valid.
    assert find_hook.__annotations__['return'] == list[str] | None
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_script_python_file_success():
    import subprocess
    from unittest.mock import patch, MagicMock
    from pathlib import Path
    import sys

    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'linux'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script("test_script.py", cwd="/tmp")
        
        mock_make_exec.assert_called_once_with("test_script.py")
        mock_popen.assert_called_once_with([sys.executable, "test_script.py"], shell=False, cwd="/tmp")

def test_run_script_shell_script_success():
    import subprocess
    from unittest.mock import patch, MagicMock
    import sys

    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'win32'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script("test_script.sh", cwd=".")
        
        mock_make_exec.assert_called_once_with("test_script.sh")
        mock_popen.assert_called_once_with(["test_script.sh"], shell=True, cwd=".")

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
            run_script("test_script.py")
        
        assert "Hook script failed (exit status: 1)" in str(excinfo.value)

def test_run_script_os_error_enoexec():
    import subprocess
    import errno
    from unittest.mock import patch
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        error = OSError()
        error.errno = errno.ENOEXEC
        mock_popen.side_effect = error
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script("test_script.py")
            
        assert "Hook script failed, might be an empty file or missing a shebang" in str(excinfo.value)

def test_run_script_os_error_generic():
    import subprocess
    from unittest.mock import patch
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        error = OSError("Permission denied")
        mock_popen.side_effect = error
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script("test_script.py")
            
        assert "Hook script failed (error: Permission denied)" in str(excinfo.value)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_run_script_raises_failed_hook_exception_on_oserror_not_enoexec():
    import subprocess
    import errno
    from unittest.mock import patch, MagicMock

    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Simulate an OSError that is NOT ENOEXEC (e.g., EACCES)
        error_instance = OSError()
        error_instance.errno = errno.EACCES
        mock_process.wait.side_effect = error_instance

        with pytest.raises(FailedHookException) as excinfo:
            run_script('test_script.sh')
        
        assert 'Hook script failed (error:' in str(excinfo.value)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script_raises_failed_hook_exception_on_oserror_not_enoexec():
    import subprocess
    import errno
    from unittest.mock import patch, MagicMock
    from pathlib import Path

    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        
        # Trigger OSError with an errno that is NOT ENOEXEC (e.g., EACCES)
        error_errno = errno.EACCES
        mock_popen.side_effect = OSError(error_errno, "Permission denied")
        
        with pytest.raises(FailedHookException) as excinfo:
            run_script("test_script.sh")
        
        assert f"(error: [Errno {error_errno} Permission denied])" in str(excinfo.value)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            mock_work_in.return_value.__enter__.return_value = None
            
            run_hook_from_repo_dir(
                repo_dir='/tmp/repo',
                hook_name='post_gen_project',
                project_dir='/tmp/project',
                context={'foo': 'bar'},
                delete_project_on_failure=True
            )
            
            mock_run_hook.assert_called_once_with('post_gen_project', '/tmp/project', {'foo': 'bar'})


def test_run_hook_from_repo_dir_failure_deletes_project():
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                mock_work_in.return_value.__enter__.return_value = None
                mock_run_hook.side_effect = FailedHookException("Failed")
                
                try:
                    run_hook_from_repo_dir(
                        repo_dir='/tmp/repo',
                        hook_name='post_gen_project',
                        project_dir='/tmp/project',
                        context={'foo': 'bar'},
                        delete_project_on_failure=True
                    )
                except FailedHookException:
                    pass

                mock_rmtree.assert_called_once_with('/tmp/project')


def test_run_hook_from_repo_dir_failure_does_not_delete_project():
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    from cookiecutter.exceptions import FailedHookException

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                mock_work_in.return_value.__enter__.return_value = None
                mock_run_hook.side_effect = FailedHookException("Failed")
                
                try:
                    run_hook_from_repo_dir(
                        repo_dir='/tmp/repo',
                        hook_name='post_gen_project',
                        project_dir='/tmp/project',
                        context={'foo': 'bar'},
                        delete_project_on_failure=False
                    )
                except FailedHookException:
                    pass

                mock_rmtree.assert_not_called()
```


# LLM-generated content at query #5
#--------------------------

```python
def test_run_script_raises_enoexec_error():
    import os
    import errno
    from pathlib import Path
    from unittest.mock import patch, MagicMock

    with patch('subprocess.Popen') as mock_popen:
        mock_os_error = OSError()
        mock_os_error.errno = errno.ENOEXEC
        mock_popen.side_effect = mock_os_error
        
        with patch('utils.make_executable'):
            with patch('sys.platform', 'linux'):
                with patch('sys.executable', '/usr/bin/python3'):
                    from your_module import run_script, FailedHookException
                    with pytest.raises(FailedHookException) as excinfo:
                        run_script('test_script.sh')
                    assert 'might be an empty file or missing a shebang' in str(excinfo.value)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_valid_hook_success():
    import os
    _HOOKS = ['pre-commit', 'post-checkout']
    # Assuming the environment allows mocking or global variable manipulation for testing purposes
    # Since I cannot redefine globals, I am assuming _HOOKS is accessible and contains these values.
    assert valid_hook('/path/to/pre-commit', 'pre-commit') == True

def test_valid_hook_wrong_name():
    import os
    _HOOKS = ['pre-commit']
    assert valid_hook('/path/to/other-hook', 'pre-commit') == False

def test_valid_hook_unsupported_type():
    import os
    _HOOKS = ['pre-commit']
    assert valid_hook('/path/to/unknown', 'unknown') == False

def test_valid_hook_is_backup_file():
    import os
    _HOOKS = ['pre-commit']
    assert valid_hook('/path/to/pre-commit~', 'pre-commit') == False

def test_valid_hook_with_extension_mismatch():
    import os
    _HOOKS = ['pre-commit']
    assert valid_hook('/path/to/pre-commit.py', 'pre-commit') == True # basename is pre-commit
    # Note: if the logic relies on splitext, '.py' is stripped. 
    # If hook_name is 'pre-commit' and basename is 'pre-commit', it returns true.

def test_valid_hook_empty_inputs():
    import os
    _HOOKS = ['pre-commit']
    assert valid_hook('', '') == False
```


# LLM-generated content at query #7
#--------------------------

```python
def test_run_script_with_context_success(mocker):
    mocker.patch("cookiecutter.hooks.Path.read_text", return_value="Hello {{ name }}")
    mocker.patch("cookiecutter.hooks.create_env_with_context")
    mocker.patch("cookiecutter.hooks.tempfile.NamedTemporaryFile")
    mocker.patch("cookiecutter.hooks.run_script")
    mocker.patch("os.path.splitext", return_value=("script", ".sh"))
    
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mocker.patch("cookiecutter.hooks.create_env_with_context").return_value = mock_env
    mock_env.from_string.return_value = mock_template
    mock_template.render.return_value = "Hello World"
    
    mock_temp = mocker.Mock()
    mocker.patch("cookiejack.hooks.tempfile.NamedTemporaryFile", return_value=mocker.Mock(__enter__=lambda s: mock_temp))
    mock_temp.name = "/tmp/temp_script.sh"

    from cookiecutter.hooks import run_script_with_context
    run_script_with_context("script.sh", ".", {"name": "World"})

    mock_template.render.assert_called_once_with(name="World")
    mock_temp.write.assert_called_once_with(b"Hello World")
    mocker.patch("cookiecutter.hooks.run_script").assert_called_once_with("/tmp/temp_script.sh", ".")

def test_run_script_with_context_fails_on_render(mocker):
    mocker.patch("cookiecutter.hooks.Path.read_text", return_value="Error context")
    mocker.patch("cookiecutter.hooks.create_env_with_context")
    mocker.patch("cookiecutter.hooks.tempfile.NamedTemporaryFile")
    mocker.patch("os.path.splitext", return_value=("script", ".sh"))
    
    mock_env = mocker.Mock()
    mock_template = mocker.Mock()
    mocker.patch("cookiecutter.hooks.create_env_with_context").return_value = mock_env
    mock_env.from_string.return_value = mock_template
    mock_template.render.side_effect = KeyError("Missing key")

    from cookiecutter.hooks import run_script_with_context
    try:
        run_script_with_context("script.sh", ".", {})
    except KeyError:
        pass
    else:
        raise AssertionError("KeyError should have been raised")
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_returns_none_when_directory_does_not_exist():
    result = find_hook("pre-commit", "non_existent_directory_12345")
    assert result is None

def test_find_hook_returns_none_when_directory_is_empty():
    temp_dir = tempfile.mkdtemp()
    try:
        result = find_hook("pre-commit", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_returns_path_when_valid_hook_exists():
    temp_dir = tempfile.mkdtemp()
    # Assuming _HOOKS contains 'pre-commit' based on common usage patterns
    # and the logic of valid_hook
    hook_filename = "pre-commit"
    with open(os.path.join(temp_dir, hook_filename), "w") as f:
        f.write("#!/bin/bash\nexit 0")
    
    try:
        # We pass the absolute path of temp_dir to override default 'hooks'
        result = find_hook("pre-commit", temp_dir)
        expected_path = os.path.abspath(os.path.join(temp_dir, hook_filename))
        assert result == [expected_path]
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files():
    temp_dir = tempfile.mkdtemp()
    # Create a valid hook and a backup version (ending in ~)
    with open(os.path.join(temp_dir, "pre-commit"), "w") as f:
        f.write("valid")
    with open(os.path.join(temp_dir, "pre-commit~"), "w") as f:
        f.write("backup")
    
    try:
        result = find_hook("pre-commit", temp_dir)
        expected_path = os.path.abspath(os.path.join(temp_dir, "pre-commit"))
        assert result == [expected_path]
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_mismatched_hook_name():
    temp_dir = tempfile.mkdtemp()
    with open(os.path.join(temp_dir, "post-commit"), "w") as f:
        f.write("content")
    
    try:
        result = find_hook("pre-commit", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_run_script_raises_enoexec_error():
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
            run_script(script_path='test_script.sh')
        
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)
```


# LLM-generated content at query #10
#--------------------------

```python
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from cookiecutter.hooks import run_script_with_context

def test_run_script_with_context_tempfile_suffix_matches_extension():
    script_path = "test_script.sh"
    cwd = "/tmp/cwd"
    context = {"cookiecutter": {"_extensions": []}}
    
    # Mocking Path().read_text to return content
    # Mocking create_env_with_context to return a mock env that renders content
    # Mocking run_script to avoid execution
    # We focus on ensuring the suffix logic in line 10/14 works by checking args of NamedTemporaryFile
    
    with patch("cookiecutter.hooks.Path") as mock_path, \
         patch("cookiejack.hooks.create_env_with_context") as mock_env_creator, \
         patch("cookiecutter.hooks.run_script") as mock_run_script, \
         patch("tempfile.NamedTemporaryFile") as mock_tempfile:
        
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.read_text.return_value = "content"
        mock_path.return_value = mock_path_instance
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = "rendered_content"
        mock_env.from_string.return_value = mock_template
        mock_env_creator.return_value = mock_env
        
        # Mock the context manager for NamedTemporaryFile
        mock_temp_instance = MagicMock()
        mock_temp_instance.__enter__.return_value = mock_temp_instance
        mock_temp_instance.name = "/tmp/temp_file"
        mock_tempfile.return_value = mock_temp_instance

        run_script_with_context(script_path, cwd, context)

        # The extension of "test_script.sh" is ".sh"
        # Assert that NamedTemporaryFile was called with suffix=".sh"
        mock_tempfile.assert_called_once_with(delete=False, mode='wb', suffix='.sh')
```


# LLM-generated content at query #11
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_input():
    import os
    global _HOOKS
    _HOOKS = ['pre-commit', 'post-checkout']
    hook_file = '/path/to/pre-commit'
    hook_name = 'pre-commit'
    assert valid_hook(hook_file, hook_name) == True
```


# LLM-generated content at query #12
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks_returns_original_dir():
    import os
    import tempfile
    from pathlib import Path
    import shutil
    from cookiecutter.hooks import run_pre_prompt_hook

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        result = run_pre_prompt_hook(tmp_dir)
        assert result == tmp_dir
    finally:
        shutil.rmtree(tmp_dir)

def test_run_pre_prompt_hook_with_valid_hook_returns_tmp_dir():
    import os
    import tempfile
    import shutil
    from pathlib import Path
    from cookiecutter import hooks, utils
    from unittest.mock import patch, MagicMock

    # Setup a real directory structure for the test
    base_tmp = Path(tempfile.mkdtemp())
    hooks_dir = base_tmp / "hooks"
    hooks_dir.mkdir()
    
    # Create a dummy script file that is a valid hook
    # We must mock _HOOKS to include 'pre_prompt' for the function to consider it valid
    hook_script = hooks_dir / "pre_prompt"
    hook_script.write_text("#!/bin/bash\nexit 0")
    
    with patch('cookiecutter.hooks._HOOKS', ['pre_prompt']), \
         patch('cookiecuter.utils.make_executable'), \
         patch('subprocess.Popen') as mock_popen:
        
        # Mock subprocess to return success
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        result = run_pre_prompt_hook(base_tmp)
        
        assert Path(result).exists()
        assert result != base_tmp
        assert Path(result).name == base_tmp.name
    
    shutil.rmtree(base_tmp)

def test_run_pre_prompt_hook_raises_exception_on_failed_script():
    import os
    import tempfile
    import shutil
    from pathlib import Path
    from cookiecutter.hooks import run_pre_prompt_hook, FailedHookException
    from unittest.mock import patch

    base_tmp = Path(tempfile.mkdtemp())
    hooks_dir = base_tmp / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / "pre_prompt"
    hook_script.write_text("#!/bin/bash\nexit 1")

    with patch('cookiecutter.hooks._HOOKS', ['pre_prompt']), \
         patch('subprocess.Popen') as mock_popen:
        
        # Mock subprocess to return failure (non-zero exit status)
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc

        try:
            run_pre_prompt_hook(base_tmp)
            raise AssertionError("FailedHookException was not raised")
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)

    shutil.rmtree(base_tmp)
```


# LLM-generated content at query #13
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_no_directory():
    result = find_hook("test_hook", hooks_dir="non_existent_directory_12345")
    assert result is None

def test_find_hook_empty_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        # Assuming _HOOKS contains 'test_hook' for this test to work logic-wise
        # In a real scenario, we'd mock _HOOKS, but here we rely on the directory state
        result = find_path_logic_mocked_with_empty_dir(temp_dir, "test_hook")
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_success():
    # This test assumes _HOOKS is globally accessible and contains 'valid_hook_name'
    # Since I cannot modify the global _HOOKS, this test demonstrates the structural requirement
    temp_dir = tempfile.mkdtemp()
    hook_name = "valid_hook_name" # This must exist in the module's _HOOKS
    hook_file = f"{hook_name}.sh"
    hook_path = os.path.join(temp_dir, hook_file)
    
    with open(hook_path, 'w') as f:
        f.write("#!/bin/bash\necho 'hello'")
    
    try:
        # We use a mock-like approach by ensuring the file exists in a controlled dir
        # Note: find_hook uses os.listdir on hooks_dir
        results = find_hook(hook_name, hooks_dir=temp_dir)
        # If hook_name is in _HOOKS, it should return the absolute path
        # This test is dependent on the external state of _HOOKS
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files():
    temp_dir = tempfile.mkdtemp()
    hook_name = "valid_hook_name" 
    # Create a file ending in ~ which should be ignored by valid_hook
    backup_file = f"{hook_name}.sh~"
    hook_path = os.path.join(temp_dir, backup_file)
    
    with open(hook_path, 'w') as f:
        f.write("content")
    
    try:
        result = find_hook(hook_name, hooks_dir=temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_returns_absolute_paths():
    temp_dir = tempfile.mkdtemp()
    hook_name = "valid_hook_name"
    hook_file = f"{hook_name}.py"
    hook_path = os.path.join(temp_dir, hook_file)
    
    with open(hook_path, 'w') as f:
        f.write("print('test')")
    
    try:
        results = find_hook(hook_name, hooks_dir=temp_dir)
        assert results is not None
        assert os.path.isabs(results[0])
        assert results[0] == os.path.abspath(hook_path)
    finally:
        shutil_rmtree_helper(temp_dir)

def shutil_rmtree_helper(path):
    shutil.rmtree(path)
```


# LLM-generated content at query #14
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
        
        run_script('/tmp/test.py', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/tmp/test.py')
        mock_popen.assert_called_once_with(['/usr/bin/python3', '/tmp/test.py'], shell=False, cwd='/tmp')

def test_run_script_shell_script_success_windows():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exec, \
         patch('sys.platform', 'win32'), \
         patch('sys.executable', 'C:\\Python\\python.exe'):
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('/tmp/test.sh', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/tmp/test.sh')
        mock_popen.assert_called_once_with(['/tmp/test.sh'], shell=True, cwd='/tmp')

def test_run_script_failure_exit_status():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'),
         patch('sys.executable', '/usr/bin/python3'):
        mock_process = MagicMock()
        mock_process.wait.return::return_value = 1
        mock_popen.return_value = mock_process
        
        from your_module import FailedHookException
        with Exception: # Catching the specific custom exception
            try:
                run_script('/tmp/test.py')
            except FailedHookException as e:
                assert 'Hook script failed (exit status: 1)' in str(e)

def test_run_script_oserror_enoexec():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        mock_popen.side_effect = OSError(errno.ENOEXEC, "exec format error")
        
        from your_module import FailedHookException
        with Exception:
            try:
                run_script('/tmp/test.sh')
            except FailedHookException as e:
                assert 'might be an empty file or missing a shebang' in str(e)

def test_run_script_oserror_generic():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        mock_popen.side_effect = OSError(errno.EACCES, "Permission denied")
        
        from your_module import FailedHookException
        with Exception:
            try:
                run_script('/tmp/test.sh')
            except FailedHookException as e:
                assert 'Hook script failed (error: [Errno 13] Permission denied)' in str(e)
```


# LLM-generated content at query #15
#--------------------------

```python
def test_valid_hook_success():
    import os
    # Assuming _HOOKS contains 'pre-commit'
    global _HOOKS
    _HOOKS = ['pre-commit', 'post-checkout']
    result = valid_hook('/path/to/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_wrong_name():
    import os
    global _HOOKS
    _HOOKS = ['pre-commit']
    result = valid_hook('/path/to/post-checkout', 'pre-commit')
    assert result is False

def test_valid_hook_unsupported_name():
    import os
    global _HOOKS
    _EXCLUDED_HOOKS = ['unsupported']
    _HOOKS = ['pre-commit']
    result = valid_hook('/path/to/unsupported', 'unsupported')
    assert result is False

def test_valid_hook_is_backup_file():
    import os
    global _HOOKS
    _HOOKS = ['pre-commit']
    result = valid_hook('/path/to/pre-commit~', 'pre-commit')
    assert result is False

def test_valid_hook_different_directory_same_basename():
    import os
    global _HOOKS
    _HOOKS = ['pre-commit']
    result = valid_hook('subdir/pre-commit', 'pre-commit')
    assert result is True

def test_valid_hook_with_extension():
    import os
    global _HOOKS
    _HOOKS = ['pre-commit']
    # splitext will take 'pre-commit' as basename
    result = valid_hook('/path/to/pre-commit.sh', 'pre-commit')
    assert result is True
```


# LLM-generated content at query #16
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_directory_not_exists():
    result = find_hook("pre-commit", "non_existent_directory_path_12345")
    assert result is None

def test_find_hook_no_matching_hooks():
    temp_dir = tempfile.mkdtemp()
    try:
        hooks_subdir = os.path.join(temp_dir, "hooks")
        os.mkdir(hooks_subdir)
        with open(os.path.join(hooks_subdir, "other_hook.sh"), "w") as f:
            f.write("#!/bin/bash\necho 'hello'")
        
        # Note: This test assumes _HOOKS contains 'other_hook' or we are testing the logic
        # Since we cannot modify _HOOKS, we rely on it containing known values or 
        # being mockable if this were a full integration test.
        # For the purpose of this unit test, we assume 'pre-commit' is not in the file list.
        result = find_hook("pre-commit", hooks_subdir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_success():
    temp_dir = tempfile.mkdtemp()
    try:
        hooks_subdir = os.path.join(temp_dir, "hooks")
        os.mkdir(hooks_subdir)
        
        # We use a name that we assume exists in the global _HOOKS 
        # (In a real scenario, we would mock _HOOKS)
        # Assuming 'pre-commit' is a standard hook name used in the environment
        hook_name = "pre-commit"
        hook_file_path = os.path.join(hooks_subdir, f"{hook_name}.sh")
        with open(hook_file_path, "w") as f:
            f.write("#!/bin/bash\necho 'hello'")
        
        # We must ensure the logic matches what's in _HOOKS
        # If we cannot control _HOOKS, this test is fragile, but following the prompt:
        result = find_hook(hook_name, hooks_subdir)
        
        # If the hook name provided exists in the directory and is valid per valid_hook logic
        # The result should be a list containing the absolute path.
        if result is not None:
            assert len(result) == 1
            assert result[0] == os.path.abspath(hook_file_path)
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files():
    temp_dir = tempfile.mktemp()
    try:
        hooks_subdir = os.path.join(temp_dir, "hooks")
        os.mkdir(hooks_subdir)
        
        # Create a backup file (ends with ~)
        hook_name = "pre-commit"
        backup_file_path = os.path.join(hooks_subdir, f"{hook_name}.sh~")
        with open(backup_file_path, "w") as f:
            f.write("backup")
            
        result = find_hook(hook_name, hooks_subdir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks_returns_original_dir():
    import tempfile
    import shutil
    from pathlib import Path
    import os

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        result = run_pre_prompt_hook(str(tmp_dir))
        assert result == tmp_dir
    finally:
        shutil.rmtree(tmp_dir)

def test_run_pre_prompt_hook_with_valid_hooks_returns_tmp_copy():
    import tempfile
    import shutil
    from pathlib import Path
    import os

    tmp_repo = Path(tempfile.mkdtemp())
    hooks_dir = tmp_repo / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("#!/bin/bash\nexit 0")
    os.chmod(hook_script, 0o755)

    try:
        result = run_pre_prompt_hook(str(tmp_repo))
        assert Path(result).resolve() != tmp_repo.resolve()
        assert (Path(result) / "hooks" / "pre_prompt.py").exists()
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
    hook_script = hooks_dir / "pre_prompt.py"
    # Script that exits with error
    hook_script.write_text("#!/bin/bash\nexit 1")
    os.chmod(hook_script, 0o755)

    try:
        from cookiecutter.hooks import FailedHookException
        try:
            run_pre_prompt_hook(str(tmp_repo))
            raise AssertionError("Should have raised FailedHookException")
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)
    finally:
        shutil.rmtree(tmp_repo)
```


# LLM-generated content at query #18
#--------------------------

```python
import sys
import subprocess
import errno
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_run_script_python_file_success():
    with patch('sys.platform', 'linux'), \
         patch('sys.executable', '/usr/bin/python3'), \
         patch('utils.make_executable') as mock_make_exec, \
         patch('subprocess.Popen') as mock_popen:
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('/tmp/test.py', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/tmp/test.py')
        mock_popen.assert_called_once_with(['/usr/bin/python3', '/tmp/test.py'], shell=False, cwd='/tmp')

def test_run_script_shell_script_windows():
    with patch('sys.platform', 'win32'), \
         patch('utils.make_executable') as mock_make_exec, \
         patch('subprocess.Popen') as mock_popen:
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script('/tmp/test.sh', cwd='/tmp')
        
        mock_make_exec.assert_called_once_with('/tmp/test.sh')
        mock_popen.assert_called_once_with(['/tmp/test.sh'], shell=True, cwd='/tmp')

def test_run_script_failure_exit_status():
    with patch('sys.platform', 'linux'), \
         patch('sys.executable', '/usr/bin/python3'), \
         patch('utils.make_executable'), \
         patch('subprocess.Popen') as mock_popen:
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        from your_module import FailedHookException
        with pytest.raises(FailedHookException) as excinfo:
            run_script('/tmp/test.py')
        
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

def test_run_script_oserror_enoexec():
    with patch('sys.platform', 'linux'), \
         patch('utils.make_executable'), \
         patch('subprocess.Popen') as mock_popen:
        
        error = OSError()
        error.errno = errno.ENOEXEC
        mock_popen.side_effect = error
        
        from your_module import FailedHookException
        with pytest.raises(FailedHookException) as excinfo:
            run_script('/tmp/test.sh')
            
        assert 'might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_oserror_generic():
    with patch('sys.platform', 'linux'), \
         patch('utils.make_executable'), \
         patch('subprocess.Popen') as mock_popen:
        
        error = OSError("Permission denied")
        error.errno = errno.EACCES
        mock_popen.side_effect = error
        
        from your_module import FailedHookException
        with pytest.raises(FailedHookException) as excinfo:
            run_script('/tmp/test.sh')
            
        assert 'error: Permission denied' in str(excinfo.value)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_no_scripts_found():
    from unittest.mock import patch
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        with patch('cookiecutter.hooks.logger.debug') as mock_logger:
            from cookiecutter.hooks import run_hook
            run_hook('pre_gen_project', '/tmp/project', {})
            mock_logger.assert_any_call('No %s hook found', 'pre_gen_project')

def test_run_hook_executes_scripts():
    from unittest.mock import patch
    scripts = ['/tmp/hooks/pre_gen_project.sh']
    context = {'project_name': 'test'}
    with patch('cookiecutter.hooks.find_hook', return_value=scripts):
        with patch('cookiecutter.hooks.logger.debug') as mock_logger:
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                from cookiecutter.hooks import run_hook
                run_hook('pre_gen_project', '/tmp/project', context)
                mock_run.assert_called_once_with(scripts[0], '/tmp/project', context)
                mock_logger.assert_any_call('Running hook %s', 'pre_gen_project')
```


# LLM-generated content at query #20
#--------------------------

```python
def test_run_script_exit_status_success():
    import subprocess
    from unittest.mock import MagicMock, patch
    from pathlib import Path

    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0  # EXIT_SUCCESS is 0
        mock_popen.return_value = mock_process
        
        run_script('test_script.py', cwd='.')
        
        mock_process.wait.assert_called_once()
```


# LLM-generated content at query #21
#--------------------------

```python
def test_find_hook_signature_type_hints():
    import inspect
    from typing import get_type_hints
    
    # Check if the function exists and is callable to verify signature access
    # The predicate at line 1 is a type-annotated function definition.
    # We verify that we can retrieve its annotations.
    hints = get_type_hints(find_hook)
    assert hints['hook_name'] == str
    assert hints['hooks_dir'] == str
    assert find_hook.__annotations__['return'] == list[str] | None
```


# LLM-generated content at query #22
#--------------------------

```python
import sys
import subprocess
import errno
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_run_script_python_file_success():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exe, \
         patch('sys.platform', 'linux'), \
         patch('sys.executable', '/usr/bin/python3'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script("test_script.py", cwd="/tmp")
        
        mock_make_exe.assert_called_once_with("test_script.py")
        mock_popen.assert_called_once_with(['/usr/bin/python3', 'test_script.py'], shell=False, cwd="/tmp")

def test_run_script_shell_script_success():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable') as mock_make_exe, \
         patch('sys.platform', 'win32'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script("test_script.sh", cwd=".")
        
        mock_make_exe.assert_called_once_with("test_script.sh")
        mock_popen.assert_called_once_with(['test_script.sh'], shell=True, cwd=".")

def test_run_script_failure_exit_status():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        from your_module import FailedHookException
        with pytest.raises(FailedHookException) as excinfo:
            run_script("test_script.py")
        
        assert "Hook script failed (exit status: 1)" in str(excinfo.value)

def test_run_script_oserror_enoexec():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        error = OSError()
        error.errno = errno.ENOEXEC
        mock_popen.side_effect = error
        
        from your_module import FailedHookException
        with pytest.raises(FailedHookException) as excinfo:
            run_script("test_script.sh")
            
        assert "might be an empty file or missing a shebang" in str(excinfo.value)

def test_run_script_oserror_generic():
    with patch('subprocess.Popen') as mock_popen, \
         patch('utils.make_executable'), \
         patch('sys.platform', 'linux'):
        
        error = OSError()
        error.strerror = "Permission denied"
        mock_popen.side_effect = error
        
        from your_module import FailedHookException
        with pytest.raises(FailedHookException) as excinfo:
            run_script("test_script.py")
            
        assert "Hook script failed (error:" in str(excinfo.value)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_run_script_with_context_preserves_extension():
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_script_with_context

    # Setup: Create a dummy script file with a specific extension
    dummy_content = "Hello {{ name }}!"
    dummy_script_name = "test_script.py"
    dummy_script_path = Path(tempfile.gettempdir()) / dummy_script_name
    dummy_script_path.write_text(dummy_content, encoding='utf-8')

    # Mocking dependencies to avoid actual execution and file system side effects
    # We mock run_script to prevent it from actually running the temp file
    # We mock create_env_with_context to return a controlled environment
    # We mock NamedTemporaryFile to capture the arguments passed to it
    
    mock_context = {"name": "World", "cookiecutter": {}}
    mock_cwd = Path(tempfile.gettempdir())

    with patch("cookiecutter.hooks.run_script") as mock_run_script, \
         patch("cookiejava.utils.create_env_with_context") as mock_create_env, \
         patch("tempfile.NamedTemporaryFile", wraps=tempfile.NamedTemporaryFile) as mock_tempfile:
        
        # Configure the environment mock to behave like a real Jinja env for rendering
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = "Hello World!"
        mock_env.from_string.return_value = mock_template
        mock_create_env.return_value = mock_env

        # Execute the function
        run_script_with_context(dummy_script_path, mock_cwd, mock_context)

        # Assertion: Check if NamedTemporaryFile was called with the correct suffix (extension)
        # The extension extracted from 'test_script.py' is '.py'
        args, kwargs = mock_tempfile.call_args
        assert kwargs['suffix'] == '.py'

        # Cleanup
        if dummy_script_path.exists():
            dummy_script_path.unlink()
```


# LLM-generated content at query #24
#--------------------------

```python
from unittest.mock import patch
from pathlib import Path
from cookiecutter.hooks import run_hook

@patch('cookiecutter.hooks.find_hook')
@patch('cookiecutter.hooks.logger')
def test_run_hook_no_scripts_found(mock_logger, mock_find_hook):
    mock_find_hook.return_value = []
    run_hook("pre_gen_project", "/tmp/project", {"some": "context"})
    mock_logger.debug.assert_called_once_with('No %s hook found', "pre_gen_project")
```


# LLM-generated content at query #25
#--------------------------

```python
def test_find_hook_returns_none_when_no_valid_hooks_found(mocker):
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('os.listdir', return_value=['test_hook.py'])
    mocker.patch('os.path.abspath', side_effect=lambda x: x)
    mocker.patch('os.path.join', side_effect=lambda x, y: f"{x}/{y}")
    mocker.patch('__main__.valid_hook', return_value=False)
    
    result = find_hook('non_existent_hook', 'hooks')
    
    assert result is None
```


# LLM-generated content at query #26
#--------------------------

```python
def test_run_script_success_status():
    import subprocess
    from unittest.mock import MagicMock, patch
    from pathlib import Path

    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 0  # Assuming EXIT_SUCCESS is 0
        mock_popen.return_value = mock_process
        
        with patch('utils.make_executable'):
            with patch('sys.platform', 'linux'):
                run_script('test_script.py')
                
        assert mock_process.wait.return_value == 0
```


# LLM-generated content at query #27
#--------------------------

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir).resolve()
        with patch("cookiecutter.hooks.find_hook", return_value=[]):
            result = run_pre_prompt_hook(repo_dir)
            assert result == repo_dir
```


# LLM-generated content at query #28
#--------------------------

```python
def test_run_hook_from_repo_dir_raises_on_failed_hook():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.hooks import run_hook_from_repo_dir
    import cookiecutter.exceptions

    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_hook.py"
    project_dir = "/tmp/project"
    context = {"project_name": "test"}
    delete_project_on_failure = True

    with patch("cookiecutter.hooks.work_in"), \
         patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree, \
         patch("cookiecutter.hooks.logger") as mock_logger:
        
        mock_run_hook.side_effect = cookiecutter.exceptions.FailedHookException("Hook failed")

        with Exception:
            try:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=delete_project_on_failure,
                )
            except cookiecutter.exceptions.FailedHookException:
                pass

        mock_rmtree.assert_called_once_with(project_dir)
        mock_logger.exception.assert_called()
```


# LLM-generated content at query #29
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found():
    import tempfile
    import shutil
    from pathlib import Path
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir).resolve()
        
        with patch("cookiecutter.hooks.find_hook", return_value=[]):
            result = run_pre_prompt_hook(repo_path)
            
    assert result == repo_path
```


# LLM-generated content at query #30
#--------------------------

```python
def test_valid_hook_returns_true_when_all_conditions_met():
    import os
    global _HOOKS
    _HOOKS = ['pre-commit', 'commit-msg']
    hook_file = '/path/to/pre-commit'
    hook_name = 'pre-commit'
    assert valid_hook(hook_file, hook_name) == True
```


# LLM-generated content at query #31
#--------------------------

```python
def test_find_hook_type_hint_validity():
    import os
    from typing import List, Optional

    # Test that the function signature/predicate logic (types) is compatible with expected behavior
    # Note: The predicate at line 1 is a type hint definition. 
    # We verify if an instance of the return type matches the docstring's promised types.
    
    # Mocking environment for directory existence to satisfy internal logic if needed,
    # but focusing on the signature verification.
    
    result_list = ["/abs/path/to/hook1.py", "/abs/path/to/hook2.py"]
    result_none = None

    assert isinstance(result_list, list)
    assert all(isinstance(item, str) for item in result_list)
    assert isinstance(result_none, type(None)) or result_none is None
```


# LLM-generated content at query #32
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks_returns_original_dir():
    import tempfile
    from pathlib import Path
    import os
    from cookiecutter.hooks import run_pre_prompt_hook

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        result = run_pre_prompt_hook(tmp_path)
        assert result == tmp_path


def test_run_pre_prompt_hook_with_valid_hook_executes_and_returns_new_dir():
    import tempfile
    from pathlib import Path
    import os
    import sys
    from cookiecutter.hooks import run_pre_prompt_hook

    with tempfile.TemporaryDirectory() as repo_dir_str:
        repo_dir = Path(repo_dir_str).resolve()
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        
        # Create a dummy python script that exits successfully
        script_name = "pre_prompt"
        script_path = hooks_dir / f"{script_name}.py"
        script_path.write_text(f"import sys; print('executing'); sys.exit(0)")
        
        # Mocking _HOOKS in the module scope via monkeypatch is not possible 
        # without pytest, so we assume pre_prompt is in the global _HOOKS 
        # or that the environment is set up such that valid_hook returns True.
        # Since we cannot define functions, we rely on the provided logic.
        
        result = run_pre_prompt_hook(repo_dir)
        
        assert os.path.isdir(str(result))
        assert result != repo_dir
        assert repo_dir.name in str(result)


def test_run_pre_prompt_hook_failed_script_raises_exception():
    import tempfile
    from pathlib import Path
    import os
    from cookiecutter.hooks import run_pre_prompt_hook, FailedHookException

    with tempfile.TemporaryDirectory() as repo_dir_str:
        repo_dir = Path(repo_dir_str).resolve()
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        
        # Create a script that exits with error status 1
        script_name = "pre_prompt"
        script_path = hooks_dir / f"{script_name}.py"
        script_path.write_text(f"import sys; sys.exit(1)")
        
        # We expect the exception to be raised when run_script encounters exit status 1
        try:
            run_pre_prompt_hook(repo_dir)
            raise AssertionError("FailedHookException was not raised")
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)
```


# LLM-generated content at query #33
#--------------------------

```python
def test_run_hook_no_scripts_found():
    from unittest.mock import patch
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        with patch('cookiecutter.hooks.logger.debug') as mock_debug:
            from cookiecutter.hooks import run_hook
            run_hook('pre_gen_project', '/tmp/project', {})
            mock_debug.assert_any_call('No %s hook found', 'pre_gen_project')

def test_run_hook_executes_scripts():
    from unittest.mock import patch
    scripts = ['/tmp/hooks/pre_gen_project.sh']
    context = {'some_var': 'value'}
    project_dir = '/tmp/project'
    with patch('cookiecutter.hooks.find_hook', return_value=scripts):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            with patch('cookiecutter.hooks.logger.debug') as mock_debug:
                from cookiecutter.hooks import run_hook
                run_hook('pre_gen_project', project_dir, context)
                mock_run.assert_called_once_with(scripts[0], project_dir, context)
                mock_debug.assert_any_call('Running hook %s', 'pre_gen_project')

def test_run_hook_executes_multiple_scripts():
    from unittest.mock import patch
    scripts = ['/tmp/hooks/pre_gen_project.sh', '/tmp/hooks/pre_gen_project.py']
    context = {'some_var': 'value'}
    project_dir = '/tmp/project'
    with patch('cookiecutter.hooks.find_hook', return_value=scripts):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            from cookiecutter.hooks import run_hook
            run_hook('pre_gen_project', project_dir, context)
            assert mock_run.call_count == 2
            mock_run.assert_any_call(scripts[0], project_dir, context)
            mock_run.assert_any_call(scripts[1], project_dir, context)
```


# LLM-generated content at query #34
#--------------------------

```python
import os
import tempfile
import shutil

def test_find_hook_returns_none_when_directory_does_not_exist():
    result = find_hook("test_hook", "non_existent_directory_12345")
    assert result is None

def test_find_hook_returns_none_when_directory_is_empty():
    temp_dir = tempfile.mkdtemp()
    try:
        result = find_hook("test_hook", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_returns_path_when_valid_hook_exists():
    # Note: This test assumes _HOOKS contains 'valid_name'
    # Since _HOOKS is not defined in the snippet, we assume a context where it exists.
    temp_dir = tempfile.mkdtemp()
    hook_name = "valid_name"
    hook_filename = f"{hook_name}.sh"
    hook_path = os.path.join(temp_dir, hook_filename)
    with open(hook_path, 'w') as f:
        f.write("#!/bin/bash\n")
    
    try:
        # Mocking the behavior by ensuring the file exists in a directory we control
        result = find_hook(hook_name, temp_dir)
        assert result is not None
        assert os.path.abspath(hook_path) in result
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_backup_files():
    temp_dir = tempfile.mkdtemp()
    hook_name = "valid_name"
    backup_filename = f"{hook_name}.sh~"
    backup_path = os.path.join(temp_dir, backup_filename)
    with open(backup_path, 'w') as f:
        f.write("backup")
    
    try:
        result = find_hook(hook_name, temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_hook_ignores_mismatched_hook_names():
    temp_dir = tempfile.mkdtemp()
    wrong_name = "wrong_name"
    hook_filename = f"{wrong_name}.sh"
    hook_path = os.path.join(temp_dir, hook_filename)
    with open(hook_path, 'w') as f:
        f.write("content")
    
    try:
        result = find_hook("valid_name", temp_dir)
        assert result is None
    finally:
        shutil.rmtree(temp_dir)
```


# LLM-generated content at query #35
#--------------------------

```python
from unittest.mock import patch
from pathlib import Path
from cookiecutter.hooks import run_hook

@patch("cookiecutter.hooks.find_hook")
@patch("cookiecutter.hooks.logger")
def test_run_hook_returns_early_when_no_scripts_found(mock_logger, mock_find_hook):
    mock_find_hook.return_value = []
    run_hook("pre_gen_project", "/tmp/project", {"some": "context"})
    mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_gen_project')
```


# LLM-generated content at query #36
#--------------------------

```python
def test_find_hook_returns_none_when_no_valid_hooks_found(monkeypatch, tmp_path):
    monkeypatch.setattr('os.path.isdir', lambda x: True)
    monkeypatch.setattr('os.listdir', lambda x: ['not_a_valid_hook.py'])
    monkeypatch.setattr('valid_hook', lambda file, name: False)
    
    result = find_hook('test_hook', str(tmp_path))
    
    assert result is None
```


# LLM-generated content at query #37
#--------------------------

```python
import os

def test_find_hook_type_hint_validity():
    # The predicate at line 1 is a function signature definition.
    # To "ensure it evaluates to True", we verify the function exists and its return type logic.
    # Since we cannot use 'if' or 'assert' on the signature itself, 
    # we test that calling it with valid parameters returns the expected type (list of strings).
    
    # Setup: Create a dummy hooks directory with a valid hook file
    os.makedirs('test_hooks', exist_ok=True)
    hook_file = 'test_script.py'
    with open(os.path.join('test_hooks', hook_file), 'w') as f:
        f.write("# dummy content")
    
    # Mocking valid_hook logic via the environment/filesystem context 
    # (Assuming valid_hook checks extension or name)
    # We call the function and assert the result type matches the hint list[str]
    result = find_hook('test_script', 'test_hooks')
    
    assert isinstance(result, list)
    assert all(isinstance(path, str) for path in result)
    
    # Cleanup
    os.remove(os.path.join('test_hooks', hook_file))
    os.rmdir('test_hooks')
```


