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

@pytest.mark.parametrize("script_name, context, rendered_content", [
    ("post_gen_project.py", {"project_name": "test_project"}, "print('Hello test_project')"),
    ("pre_gen_project.sh", {"user": "admin"}, "echo 'Hello admin'"),
])
def test_run_script_with_context(script_name, context, rendered_content):
    """Test that run_script_with_context correctly renders a template and calls run_script."""
    
    # Create a dummy script file with Jinja2 syntax
    template_content = f"print('Hello {{project_name if 'project_name' in context else user}}')"
    if "project_name" in context:
        template_content = "print('Hello {{project_name}}')"
    else:
        template_content = "print('Hello {{user}}')"

    # Mocking dependencies
    # 1. Mock Path.read_text to return our template content
    # 2. Mock create_env_with_context to return a mock Jinja Environment
    # 3. Mock run_script to avoid actual execution
    # 4. Mock tempfile.NamedTemporaryFile to control the file output
    
    with patch("pathlib.Path.read_text", return_value=template_content), \
         patch("cookiecutter.utils.create_env_with_context") as mock_create_env, \
         patch("cookiecutter.hooks.run_script") as mock_run_script, \
         patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        
        # Setup Jinja Mocking
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = rendered_content
        
        # Setup TempFile Mocking
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/dummy_script.py"
        # Simulate the context manager behavior (with statement)
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        # Execution
        script_path = Path(f"/fake/path/{script_name}")
        cwd = "/fake/cwd"
        
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        
        # Assertions
        # Verify template was rendered with the provided context
        mock_template.render.assert_called_once_with(**context)
        
        # Verify the rendered content was written to the temp file
        # We check if write was called with encoded bytes
        mock_temp.write.assert_called_once_with(rendered_content.encode('utf-8'))
        
        # Verify run_script was called with the temp file name and correct cwd
        mock_run_script.assert_called_once_with(mock_temp.name, cwd)

def test_run_script_with_context_file_not_found():
    """Test that run_script_with_context raises error if script path doesn't exist."""
    from cookiecutter.hooks import run_script_with_context
    
    with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            run_script_with_context(Path("non_existent.py"), ".", {})
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import pytest
from unittest.mock import patch

def test_find_hook():
    """Test find_hook with various directory and file scenarios."""
    
    # Scenario 1: hooks_dir does not exist
    with patch('os.path.isdir', return_value=False):
        assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Scenario 2: hooks_dir exists but is empty
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[]):
        assert find_hook('pre_gen_project', 'hooks_dir') is None

    # Scenario 3: hooks_dir exists with valid and invalid hooks
    # Valid: pre_prompt, pre_gen_project, post_gen_project
    # Invalid: wrong name, backup file (~), unsupported hook name
    mock_files = [
        'pre_prompt.py',          # Valid
        'pre_gen_project.sh',     # Valid
        'post_gen_project.py',    # Valid
        'pre_prompt.py~',         # Invalid (backup)
        'wrong_hook.py',          # Invalid (wrong name)
        'other_hook.py',          # Invalid (unsupported name)
        'pre_prompt.txt'          # Valid (basename matches)
    ]
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=mock_files), \
         patch('os.path.abspath', side_effect=lambda x: x):
        
        # Test finding 'pre_prompt'
        # Note: valid_hook uses os.path.basename and splitext
        # 'pre_prompt.py' -> basename 'pre_prompt', matches
        # 'pre_prompt.py~' -> basename 'pre_prompt.py', matches name but is backup
        # 'pre_prompt.txt' -> basename 'pre_prompt', matches
        
        results = find_hook('pre_prompt', 'hooks_dir')
        
        # We expect the absolute paths of the valid files found
        # Based on logic: 
        # 'pre_prompt.py' -> valid
        # 'pre_prompt.txt' -> valid
        # 'pre_prompt.py~' -> invalid (backup_file is True)
        assert len(results) == 2
        assert os.path.join('hooks_dir', 'pre_prompt.py') in results
        assert os.path.join('hooks_dir', 'pre_prompt.txt') in results

        # Test finding 'post_gen_project'
        results_post = find_hook('post_gen_project', 'hooks_dir')
        assert len(results_post) == 1
        assert os.path.join('hooks_dir', 'post_gen_project.py') in results_post

        # Test finding a hook that doesn't exist in the list
        assert find_hook('non_existent', 'hooks_dir') is None
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja2.exceptions import UndefinedError
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize(
    "hook_name, project_dir, context, delete_project_on_failure, exception_to_raise, expected_rmtree_called",
    [
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            True,
            FailedHookException("Hook failed"),
            True,
        ),
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            True,
            UndefinedError("Template error"),
            True,
        ),
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            False,
            FailedHookException("Hook failed"),
            False,
        ),
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            False,
            None,
            False,
        ),
    ],
)
def test_run_hook_from_repo_dir(
    hook_name,
    project_dir,
    context,
    delete_project_on_failure,
    exception_to_raise,
    expected_rmtree_called,
):
    repo_dir = "/tmp/repo"
    
    with patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.work_in") as mock_work_in, \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        
        # Setup the mock to raise an exception if one is provided
        if exception_to_raise:
            mock_run_hook.side_effect = exception_to_raise
        else:
            mock_run_hook.side_effect = None

        # Execute the function
        if exception_to_raise:
            with pytest.raises(type(exception_to_raise)):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=delete_project_on_failure,
                )
        else:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=delete_project_on_failure,
            )

        # Verify work_in was called with repo_dir
        mock_work_in.assert_called_once_with(repo_dir)
        
        # Verify run_hook was called with correct arguments
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        
        # Verify rmtree was called only when expected
        if expected_rmtree_called:
            mock_rmtree.assert_called_once_with(project_dir)
        else:
            mock_rmtree.assert_not_called()
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja2.exceptions import UndefinedError
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize(
    "hook_name, project_dir, context, delete_project_on_failure, exception_to_raise, expected_rmtree_called",
    [
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            True,
            FailedHookException("Failed"),
            True,
        ),
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            True,
            UndefinedError("Undefined"),
            True,
        ),
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            False,
            FailedHookException("Failed"),
            False,
        ),
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            False,
            None,
            False,
        ),
    ],
)
def test_run_hook_from_repo_dir(
    hook_name,
    project_dir,
    context,
    delete_project_on_failure,
    exception_to_raise,
    expected_rmtree_called,
):
    repo_dir = "/tmp/repo"
    
    with patch("cookiecutter.utils.work_in"), \
         patch("run_hook_from_repo_dir.run_hook") as mock_run_hook, \
         patch("cookiecutter.utils.rmtree") as mock_rmtree:
        
        if exception_to_raise:
            mock_run_hook.side_effect = exception_to_raise
        
        if pytest.raises(exception_to_raise if exception_to_raise else Exception):
            # If no exception is raised in the code, we don't want the test to fail 
            # unless we specifically expect an exception.
            # However, the function raises the caught exception.
            try:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=delete_project_on_failure,
                )
            except (FailedHookException, UndefinedError):
                pass
        else:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=delete_project_on_failure,
            )

        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        
        if expected_rmtree_called:
            mock_rmtree.assert_called_once_with(project_dir)
        else:
            mock_rmtree.assert_not_called()
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

@pytest.fixture
def temp_repo_dir(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    return tmp_path

def test_run_pre_prompt_hook(temp_repo_dir):
    """
    Test run_pre_prompt_hook with various scenarios:
    1. No pre_prompt hook exists.
    2. pre_prompt hook exists and runs successfully.
    3. pre_prompt hook exists but fails.
    """
    
    # Scenario 1: No pre_prompt hook exists in the repo_dir
    # The function should return the original repo_dir
    with patch("cookiecutter.hooks.hooks.work_in"):
        result = run_pre_prompt_hook(str(temp_repo_dir))
        assert result == str(temp_repo_dir)

    # Scenario 2: pre_prompt hook exists and runs successfully
    # We create a dummy hook file and mock the execution
    hooks_dir = temp_repo_dir / "hooks"
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('hello')", encoding="utf-8")
    
    # Mocking create_tmp_repo_dir to return a new tmp path to simulate isolation
    new_tmp_dir = temp_repo_dir / "new_tmp_dir"
    new_tmp_dir.mkdir()
    
    # We need to make sure the hook is found in the "new" directory too
    # So we duplicate the hook in the new directory
    new_hooks_dir = new_tmp_dir / "hooks"
    new_hooks_dir.mkdir()
    new_hook_file = new_hooks_dir / "pre_prompt.py"
    new_hook_file.write_text("#!/usr/bin/env python\nprint('hello')", encoding="utf-8")

    with patch("cookiecutter.hooks.hooks.create_tmp_repo_dir", return_value=str(new_tmp_dir)):
        with patch("cookiecutter.hooks.hooks.run_script") as mock_run_script:
            result = run_pre_prompt_hook(str(temp_repo_dir))
            
            assert result == str(new_tmp_dir)
            mock_run_script.assert_called_once()
            # Check if the script path passed to run_script is the one in the tmp dir
            args, _ = mock_run_script.call_args
            assert str(new_hook_file) in args[0]

    # Scenario 3: pre_prompt hook exists but fails
    with patch("cookiecutter.hooks.hooks.create_tmp_repo_dir", return_value=str(new_tmp_dir)):
        with patch("cookiecutter.hooks.hooks.run_script") as mock_run_script:
            mock_run_script.side_effect = FailedHookException("Hook failed")
            
            with pytest.raises(FailedHookException) as excinfo:
                run_pre_tmp_hook_wrapper(temp_repo_dir)
            
            assert "Pre-Prompt Hook script failed" in str(excinfo.value)

# Helper to avoid complex mocking of work_in context manager in the failure test
def run_pre_tmp_hook_wrapper(repo_dir):
    # This mimics the logic inside run_pre_prompt_hook to allow clean testing of the try/except block
    with patch("cookiecutter.hooks.hooks.work_in"):
        # Manually trigger the logic that would be inside the second work_in block
        # because mocking context managers is brittle
        scripts = ["/tmp/fake_hook.py"] 
        with patch("cookiecutter.hooks.hooks.find_hook", return_value=scripts):
            with patch("cookiecutter.hooks.hooks.run_script") as mock_run:
                mock_run.side_effect = FailedHookException("Original Error")
                # We use the actual function call logic here
                from cookiecutter.hooks.hooks import run_pre_prompt_hook
                run_pre_prompt_hook(str(repo_dir))
```

*Note: Since I cannot use imports, I am assuming `pytest`, `unittest.mock`, `pathlib`, and the module under test are available in the environment.*


# LLM-generated content at query #6
#--------------------------

```python
import os
import pytest
from unittest.mock import patch

def test_find_hook():
    """Tests the find_hook function with various scenarios."""
    
    # Scenario 1: hooks_dir does not exist
    with patch('os.path.isdir', return_value=False):
        assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Scenario 2: hooks_dir exists but is empty
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[]):
        assert find_hook('pre_gen_project', 'hooks_dir') is None

    # Scenario 3: hooks_dir exists but contains no valid hooks for the requested name
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['post_gen_project.py', 'other_script.sh']):
        # Only searching for pre_gen_project
        assert find_hook('pre_gen_project', 'hooks_dir') is None

    # Scenario 4: hooks_dir contains valid hooks
    # We mock os.path.abspath to return predictable strings for testing
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_gen_project.py', 'post_gen_project.py', 'pre_prompt.sh']), \
         patch('os.path.abspath', side_effect=lambda x: f"/abs/path/{os.path.basename(x)}"):
        
        # Search for pre_gen_project
        results = find_hook('pre_gen_project', 'hooks_dir')
        assert results == ['/abs/path/pre_gen_project.py']

        # Search for post_gen_project
        results = find_hook('post_gen_project', 'hooks_dir')
        assert results == ['/abs/path/post_gen_project.py']

    # Scenario 5: hooks_dir contains invalid hook names (not in _HOOKS)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['invalid_hook.py']):
        assert find_hook('pre_gen_project', 'hooks_dir') is None

    # Scenario 6: hooks_dir contains backup files (ending with ~)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_gen_project.py~']):
        assert find_hook('pre_gen_project', 'hooks_dir') is None
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock
from cookiecutter.exceptions import FailedHookException

def test_run_script(tmp_path):
    """Test run_script with various scenarios: success, python script, non-python script, and failure."""
    
    # 1. Test success with a Python script
    py_script = tmp_path / "test_script.py"
    py_script.write_text("import sys; print('hello'); sys.exit(0)")
    
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        run_script(str(py_script), cwd=str(tmp_path))
        
        # Verify it used sys.executable for .py files
        args, kwargs = mock_popen.call_args
        assert args[0][0] == sys.executable
        assert args[0][1] == str(py_script)
        assert kwargs["cwd"] == str(tmp_path)

    # 2. Test success with a shell script (non-python)
    sh_script = tmp_path / "test_script.sh"
    sh_script.write_text("#!/bin/bash\nexit 0")
    
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = Magicarg = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        run_script(str(sh_script), cwd=str(tmp_path))
        
        # Verify it uses the script path directly for non-py files
        args, kwargs = mock_popen.call_args
        assert args[0][0] == str(sh_script)

    # 3. Test failure via non-zero exit status
    fail_script = tmp_path / "fail_script.py"
    fail_script.write_text("import sys; sys.exit(1)")
    
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc
        
        with pytest.raises(FailedHookException, match="Hook script failed \(exit status: 1\)"):
            run_script(str(fail_script), cwd=str(tmp_path))

    # 4. Test failure via OSError (ENOEXEC - e.g., missing shebang/empty file)
    empty_script = tmp_path / "empty.py"
    empty_script.write_text("")
    
    with patch("subprocess.Popen") as mock_popen:
        # Simulate ENOEXEC error
        mock_popen.side_effect = OSError(errno.ENOEXEC, "Exec format error")
        
        with pytest.raises(FailedHookException, match="might be an empty file or missing a shebang"):
            run_script(str(empty_script), cwd=str(tmp_path))

    # 5. Test failure via generic OSError
    broken_script = tmp_path / "broken.py"
    broken_script.write_text("print('fail')")
    
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.side_effect = OSError(errno.EACCES, "Permission denied")
        
        with pytest.raises(FailedHookException, match="Hook script failed \(error:"):
            run_script(str(broken_script), cwd=str(tmp_path))
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja2.exceptions import UndefinedError
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize("hook_name, delete_on_fail, side_effect, expected_raise", [
    ("post_gen_project", True, None, None),
    ("post_gen_project", False, None, None),
    ("post_gen_project", True, FailedHookException("Fail"), FailedHookException),
    ("post_gen_project", True, UndefinedError("Fail"), UndefinedError),
])
def test_run_hook_from_repo_dir(hook_name, delete_on_fail, side_effect, expected_raise):
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    context = {"foo": "bar"}

    with patch("work_in") as mock_work_in, \
         patch("run_hook") as mock_run_hook, \
         patch("rmtree") as mock_rmtree:
        
        if side_effect:
            mock_run_hook.side_effect = side_effect

        if expected_raise:
            with pytest.raises(expected_raise):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=delete_on_fail
                )
            
            if delete_on_fail:
                mock_rmtree.assert_called_once_with(project_dir)
        else:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=delete_on_fail
            )
            mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
            if not delete_on_fail:
                mock_rmtree.assert_not_called()
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock
from cookiecutter.exceptions import FailedHookException

def test_run_script(tmp_path):
    """
    Tests the run_script function for various scenarios:
    1. Successful execution of a Python script.
    2. Successful execution of a shell script.
    3. Failure of a script (non-zero exit status).
    4. Failure due to OSError (e.g., ENOEXEC).
    """
    
    # 1. Test successful Python script execution
    py_script = tmp_path / "test_script.py"
    py_script.write_text("import sys; sys.exit(0)")
    
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        # Should not raise any exception
        run_script(str(py_script), cwd=str(tmp_path))
        
        # Verify command construction for .py files
        args, kwargs = mock_popen.call_args
        assert args[0] == [sys.executable, str(py_script)]
        assert kwargs["cwd"] == str(tmp_path)

    # 2. Test successful shell script execution
    shell_script = tmp_path / "test_hook.sh"
    shell_script.write_text("#!/bin/bash\nexit 0")
    
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        run_script(str(shell_script), cwd=str(tmp_path))
        
        # Verify command construction for non-.py files
        args, kwargs = mock_popen.call_args
        assert args[0] == [str(shell_script)]

    # 3. Test failure via non-zero exit status
    fail_script = tmp_path / "fail_script.py"
    fail_script.write_text("import sys; sys.exit(1)")
    
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc
        
        with pytest.raises(FailedHookException, match="Hook script failed \(exit status: 1\)"):
            run_script(str(fail_script), cwd=str(tmp_path))

    # 4. Test failure via OSError (ENOEXEC)
    with patch("subprocess.Popen") as mock_popen:
        # Simulate ENOEXEC error
        err = OSError()
        err.errno = 8  # errno.ENOEXEC
        mock_popen.side_effect = err
        
        with pytest.raises(FailedHookException, match="might be an empty file or missing a shebang"):
            run_script(str(py_script), cwd=str(tmp_path))

    # 5. Test failure via generic OSError
    with patch("subprocess.Popen") as mock_popen:
        # Simulate a generic OSError (e.g., Permission Denied)
        err = OSError(errno.EACCES, "Permission denied")
        mock_popen.side_effect = err
        
        with pytest.raises(FailedHookException, match="Hook script failed \(error:.*\)"):
            run_script(str(py_script), cwd=str(tmp_path))
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import pytest
from unittest.mock import patch

def test_find_hook():
    """Test the find_hook function with various scenarios."""
    
    # Scenario 1: hooks_dir does not exist
    with patch('os.path.isdir', return_value=False):
        assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Scenario 2: hooks_dir exists but is empty
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[]):
        assert find_hook('pre_gen_project', 'hooks_dir') is None

    # Scenario 3: hooks_dir exists and contains a valid hook
    # We mock os.path.abspath to return a predictable path
    # and os.path.join to simulate the logic
    valid_hook_file = 'pre_gen_project.py'
    expected_path = os.path.abspath(os.path.join('hooks_dir', valid_hook_file))
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[valid_hook_file, 'post_gen_project.py', 'other.txt']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        
        # Testing specifically for 'pre_gen_project'
        # valid_hook will return True for 'pre_gen_project.py' 
        # and True for 'post_gen_project.py'
        # But we are looking for a specific hook_name passed to find_hook
        
        # If we search for 'pre_gen_project'
        results = find_hook('pre_gen_project', 'hooks_dir')
        # Note: find_hook iterates all files and checks valid_hook(file, 'pre_gen_project')
        # valid_hook checks if basename == hook_name.
        # 'pre_gen_project.py' basename is 'pre_gen_project' -> Match!
        # 'post_gen_project.py' basename is 'post_gen_project' -> No Match!
        
        assert results is not None
        assert any(os.path.basename(p) == 'pre_gen_project.py' for p in results)
        assert not any('post_gen_project' in p for p in results if 'pre_gen_project' not in p)

    # Scenario 4: hooks_dir contains a hook but it's a backup file (ends with ~)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_prompt~']):
        assert find_hook('pre_prompt', 'hooks_dir') is None

    # Scenario 5: hooks_dir contains a file with the right name but not in _HOOKS
    # (Though in this specific implementation, the name check happens first, 
    # if the name is 'invalid_hook', valid_hook returns False)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['invalid_hook.py']):
        assert find_hook('invalid_hook', 'hooks_dir') is None
```


# LLM-generated content at query #11
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

def test_run_script_with_context(tmp_path):
    """
    Test run_script_with_context:
    1. Mocks create_env_with_context to return a mock Jinja2 environment.
    2. Mocks the template rendering process.
    3. Verifies that run_script is called with the path to the rendered temporary file.
    """
    # Setup input data
    script_name = "post_gen_project.py"
    script_path = tmp_path / script_name
    content = "print('Hello {{ name }}')"
    context = {"name": "World"}
    expected_output = "print('Hello World')"
    cwd = tmp_path / "project_dir"
    cwd.mkdir()

    # Create the initial script file
    script_path.write_text(content, encoding='utf-8')

    # Mocking the Jinja2 environment and template
    mock_template = MagicMock()
    mock_template.render.return_value = expected_output
    
    mock_env = MagicMock()
    mock_env.from_string.return_value = mock_template

    # Mocking the utilities and subprocess calls
    with patch('cookiecutter.hooks.utils.create_env_with_context', return_value=mock_env), \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        
        # Setup the mock temporary file behavior
        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = mock_temp
        mock_temp.name = str(tmp_path / "temp_script.py")
        mock_temp_file.return_value = mock_temp

        # Execute the function
        run_script_with_context(script_path, cwd, context)

        # Assertions
        # Verify template was created with correct content
        mock_env.from_string.assert_called_once_with(content)
        
        # Verify template was rendered with the correct context
        mock_template.render.assert_called_once_with(**context)

        # Verify the rendered content was written to the temp file
        # We check if write was called with the encoded output
        mock_temp.write.assert_called_once_with(expected_output.encode('utf-8'))

        # Verify run_script was called with the temp file path and the correct cwd
        mock_run_script.assert_called_once_with(mock_temp.name, cwd)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize("scripts, context, expected_calls", [
    ([], {"foo": "bar"}, []),
    (["/abs/path/to/pre_gen_project.py"], {"foo": "bar"}, [
        ("run_script_with_context", ["/abs/path/to/pre_gen_project.py", "/tmp/project", {"foo": "bar"}])
    ]),
    (["/path/a.py", "/path/b.py"], {"x": 1}, [
        ("run_script_with_context", ["/path/a.py", "/tmp/project", {"x": 1}]),
        ("run_script_with_context", ["/path/b.py", "/tmp/project", {"x": 1}]),
    ]),
])
def test_run_hook(scripts, context, expected_calls):
    """
    Tests run_hook by mocking find_hook and run_script_with_context.
    Verifies that scripts are found and executed with the correct arguments.
    """
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"

    with patch("cookiecutter.hooks.find_hook") as mock_find, \
         patch("cookiecutter.hooks.run_script_with_context") as mock_run_ctx:
        
        # Setup mock behavior
        mock_find.return_value = scripts
        
        # Execute function under test
        run_hook(hook_name, project_dir, context)

        # Verify find_hook was called correctly
        mock_find.assert_called_once_with(hook_name)

        # Verify run_script_with_context calls
        if not expected_calls:
            assert mock_run_ctx.call_count == 0
        else:
            assert mock_run_ctx.call_count == len(expected_calls)
            for i, (func_name, args) in enumerate(expected_calls):
                # Check if the call matches the expected arguments
                actual_args = mock_run_ctx.call_args_list[i].args
                assert actual_args == args
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import sys
import subprocess
import errno
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize("script_path, expected_command", [
    ("script.py", [sys.executable, "script.py"]),
    ("script.sh", ["script.sh"]),
])
def test_run_script_success(script_path, expected_command):
    with patch("subprocess.Popen") as mock_popen, \
         patch("cookiecutter.utils.make_executable") as mock_make_executable:
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script(script_path, cwd="/tmp")
        
        mock_make_executable.assert_called_once_with(script_path)
        mock_popen.assert_called_once_with(
            expected_command, 
            shell=sys.platform.startswith('win'), 
            cwd="/tmp"
        )

def test_run_script_failure_exit_status():
    with patch("subprocess.Popen") as mock_popen, \
         patch("cookiecutter.utils.make_executable"):
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        with pytest.raises(FailedHookException, match="Hook script failed \(exit status: 1\)"):
            run_script("script.py")

def test_run_script_os_error_errno_exec():
    with patch("subprocess.Popen") as mock_popen, \
         patch("cookiecutter.utils.make_executeable"):
        
        # Simulate ENOEXEC error (e.g., missing shebang)
        err = OSError()
        err.errno = errno.ENOEXEC
        mock_popen.side_effect = err
        
        with pytest.raises(FailedHookException, match="might be an empty file or missing a shebang"):
            run_script("script.sh")

def test_run_script_other_os_error():
    with patch("subprocess.Popen") as mock_popen, \
         patch("cookiecutter.utils.make_executable"):
        
        err = OSError()
        err.errno = errno.EACCES
        mock_popen.side_effect = err
        
        with pytest.raises(FailedHookException, match="Hook script failed \(error: .*\)"):
            run_script("script.sh")
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

@pytest.fixture
def mock_context():
    return {"project_name": "test_project", "author": "tester"}

@patch("pathlib.Path.read_text")
@patch("tempfile.NamedTemporaryFile")
@patch("cookiecutter.utils.create_env_with_context")
@patch("cookiecutter.hooks.run_script")
def test_run_script_with_context(
    mock_run_script,
    mock_create_env,
    mock_tempfile,
    mock_read_text,
    mock_context
):
    # Setup
    script_path = "/path/to/hook.py"
    cwd = "/path/to/project"
    template_content = "Hello {{ project_name }}!"
    rendered_content = "Hello test_project!"
    
    mock_read_text.return_value = template_content
    
    # Mocking Jinja2 environment and template
    mock_template = MagicMock()
    mock_template.render.return_value = rendered_content
    mock_env = MagicMock()
    mock_env.from_string.return_value = mock_template
    mock_create_env.return_value = mock_env
    
    # Mocking the NamedTemporaryFile context manager
    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "/tmp/temp_script.py"
    mock_temp_file_instance.__enter__.return_value = mock_temp_file_instance
    mock_tempfile.return_value.__enter__.return_value = mock_temp_file_instance
    
    # Execute
    run_script_with_context(script_path, cwd, mock_context)
    
    # Assertions
    mock_read_text.assert_called_once_with(encoding='utf-8')
    mock_create_env.assert_called_once_with(mock_context)
    mock_env.from_string.assert_called_once_with(template_content)
    mock_template.render.assert_called_once_with(**mock_context)
    
    # Verify the temp file was written with the rendered content
    # Note: encode('utf-8') returns bytes
    mock_temp_file_instance.write.assert_called_once_with(
        rendered_content.encode('utf-8')
    )
    
    # Verify the actual execution command was called with the temp file path
    mock_run_script.assert_called_once_with(
        "/tmp/temp_script.py", cwd
    )
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja2.exceptions import UndefinedError
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize("exception_type, should_delete", [
    (FailedHookException, True),
    (UndefinedError, True),
    (ValueError, False),  # Non-hook related error should not trigger deletion
])
def test_run_hook_from_repo_dir(exception_type, should_delete):
    """
    Tests run_hook_from_repo_dir for correct execution, 
    exception handling, and conditional directory deletion.
    """
    repo_dir = "/fake/repo"
    project_dir = "/fake/project"
    hook_name = "pre_gen_project"
    context = {"project_name": "test"}
    
    # Mocking dependencies
    with patch("hooks.work_in") as mock_work_in, \
         patch("hooks.run_hook") as mock_run_hook, \
         patch("hooks.rmtree") as mock_rmtree, \
         patch("logging.Logger.exception") as mock_log_exception:
        
        # Setup the exception to be raised by run_hook
        if exception_type in (FailedHookException, UndefinedError):
            mock_run_hook.side_effect = exception_type("Error occurred")
        else:
            mock_run_hook.side_effect = exception_type("Unexpected error")

        # Execute the function
        with pytest.raises(exception_type):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=should_delete
            )

        # Verify work_in was called with repo_dir
        mock_work_in.assert_called_once_with(repo_dir)
        
        # Verify run_hook was called with correct arguments
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

        # Verify rmtree was called only if deletion was requested and a hook error occurred
        if should_delete and exception_type in (FailedHookException, UndefinedError):
            mock_rmtree.assert_called_once_with(project_dir)
            mock_log_exception.assert_called_once()
        else:
            mock_rmtree.assert_not_called()
            if exception_type not in (FailedHookException, UndefinedError):
                mock_log_exception.assert_not_called()
```


# LLM-generated content at query #16
#--------------------------

```python
import os
import pytest
from unittest.mock import patch

def test_find_hook():
    """Test find_hook with various scenarios including missing dir, no hooks, and valid hooks."""
    
    # Scenario 1: hooks_dir does not exist
    with patch('os.path.isdir', return_value=False):
        assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Scenario 2: hooks_dir exists but contains no valid hooks
    # We mock os.path.isdir to return True and os.listdir to return unrelated files
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['random_file.txt', 'pre_gen_project.bak', 'other_hook.py']):
        # 'pre_gen_project.bak' is invalid due to suffix, 'other_hook' is not in _HOOKS
        assert find_hook('pre_gen_project', 'hooks') is None

    # Scenario 3: hooks_dir exists and contains a valid hook
    # We mock os.path.isdir to return True and os.listdir to return a valid hook name
    # We also need to mock os.path.abspath to return a predictable path
    valid_hook_name = 'pre_gen_project'
    hook_filename = 'pre_gen_project.py'
    expected_path = os.path.abspath(os.path.join('hooks', hook_filename))
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[hook_filename]), \
         patch('os.path.abspath', side_effect=lambda x: x):
        
        # Note: valid_hook logic is used inside find_hook. 
        # Since we are patching os.path.abspath to return the input, 
        # we ensure the result matches our expectation.
        result = find_hook(valid_hook_name, 'hooks')
        assert result == [os.path.join('hooks', hook_filename)]

    # Scenario 4: hooks_dir exists but contains only an invalid hook name (not in _HOOKS)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['invalid_hook_name.py']):
        assert find_hook('pre_gen_project', 'hooks') is None

    # Scenario 5: hooks_dir exists but contains a hook with a backup extension (~)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_gen_project.py~']):
        assert find_hook('pre_gen_project', 'hooks') is None
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException

@pytest.fixture
def temp_repo_dir(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    return tmp_path

def test_run_pre_prompt_hook_no_hooks(temp_repo_dir):
    """Test that it returns original dir if no pre_prompt hook exists."""
    result = run_pre_prompt_hook(str(temp_repo_dir))
    assert result == str(temp_repo_dir)

def test_run_pre_prompt_hook_success(temp_repo_dir):
    """Test successful execution of a pre_prompt hook."""
    hooks_dir = temp_repo_dir / "hooks"
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nimport sys\nsysys.exit(0)")
    
    # We need to mock create_tmp_repo_dir because it creates a new directory
    # and we want to control the environment for the test
    with patch("cookiecutter.hooks.run_pre_prompt_hook.create_tmp_repo_dir") as mock_tmp_dir, \
         patch("cookiecutter.hooks.run_pre_prompt_hook.run_script") as mock_run:
        
        # Setup mock to return a new path and simulate finding the script in that path
        new_tmp_path = Path(temp_repo_dir, "new_tmp")
        mock_tmp_dir.return_value = str(new_tmp_path)
        
        # Mock find_hook to return our script
        with patch("cookiecutter.hooks.run_pre_prompt_hook.find_hook") as mock_find:
            mock_find.side_effect = [
                [str(hook_script)], # First call in work_in(repo_dir)
                [str(hook_script)]  # Second call in work_in(repo_dir_tmp)
            ]
            
            result = run_pre_prompt_hook(str(temp_repo_dir))
            
            assert result == str(new_tmp_path)
            mock_run.assert_called_once_with(str(hook_script), str(new_tmp_path))

def test_run_pre_prompt_hook_failure(temp_repo_dir):
    """Test that FailedHookException is re-raised during pre_prompt execution."""
    hooks_dir = temp_repo_dir / "hooks"
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")

    with patch("cookiecutter.hooks.run_pre_prompt_hook.create_tmp_repo_dir") as mock_tmp_dir, \
         patch("cookiecutter.hooks.run_pre_prompt_hook.run_script") as mock_run:
        
        new_tmp_path = Path(temp_repo_dir, "new_tmp")
        mock_tmp_dir.return_value = str(new_tmp_path)
        
        with patch("cookiecutter.hooks.run_pre_prompt_hook.find_hook") as mock_find:
            mock_find.side_effect = [
                [str(hook_script)], 
                [str(hook_script)]
            ]
            # Simulate the script failing
            mock_run.side_effect = FailedHookException("Script failed")
            
            with pytest.raises(FailedHookException) as excinfo:
                run_pre_prompt_hook(str(temp_repo_dir))
            
            assert "Pre-Prompt Hook script failed" in str(excinfo.value)
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import pytest
from unittest.mock import patch

def test_find_hook():
    """Test the find_hook function with various scenarios."""
    
    # Scenario 1: hooks_dir does not exist
    with patch('os.path.isdir', return_value=False):
        assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Scenario 2: hooks_dir exists but is empty
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[]):
        assert find_hook('pre_gen_project', 'hooks') is None

    # Scenario 3: hooks_dir exists but contains no matching hooks
    # (e.g., wrong hook name or invalid hook type)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['post_gen_project.sh', 'random_script.py']):
        # Looking for 'pre_prompt' but only 'post_gen_project' exists
        assert find_hook('pre_prompt', 'hooks') is None

    # Scenario 4: hooks_dir contains a valid hook
    # We mock os.path.abspath to return predictable paths for testing
    valid_hook_file = 'pre_prompt.py'
    expected_path = os.path.abspath(os.path.join('hooks', valid_hook_file))
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[valid_hook_file]), \
         patch('os.path.abspath', side_effect=lambda x: x):
        # Note: we bypass the real abspath logic to keep it simple
        # In a real environment, abspath would return the full system path
        results = find_hook('pre_prompt', 'hooks')
        assert results is not None
        assert len(results) == 1
        assert results[0].endswith(valid_hook_file)

    # Scenario 5: hooks_dir contains a valid hook name but it is a backup file (ends with ~)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_prompt.py~']):
        assert find_hook('pre_prompt', 'hooks') is None

    # Scenario 6: hooks_dir contains multiple valid hooks
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_prompt.py', 'post_gen_project.sh']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        results = find_hook('pre_prompt', 'hooks')
        # Only pre_prompt matches the requested hook_name
        assert len(results) == 1
        assert 'pre_prompt.py' in results[0]
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

def test_run_script_with_context(tmp_path):
    """
    Test run_script_with_context:
    1. Verifies that the script content is read.
    2. Verifies that Jinja2 renders the content with the provided context.
    3. Verifies that a temporary file is created with the rendered content.
    4. Verifies that run_script is called with the temporary file path and cwd.
    """
    # Setup: Create a template script with a Jinja2 variable
    script_content = "Hello {{ name }}!"
    script_file = tmp_path / "pre_gen_project.py"
    script_file.write_text(script_content, encoding="utf-8")
    
    context = {"name": "World"}
    cwd = str(tmp_path)
    
    # Mocking dependencies
    # 1. Mock create_env_with_context to return a mock Jinja Environment
    # 2. Mock run_script to prevent actual execution of the generated temp file
    with patch("cookiecutter.hooks.hooks.create_env_with_context") as mock_create_env, \
         patch("cookiecutter.hooks.hooks.run_script") as mock_run_script, \
         patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        
        # Setup Mock Jinja Environment and Template
        mock_env = MagicMock()
        mock_template = Magiclama_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "Hello World!"
        
        # Setup Mock Temporary File
        # We need to simulate the context manager behavior of NamedTemporaryFile
        mock_temp_instance = MagicMock()
        # Mock the __enter__ to return the mock_temp_instance
        mock_temp_file.return_value.__enter__.return_value = mock_temp_instance
        # Ensure the name attribute is something predictable for the test
        mock_temp_instance.name = str(tmp_path / "temp_script.py")
        
        # Execute the function
        from cookiecutter.hooks.hooks import run_script_with_context
        run_script_with_context(script_file, cwd, context)
        
        # Assertions
        # Check if env was created with correct context
        mock_create_env.assert_called_once_with(context)
        
        # Check if template was created from the original script content
        mock_env.from_string.assert_called_once_with(script_content)
        
        # Check if template was rendered with the context
        mock_template.render.assert_called_once_with(**context)
        
        # Check if run_script was called with the temp file path and correct cwd
        mock_run_script.assert_called_once_with(
            mock_temp_instance.name, 
            cwd
        )
        
        # Verify that the rendered content was written to the temp file
        # Note: .encode('utf-8') is called on the rendered string
        mock_temp_instance.write.assert_called_once_with(b"Hello World!")
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError

@pytest.mark.parametrize("hook_exception, should_delete", [
    (FailedHookException("Failed"), True),
    (UndefinedError("Undefined"), True),
    (ValueError("Not a hook error"), False),
])
def test_run_hook_from_repo_dir(hook_exception, should_delete):
    """
    Tests run_hook_from_repo_dir for correct handling of hook failures,
    deletion of project directory, and re-raising of specific exceptions.
    """
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    hook_name = "pre_gen_project"
    context = {"project_name": "test"}

    with patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.work_in") as mock_work_in, \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        
        # Setup the side effect to raise the target exception
        mock_run_hook.side_effect = hook_exception
        
        # If the exception is one of the caught ones, we expect it to be re-raised
        if isinstance(hook_exception, (FailedHookException, UndefinedError)):
            with pytest.raises((FailedHookException, UndefinedError)):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=should_delete
                )
            
            # Verify rmtree was called only if delete_project_on_failure is True
            if should_delete:
                mock_rmtree.assert_called_once_with(project_dir)
            else:
                mock_rmtree.assert_not_called()
        else:
            # If it's an unhandled exception, it should propagate without rmtree
            with pytest.raises(ValueError):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=should_delete
                )
            mock_rmtree.assert_not_called()

        # Ensure work_in was called with the correct repo_dir
        mock_work_in.assert_called_with(repo_dir)
        # Ensure the hook was actually attempted
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

def test_run_hook_from_repo_dir_success():
    """
    Tests run_hook_from_repo_dir when the hook executes successfully.
    """
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    hook_name = "pre_gen_project"
    context = {"project_name": "test"}

    with patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.work_in") as mock_work_in, \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        
        # Successful execution
        mock_run_hook.return_value = None
        
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name=hook_name,
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )

        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_not_called()
        mock_work_in.assert_called_with(repo_dir)
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.fixture
def temp_repo_dir(tmp_path):
    """Fixture to create a temporary repository directory."""
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    return repo_dir


def test_run_pre_prompt_hook_no_hooks(temp_repo_dir):
    """Test that the function returns the original repo_dir if no pre_prompt hook exists."""
    # No hooks directory or pre_prompt script created
    result = run_pre_prompt_hook(str(temp_repo_dir))
    assert result == str(temp_repo_dir)


def test_run_pre_prompt_hook_with_valid_hook(temp_repo_dir, tmp_path):
    """Test that the function executes the pre_prompt hook if it exists."""
    hooks_dir = temp_repo_dir / "hooks"
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nimport sys\nsys_exit = 0\n", encoding="utf-8")
    
    # Mock run_script to avoid actual execution of subprocess
    with patch("cookiecutter.hooks.run_script") as mock_run_script:
        # We need to mock create_tmp_repo_dir because the function calls it
        # to create a copy of the repo to run the hook in isolation.
        with patch("cookiecutter.hooks.create_tmp_repo_dir", return_value=str(temp_repo_dir)) as mock_tmp_dir:
            result = run_pre_prompt_hook(str(temp_repo_dir))
            
            assert result == str(temp_repo_dir)
            mock_run_script.assert_called()
            # Verify the script path passed to run_script is the one we created
            args, _ = mock_run_script.call_args
            assert str(hook_script) in args[0]


def test_run_pre_prompt_hook_failure(temp_repo_dir):
    """Test that FailedHookException is raised if the pre_prompt hook fails."""
    hooks_dir = temp_repo_dir / "hooks"
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)\n", encoding="utf-8")

    with patch("cookiecutter.hooks.run_script") as mock_run_script:
        mock_run_script.side_effect = FailedHookException("Hook failed")
        
        with patch("cookiecutter.hooks.create_tmp_repo_dir", return_value=str(temp_repo_dir)):
            with pytest.raises(FailedHookException) as excinfo:
                run_pre_prompt_hook(str(temp_repo_dir))
            
            assert "Pre-Prompt Hook script failed" in str(excinfo.value)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock
from cookiecutter.exceptions import FailedHookException

def test_run_script(tmp_path):
    """
    Tests the run_script function for various scenarios including 
    successful execution, Python script execution, and failure handling.
    """
    
    # Scenario 1: Successful execution of a shell script
    script_shell = tmp_path / "test_hook.sh"
    script_shell.write_text("#!/bin/bash\nexit 0")
    
    with patch("subprocess.Popen") as mock_popen, \
         patch("cookiecutter.utils.make_executable") as mock_make_exec:
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Should not raise exception
        run_script(str(script_shell), cwd=str(tmp_path))
        
        mock_make_exec.assert_called_once_with(str(script_shell))
        # Check if Popen was called with the correct command
        args, kwargs = mock_popen.call_args
        assert args[0] == [str(script_shell)]
        assert kwargs["cwd"] == str(tmp_path)

    # Scenario 2: Successful execution of a Python script
    script_py = tmp_path / "test_hook.py"
    script_py.write_text("import sys; sys.exit(0)")
    
    with patch("subprocess.Popen") as mock_popen, \
         patch("cookiecutter.utils.make_executable") as mock_make_exec:
        
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        run_script(str(script_py), cwd=str(tmp_path))
        
        # Verify it uses sys.executable for .py files
        args, _ = mock_popen.call_args
        assert args[0] == [sys.executable, str(script_py)]

    # Scenario 3: Script fails with non-zero exit status
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        with pytest.raises(FailedHookException, match="Hook script failed \(exit status: 1\)"):
            run_script(str(script_shell), cwd=str(tmp_path))

    # Scenario 4: Script fails with OSError (ENOEXEC)
    with patch("subprocess.Popen") as mock_popen:
        err = OSError()
        err.errno = os.errno.ENOEXEC
        mock_popen.side_effect = err
        
        with pytest.raises(FailedHookException, match="might be an empty file or missing a shebang"):
            run_script(str(script_shell), cwd=str(tmp_path))

    # Scenario 5: Script fails with general OSError
    with patch("subprocess.Popen") as mock_popen:
        err = OSError()
        err.errno = os.errno.EACCES  # Permission denied
        mock_popen.side_effect = err
        
        with pytest.raises(FailedHookException, match="Hook script failed \(error:"):
            run_script(str(script_shell), cwd=str(tmp_path))
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize("hooks_found, script_fails, expected_return_val, expected_exception", [
    # Case 1: No pre_prompt hooks found in the original repo_dir
    ([], False, None, None),
    
    # Case 2: Hooks found, script runs successfully
    (["/tmp/hooks/pre_prompt.py"], False, "/tmp/tmp_repo_dir", None),
    
    # Case 3: Hooks found, but the script execution raises FailedHookException
    (["/tmp/hooks/pre_prompt.py"], True, None, FailedHookException),
])
def test_run_pre_prompt_hook(hooks_found, script_fails, expected_return_val, expected_exception):
    """
    Tests run_pre_prompt_hook for various scenarios:
    - No hooks present (returns original repo_dir)
    - Hooks present and run successfully (returns new tmp_repo_dir)
    - Hooks present but execution fails (raises FailedHookException)
    """
    repo_dir = "/original/repo_dir"
    tmp_repo_dir = "/tmp/tmp_repo_dir"
    
    # Mocking dependencies
    # We need to mock work_in to avoid actual directory changes
    # We need to mock find_hook to control hook discovery
    # We need to mock create_tmp_repo_dir to control the new directory path
    # We need to mock run_script to simulate success or failure
    
    with patch('cookiecutter.utils.work_in'), \
         patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecal.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        
        # Setup find_hook behavior
        # First call is for the original repo_dir check, second is for the tmp_repo_dir execution
        if not hooks_found:
            mock_find_hook.side_effect = [None]
        else:
            mock_find_hook.side_effect = [hooks_found, hooks_found]
            
        # Setup create_tmp_repo_dir behavior
        mock_create_tmp.return_value = tmp_repo_dir
        
        # Setup run_script behavior
        if script_fails:
            mock_run_script.side_effect = FailedHookException("Hook failed")
        else:
            mock_run_script.return_value = None

        # Execute the function
        if expected_exception:
            with pytest.raises(expected_exception):
                run_pre_prompt_hook(repo_dir)
        else:
            result = run_pre_prompt_hook(repo_dir)
            
        # Assertions
        if not hooks_found:
            # Should return original dir and not have called create_tmp_repo_dir
            assert result == repo_dir
            mock_create_tmp.assert_not_called()
        else:
            # Should have returned the new tmp dir
            assert result == tmp_repo_dir
            # Should have attempted to run the scripts found
            assert mock_run_script.called
            for script_path in hooks_found:
                mock_run_script.assert_any_call(script_path, tmp_repo_dir)
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize("should_fail, hook_name, hooks_dir_content", [
    (False, "pre_prompt", ["pre_prompt.py"]),
    (True, "pre_prompt", ["pre_prompt.py"]),
    (False, "invalid_hook", ["pre_prompt.py"]),
    (False, "pre_prompt", ["other_script.py"]),
])
def test_run_pre_prompt_hook(should_fail, hook_name, hooks_dir_content):
    """
    Tests run_pre_prompt_hook for various scenarios:
    1. No hook found: Returns original repo_dir.
    2. Hook found and succeeds: Returns new tmp_repo_dir.
    3. Hook found but fails: Raises FailedHookException.
    4. Hook name is invalid: Returns original repo_dir.
    """
    repo_dir = "/fake/repo_dir"
    tmp_repo_dir = "/fake/tmp_repo_dir"
    
    # Mocking dependencies
    with patch("os.path.isdir") as mock_isdir, \
         patch("os.listdir") as mock_listdir, \
         patch("cookiecutter.utils.work_in") as mock_work_in, \
         patch("cookiecutter.utils.create_tmp_repo_dir") as mock_create_tmp_dir, \
         patch("cookiecutter.hooks.hooks.find_hook") as mock_find_hook, \
         patch("cookiecutter.hooks.hooks.run_script") as mock_run_script:

        # Setup mocks for initial check (inside first work_in)
        mock_isdir.return_value = True
        mock_listdir.return_value = hooks_dir_content
        
        # Setup mocks for the second work_in (the tmp directory)
        mock_create_tmp_dir.return_value = tmp_repo_dir
        
        if hook_name == "pre_prompt" and hooks_dir_content == ["pre_prompt.py"]:
            # If we are testing the success/failure of the actual execution
            mock_find_hook.side_effect = [
                ["/fake/repo_dir/hooks/pre_prompt.py"], # First call in original repo_dir
                ["/fake/tmp_repo_dir/hooks/pre_prompt.py"] # Second call in tmp_repo_dir
            ]
            
            if should_fail:
                mock_run_script.side_effect = FailedHookException("Hook failed")
                with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
                    run_pre_prompt_hook(repo_dir)
            else:
                mock_run_script.return_value = None
                result = run_pre_prompt_hook(repo_dir)
                assert result == tmp_repo_dir
                mock_run_script.assert_called_once()
        else:
            # If no hook is found or invalid hook name, it should return original repo_dir
            mock_find_hook.return_value = None
            result = run_pre_prompt_hook(repo_dir)
            assert result == repo_dir
            mock_create_tmp_dir.assert_not_called()

def test_run_pre_prompt_hook_exec_failure_wraps_exception():
    """Specific test to ensure the exception message is wrapped correctly."""
    repo_dir = "/fake/repo_dir"
    tmp_repo_dir = "/fake/tmp_repo_dir"
    
    with patch("cookiecutter.hooks.hooks.work_in"), \
         patch("cookiecutter.hooks.hooks.find_hook") as mock_find_hook, \
         patch("cookiecutter.utils.create_tmp_repo_dir") as mock_create_tmp_dir, \
         patch("cookiecutter.hooks.hooks.run_script") as mock_run_script:

        # Scenario: Hook exists, but execution fails
        mock_find_hook.side_effect = [
            ["/fake/repo_dir/hooks/pre_prompt.py"], 
            ["/fake/tmp_repo_dir/hooks/pre_prompt.py"]
        ]
        mock_create_tmp_dir.return_value = tmp_repo_dir
        mock_run_script.side_effect = FailedHookException("Original Error")

        with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
            run_pre_prompt_hook(repo_dir)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize("hook_name, scripts, context, expected_calls", [
    # Case 1: No scripts found, nothing should be executed
    ("pre_gen_project", None, {"project_name": "test"}, []),
    
    # Case 2: Scripts found, run_script_with_context should be called for each
    (
        "pre_gen_project", 
        ["/tmp/hooks/pre_gen_project.py", "/tmp/hooks/pre_gen_project.sh"], 
        {"project_name": "test"}, 
        [
            patch.object(MagicMock, 'run_script_with_context', return_value=None)
        ]
    ),
])
def test_run_hook(hook_name, scripts, context, expected_calls):
    """
    Tests run_hook logic for both the case where no hooks are found
    and the case where multiple hooks are found and executed.
    """
    with patch('find_hook') as mock_find_hook:
        with patch('run_script_with_context') as mock_run_context:
            # Setup the mock to return our scripts list
            mock_find_hook.return_value = scripts
            
            project_dir = "/tmp/project"
            
            # Execute the function
            run_hook(hook_name, project_dir, context)
            
            # Assertions
            if scripts is None:
                # If no scripts, run_script_with_context should never be called
                mock_run_context.assert_not_called()
            else:
                # If scripts exist, it should call run_script_with_context for each script
                assert mock_run_context.call_count == len(scripts)
                for i, script_path in enumerate(scripts):
                    mock_run_context.assert_any_call(script_path, project_dir, context)

def test_run_hook_integration_flow():
    """
    A more detailed test simulating a successful execution flow 
    of run_hook with a single valid script.
    """
    hook_name = "post_gen_project"
    project_dir = Path("/tmp/output")
    context = {"user": "tester"}
    script_path = "/tmp/template/hooks/post_gen_project.py"

    with patch('find_hook', return_value=[script_path]):
        with patch('run_script_with_context') as mock_run_context:
            run_hook(hook_name, project_dir, context)
            
            mock_run_context.assert_called_once_with(
                script_path, project_dir, context
            )
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja2.exceptions import UndefinedError
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize(
    "hook_name, project_dir, context, delete_on_failure, should_raise, mock_run_hook_side_effect",
    [
        # Case 1: Success - Hook runs without error
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            False,
            False,
            None,
        ),
        # Case 2: Failure - Hook raises FailedHookException, delete_project_on_failure is True
        (
            "post_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            True,
            True,
            FailedHookException("Hook failed"),
        ),
        # Case 3: Failure - Hook raises FailedHookException, delete_project_on_failure is False
        (
            "post_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            False,
            True,
            FailedHookException("Hook failed"),
        ),
        # Case 4: Failure - Hook raises UndefinedError (Jinja error), delete_project_on_failure is True
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            True,
            True,
            UndefinedError("Undefined variable"),
        ),
        # Case 5: Success - Hook runs, but no scripts found (run_hook returns None)
        (
            "non_existent_hook",
            "/tmp/project",
            {"project_name": "test"},
            True,
            False,
            None,
        ),
    ],
)
def test_run_hook_from_repo_dir(
    hook_name,
    project_dir,
    context,
    delete_on_failure,
    should_raise,
    mock_run_hook_side_effect,
):
    repo_dir = "/tmp/repo"
    
    with patch("work_in") as mock_work_in, \
         patch("run_hook") as mock_run_hook, \
         patch("rmtree") as mock_rmtree:
        
        # Setup mock behavior for run_hook
        if mock_run_hook_side_effect:
            mock_run_hook.side_effect = mock_run_hook_side_effect
        else:
            mock_run_hook.return_value = None

        if should_raise:
            with pytest.raises((FailedHookException, UndefinedError)):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=delete_on_failure,
                )
        else:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=delete_on_failure,
            )

        # Verify work_in was called with repo_dir
        mock_work_in.assert_called_with(repo_dir)
        
        # Verify run_hook was called with correct parameters
        mock_run_hook.assert_called_with(hook_name, project_dir, context)
        
        # Verify rmtree was called only if failure occurred and delete flag was True
        if delete_on_failure and mock_run_hook_side_effect:
            mock_rmtree.assert_called_once_with(project_dir)
        else:
            mock_rmtree.assert_not_called()
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

@pytest.fixture
def context():
    return {"project_name": "test_project", "user": "tester"}

@pytest.fixture
def script_content():
    return "Hello {{ user }}! Welcome to {{ project_name }}."

@pytest.fixture
def temp_script_file(tmp_path, script_content):
    script_path = tmp_path / "pre_gen_project.py"
    script_path.write_text(script_content, encoding="utf-8")
    return script_path

def test_run_script_with_context(tmp_path, temp_script_file, context, script_content):
    """
    Test that run_script_with_context correctly renders a Jinja2 template,
    writes it to a temporary file, and calls run_script.
    """
    cwd = str(tmp_path)
    
    # We patch:
    # 1. create_env_with_context to return a mock Jinja environment
    # 2. run_script to verify it's called with the rendered content
    # 3. Path.read_text to control the input
    
    with patch("cookiecutter.hooks.hooks.create_env_with_template_env") as mock_create_env, \
         patch("cookiecutter.hooks.hooks.run_script") as mock_run_script, \
         patch("pathlib.Path.read_text") as mock_read_text:
        
        # Setup Mock Environment
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        
        # Setup Mock Rendering
        rendered_output = "Hello tester! Welcome to test_project."
        mock_template.render.return_value = rendered_output
        mock_read_text.return_value = "Hello {{ user }}! Welcome to {{ project_name }}."

        # Execute the function
        from cookiecutter.hooks.hooks import run_script_with_context
        run_script_with_context(temp_script_file, cwd, context)

        # Assertions
        # Verify Jinja was called with the original content
        mock_env.from_string.assert_called_once_with(script_content)
        
        # Verify rendering was called with context
        mock_template.render.assert_called_once_with(**context)

        # Verify run_script was called
        # Since the function uses NamedTemporaryFile, we can't easily predict the exact path,
        # but we can verify run_script was called with some path that exists and the correct cwd
        args, kwargs = mock_run_script.call_args
        called_script_path = args[0]
        
        assert os.path.exists(called_script_path)
        assert args[1] == cwd
        
        # Verify the content of the rendered temporary file
        with open(called_script_path, 'r', encoding='utf-8') as f:
            assert f.read() == rendered_output

def test_run_script_with_context_extension_handling(tmp_path, context):
    """
    Test that run_script_with_context preserves the file extension in the temp file.
    """
    script_path = tmp_path / "hook.sh"
    script_path.write_text("#!/bin/bash\necho 'test'", encoding="utf-8")
    
    with patch("cookiecutter.hooks.hooks.create_env_with_context") as mock_env_func, \
         patch("cookiecutter.hooks.hooks.run_script") as mock_run_script, \
         patch("pathlib.Path.read_text") as mock_read_text:
        
        mock_env = MagicMock()
        mock_env_func.return_value = mock_env
        mock_read_text.return_value = "content"
        mock_env.from_string.return_value.render.return_value = "rendered"

        from cookiecutter.hooks.hooks import run_script_with_context
        run_script_with_context(script_path, str(tmp_path), context)

        # Check if the temporary file created ends with .sh
        args, _ = mock_run_script.call_args
        temp_file_path = args[0]
        assert temp_file_path.endswith(".sh")
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize("scripts, context, expected_calls", [
    ([], {"project_name": "test"}, 0),
    (["/abs/path/to/pre_gen_project.py"], {"project_name": "test"}, 1),
    (["/abs/path/to/post_gen_project.sh"], {"project_name": "test"}, 1),
])
def test_run_hook(scripts, context, expected_calls):
    """
    Test run_hook execution logic.
    Verifies that find_hook is called and run_script_with_context 
    is called the correct number of times.
    """
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"

    with patch("hooks.find_hook") as mock_find_hook, \
         patch("hooks.run_script_with_context") as mock_run_with_context:
        
        # Setup mock to return our list of scripts
        mock_find_hook.return_value = scripts
        
        run_hook(hook_name, project_dir, context)

        # Verify find_hook was called with correct name
        mock_find_hook.assert_called_once_with(hook_name)
        
        # Verify run_script_with_context was called for each found script
        assert mock_run_with_context.call_count == expected_calls
        if expected_calls > 0:
            mock_run_with_context.assert_any_call(scripts[0], project_dir, context)

def test_run_hook_empty_scripts_returns_early():
    """
    Test that run_hook returns early if no scripts are found.
    """
    hook_name = "non_existent_hook"
    project_dir = "/tmp/project"
    context = {}

    with patch("hooks.find_hook") as mock_find_hook, \
         patch("hooks.run_script_with_context") as mock_run_with_context:
        
        mock_find_hook.return_value = None
        
        run_hook(hook_name, project_dir, context)

        mock_find_hook.assert_called_once_with(hook_name)
        mock_run_with_context.assert_not_called()
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import os

@pytest.mark.parametrize(
    "hook_file, hook_name, expected",
    [
        ("pre_prompt.py", "pre_prompt", True),
        ("post_gen_project.sh", "post_gen_project", True),
        ("pre_gen_project.py", "pre_gen_project", True),
        ("unknown_hook.py", "unknown_hook", False),
        ("pre_prompt.py~", "pre_prompt", False),
        ("pre_prompt.bak", "pre_prompt", True),  # basename is pre_prompt, matches _HOOKS
        ("pre_prompt", "pre_prompt", True),
        ("not_a_hook.py", "pre_gen_project", False),
        ("pre_prompt.py.tmp", "pre_prompt", False), # basename is pre_prompt.py, not in _HOOKS
    ],
)
def test_valid_hook(hook_file, hook_name, expected):
    assert valid_hook(hook_file, hook_name) == expected
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

@pytest.mark.parametrize("should_fail,hook_exists", [
    (False, True),
    (False, False),
    (True, True),
])
def test_run_pre_prompt_hook(should_fail, hook_exists):
    """
    Tests run_pre_prompt_hook with various scenarios:
    1. No pre_prompt hook exists (returns original repo_dir).
    2. pre_prompt hook exists and runs successfully.
    3. pre_prompt hook exists but fails (raises FailedHookException).
    """
    repo_dir = "/fake/repo/dir"
    
    # Mocking dependencies
    with patch('os.path.isdir', return_value=True), \
         patch('cookiecutter.utils.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_dir:
        
        # Setup behavior for hook existence
        if hook_exists:
            mock_find_hook.side_effect = [
                ['/fake/repo/dir/hooks/pre_prompt.py'], # First call in original repo_dir
                ['/tmp/tmp_repo/hooks/pre_prompt.py']  # Second call in tmp_repo_dir
            ]
        else:
            mock_find_hook.return_value = None

        # Setup behavior for script execution
        if should_fail:
            mock_run_script.side_effect = FailedHookException("Hook failed")
        else:
            mock_run_script.return_value = None

        # Setup tmp repo dir path
        tmp_repo_dir = "/tmp/tmp_repo"
        mock_create_tmp_dir.return_value = tmp_repo_dir

        # Execution
        if not hook_exists:
            result = run_pre_prompt_hook(repo_dir)
            assert result == repo_dir
            assert mock_create_tmp_dir.call_count == 0
        
        elif not should_fail:
            result = run_pre_prompt_hook(repo_dir)
            assert result == tmp_repo_dir
            mock_run_script.assert_called_once()
            
        elif should_fail:
            with pytest.raises(FailedHookException) as excinfo:
                run_pre_prompt_hook(repo_dir)
            assert "Pre-Prompt Hook script failed" in str(excinfo.value)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock
from cookiecutter.exceptions import FailedHookException

def test_run_script(tmp_path):
    """Test the run_script function with various scenarios."""
    
    # 1. Test successful execution of a Python script
    py_script = tmp_path / "test_hook.py"
    py_script.write_text("import sys; sys.exit(0)")
    
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        run_script(str(py_script), cwd=str(tmp_path))
        
        # Check if it called python executable
        args, kwargs = mock_popen.call_args
        assert args[0][0] == sys.executable
        assert args[0][1] == str(py_script)
        assert kwargs["cwd"] == str(tmp_path)

    # 2. Test successful execution of a shell script (non-python)
    sh_script = tmp_path / "test_hook.sh"
    sh_script.write_text("#!/bin/bash\nexit 0")
    
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        run_script(str(sh_script), cwd=str(tmp_path))
        
        # Check if it called the script directly
        args, kwargs = mock_popen.call_args
        assert args[0][0] == str(sh_script)

    # 3. Test failure via non-zero exit status
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1  # Non-zero exit
        mock_popen.return_value = mock_proc
        
        with pytest.raises(FailedHookException, match="Hook script failed \(exit status: 1\)"):
            run_script(str(py_script), cwd=str(tmp_path))

    # 4. Test failure via OSError (e.g., file not found or permission denied)
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOENT, "No such file or directory")
        
        with pytest.raises(FailedHookException, match="Hook script failed \(error:"):
            run_script(str(py_script), cwd=str(tmp_path))

    # 5. Test failure via ENOEXEC (e.g., empty file or missing shebang)
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, "Exec format error")
        
        with pytest.raises(FailedHookException, match="might be an empty file or missing a shebang"):
            run_script(str(py_script), cwd=str(tmp_path))
```


